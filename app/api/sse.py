"""SSE event protocol + LangGraph -> SSE adapter.

Defines the event types the API emits over the wire, plus a single
function (``langgraph_stream_to_sse``) that converts the *(mode, payload)*
tuples produced by ``graph.stream(stream_mode=["updates","custom"])`` into
a sequence of SSE-ready dicts.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Iterable, Iterator


class StreamEventType(str, Enum):
    """Event types emitted by ``POST /chat`` over Server-Sent Events.

    The sequence for a successful request is roughly:

    * ``route``       - one event after the router node decides.
    * ``tool_call``   - zero or one event when ``rag_search`` is invoked.
    * ``token``       - many events while the LLM is generating.
    * ``done``        - exactly one event with the full ``AgentResponse``-shaped trace.
    * ``error``       - emitted instead of ``done`` if a layer raises.
    """

    ROUTE = "route"
    TOOL_CALL = "tool_call"
    TOKEN = "token"
    DONE = "done"
    ERROR = "error"


def _to_jsonable(obj: Any) -> Any:
    """Best-effort coercion of agent-layer objects into JSON-safe primitives."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def format_sse(event_type: StreamEventType, data: dict[str, Any]) -> dict[str, str]:
    """Format a payload as the dict shape ``EventSourceResponse`` expects.

    Returns a ``{"event": ..., "data": ...}`` dict; ``sse-starlette`` turns
    that into the textual ``event: foo\\ndata: {...}\\n\\n`` SSE frame.
    """
    return {
        "event": event_type.value,
        "data": json.dumps(_to_jsonable(data), ensure_ascii=False),
    }


def langgraph_stream_to_sse(
    stream: Iterable[tuple[str, Any]],
    *,
    started_perf: float,
) -> Iterator[dict[str, str]]:
    """Map ``graph.stream(stream_mode=["updates","custom"])`` tuples to SSE dicts.

    ``updates``: router/rag deltas become ``route`` / ``tool_call`` events and
    merge into ``final_state``. ``custom``: token dicts from ``get_stream_writer``.
    The caller emits ``done`` with ``total_elapsed_ms``.
    """
    final_state: dict[str, Any] = {}

    for mode, payload in stream:
        elapsed_ms = round((time.perf_counter() - started_perf) * 1000.0, 2)

        if mode == "custom":
            yield format_sse(
                StreamEventType.TOKEN,
                {**payload, "elapsed_ms": elapsed_ms},
            )
            continue

        if mode != "updates" or not isinstance(payload, dict):
            continue

        # ``payload`` is ``{node_name: state_delta}``.
        for node_name, delta in payload.items():
            if not isinstance(delta, dict):
                continue
            _merge_state(final_state, delta)

            if node_name == "router":
                trace = delta.get("trace") or []
                outputs = trace[-1].outputs if trace else {}
                yield format_sse(
                    StreamEventType.ROUTE,
                    {
                        "route": delta.get("route"),
                        "method": outputs.get("method"),
                        "elapsed_ms": elapsed_ms,
                    },
                )

            elif node_name == "rag":
                tool_calls = delta.get("tool_calls") or []
                if tool_calls:
                    tc = tool_calls[-1]
                    output = tc.output or {}
                    hits = output.get("hits", [])
                    yield format_sse(
                        StreamEventType.TOOL_CALL,
                        {
                            "name": tc.name,
                            "input": tc.input,
                            "hits": len(hits),
                            "top_score": hits[0]["score"] if hits else None,
                            "tool_elapsed_ms": tc.elapsed_ms,
                            "elapsed_ms": elapsed_ms,
                        },
                    )

            elif node_name == "fallback":
                # Surface the canonical no-context message as a single token,
                # so a streaming client always sees text in its main pane.
                answer = delta.get("answer") or ""
                if answer:
                    yield format_sse(
                        StreamEventType.TOKEN,
                        {
                            "type": "token",
                            "node": "fallback",
                            "value": answer,
                            "elapsed_ms": elapsed_ms,
                        },
                    )

    # Stash the assembled state on a sentinel so the caller can build ``done``.
    yield {"__final_state__": final_state}  # type: ignore[dict-item]


def _merge_state(acc: dict[str, Any], delta: dict[str, Any]) -> None:
    """Merge a per-node state delta into the accumulator (append-only lists)."""
    for key, value in delta.items():
        if key in ("tool_calls", "trace") and isinstance(value, list):
            acc.setdefault(key, []).extend(value)
        else:
            acc[key] = value
