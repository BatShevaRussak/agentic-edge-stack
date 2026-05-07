"""LangGraph nodes implementing the agent's behaviour.

Each node is a plain ``(state) -> partial_state`` function. Dependencies
(LLM, retriever) are injected by ``app.agent.graph`` via ``functools.partial``.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable

from langgraph.config import get_stream_writer

from app.agent.errors import ToolExecutionError
from app.agent.prompts import (
    FALLBACK_MESSAGE,
    build_direct_prompt,
    build_router_prompt,
    parse_route,
    quick_classify,
)
from app.agent.tools import rag_search
from app.agent.types import AgentState, ToolCall, TraceEvent
from app.llm.errors import LLMClientError
from app.llm.ollama_client import OllamaClient
from app.rag.errors import RAGError
from app.rag.prompt_builder import build_rag_prompt
from app.rag.types import RetrievalHit, RetrievalResult

logger = logging.getLogger(__name__)


def _stream_llm_tokens(
    *,
    llm: OllamaClient,
    prompt: str,
    node_name: str,
    writer: Callable[[dict[str, Any]], None],
) -> tuple[str, str | None]:
    """Run ``llm.generate_stream`` while pushing each token to the writer.

    Returns ``(answer, error_message)``. The writer is a no-op when the graph
    is invoked outside a stream context (e.g. ``AgentRunner.run``); inside
    ``graph.stream(stream_mode=["updates","custom"])`` each token surfaces as
    a custom-mode payload that the API translates to an SSE ``token`` event.
    """
    chunks: list[str] = []
    try:
        for token in llm.generate_stream(prompt):
            if not token:
                continue
            chunks.append(token)
            writer(
                {
                    "type": "token",
                    "node": node_name,
                    "value": token,
                }
            )
    except LLMClientError as exc:
        partial = "".join(chunks).strip()
        msg = f"[LLM error during {node_name}: {exc}]"
        return (f"{partial}\n{msg}".strip() if partial else msg, str(exc))
    return ("".join(chunks).strip(), None)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_event(
    *,
    node: str,
    started_at: str,
    started_perf: float,
    inputs: dict[str, Any],
    outputs: dict[str, Any],
) -> TraceEvent:
    """Build a ``TraceEvent`` and stamp it with elapsed time."""
    elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
    return TraceEvent(
        node=node,
        timestamp=started_at,
        elapsed_ms=round(elapsed_ms, 2),
        inputs=inputs,
        outputs={**outputs, "elapsed_ms": round(elapsed_ms, 2)},
    )


def _serialize_hit(hit: RetrievalHit) -> dict[str, Any]:
    return {
        "score": round(hit.score, 4),
        "source": hit.chunk.source,
        "chunk_index": hit.chunk.chunk_index,
        "text": hit.chunk.text,
    }


def _serialize_retrieval(result: RetrievalResult) -> dict[str, Any]:
    return {
        "query": result.query,
        "hits": [_serialize_hit(h) for h in result.hits],
    }


# --- Nodes -----------------------------------------------------------------


def router_node(state: AgentState, *, llm: OllamaClient) -> dict[str, Any]:
    """Classify the query as ``rag`` or ``direct`` (heuristic, then LLM)."""
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    query = state["query"]

    heuristic_route = quick_classify(query)
    if heuristic_route is not None:
        event = _make_event(
            node="router",
            started_at=started_at,
            started_perf=started_perf,
            inputs={"query": query},
            outputs={"method": "heuristic", "decided_route": heuristic_route},
        )
        return {"route": heuristic_route, "trace": [event]}

    prompt = build_router_prompt(query)
    try:
        raw = llm.generate(prompt)
    except LLMClientError as exc:
        logger.warning("router LLM failed (%s); defaulting to RAG", exc)
        raw = ""

    route = parse_route(raw)
    event = _make_event(
        node="router",
        started_at=started_at,
        started_perf=started_perf,
        inputs={"query": query, "prompt_chars": len(prompt)},
        outputs={
            "method": "llm",
            "raw_output": raw.strip() if raw else "",
            "decided_route": route,
        },
    )
    return {"route": route, "trace": [event]}


def rag_node(state: AgentState) -> dict[str, Any]:
    """Invoke ``rag_search`` and record both a ``ToolCall`` and a ``TraceEvent``."""
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    query = state["query"]

    try:
        result: RetrievalResult = rag_search.invoke({"query": query})
    except RAGError as exc:
        raise ToolExecutionError(f"rag_search failed: {exc}") from exc

    elapsed_ms = (time.perf_counter() - started_perf) * 1000.0
    tool_call = ToolCall(
        name="rag_search",
        input={"query": query},
        output=_serialize_retrieval(result),
        elapsed_ms=round(elapsed_ms, 2),
    )
    event = _make_event(
        node="rag_search",
        started_at=started_at,
        started_perf=started_perf,
        inputs={"query": query},
        outputs={
            "hits_count": len(result.hits),
            "top_score": result.hits[0].score if result.hits else None,
        },
    )
    return {
        "retrieved": result,
        "tool_calls": [tool_call],
        "trace": [event],
    }


def synthesis_node(state: AgentState, *, llm: OllamaClient) -> dict[str, Any]:
    """Generate the final answer from retrieved context (re-uses ``build_rag_prompt``).

    Streams tokens through LangGraph's custom-stream channel so the API layer
    can push them to the client as SSE ``token`` events. When invoked
    synchronously (``AgentRunner.run``) the writer is a no-op, so the same
    code powers both the Part 3 batch trace and the Part 4 streaming API.
    """
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    result: RetrievalResult | None = state.get("retrieved")
    if result is None:
        event = _make_event(
            node="synthesis",
            started_at=started_at,
            started_perf=started_perf,
            inputs={"reason": "no_retrieval_result"},
            outputs={"answer": FALLBACK_MESSAGE},
        )
        return {"answer": FALLBACK_MESSAGE, "trace": [event]}

    prompt = build_rag_prompt(result)
    writer = get_stream_writer()
    answer, error = _stream_llm_tokens(
        llm=llm, prompt=prompt, node_name="synthesis", writer=writer
    )

    event = _make_event(
        node="synthesis",
        started_at=started_at,
        started_perf=started_perf,
        inputs={"prompt_chars": len(prompt), "num_hits": len(result.hits)},
        outputs={"answer_chars": len(answer), "error": error},
    )
    return {"answer": answer, "trace": [event]}


def direct_node(state: AgentState, *, llm: OllamaClient) -> dict[str, Any]:
    """Generate an answer from the LLM's parametric knowledge alone.

    Same streaming contract as ``synthesis_node`` (see its docstring).
    """
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    query = state["query"]

    prompt = build_direct_prompt(query)
    writer = get_stream_writer()
    answer, error = _stream_llm_tokens(
        llm=llm, prompt=prompt, node_name="direct", writer=writer
    )

    event = _make_event(
        node="direct",
        started_at=started_at,
        started_perf=started_perf,
        inputs={"query": query, "prompt_chars": len(prompt)},
        outputs={"answer_chars": len(answer), "error": error},
    )
    return {"answer": answer, "trace": [event]}


def fallback_node(state: AgentState) -> dict[str, Any]:
    """Return the canonical "no information" response without calling the LLM."""
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    event = _make_event(
        node="fallback",
        started_at=started_at,
        started_perf=started_perf,
        inputs={"reason": "no_hits_above_threshold"},
        outputs={"answer": FALLBACK_MESSAGE},
    )
    return {"answer": FALLBACK_MESSAGE, "trace": [event]}


__all__ = [
    "router_node",
    "rag_node",
    "synthesis_node",
    "direct_node",
    "fallback_node",
]
