"""HTTP routes for Part 4.

* ``POST /chat`` - main streaming endpoint (Server-Sent Events).
* ``GET  /health`` - lightweight liveness probe.

The ``/chat`` handler turns LangGraph's tagged stream into SSE events via
``app.api.sse.langgraph_stream_to_sse`` and emits a final ``done`` (or
``error``) frame stamped with the total elapsed time. The agent runner is
a singleton created at app startup (see ``app.api.main``); both routes
receive it via FastAPI's ``Depends`` mechanism.
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Iterator

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from app.agent import AgentError, AgentRunner
from app.api.schemas import ChatRequest, HealthResponse
from app.api.sse import (
    StreamEventType,
    format_sse,
    langgraph_stream_to_sse,
)
from app.core.config import settings
from app.llm.errors import LLMClientError
from app.rag.errors import RAGError

logger = logging.getLogger(__name__)

router = APIRouter()


def get_runner(request: Request) -> AgentRunner:
    """FastAPI dependency that returns the app-wide ``AgentRunner`` singleton."""
    runner = getattr(request.app.state, "runner", None)
    if runner is None:
        # Should never happen if the lifespan ran; surfacing it loudly is the right move.
        raise RuntimeError(
            "AgentRunner not initialised. Check the FastAPI lifespan handler."
        )
    return runner


def _build_initial_state(query: str) -> dict[str, Any]:
    """Construct the LangGraph initial state for a single query."""
    return {
        "query": query.strip(),
        "route": None,
        "retrieved": None,
        "answer": None,
        "tool_calls": [],
        "trace": [],
    }


def _sync_event_stream(
    runner: AgentRunner, query: str, started_perf: float
) -> Iterator[dict[str, str]]:
    """Synchronous generator that yields SSE-ready dicts for one query.

    Runs the LangGraph pipeline with ``stream_mode=["updates","custom"]``,
    converts each tagged tuple into an SSE event via the adapter, and
    finally emits a single ``done`` frame with the full trace.
    """
    final_state: dict[str, Any] = {}
    try:
        graph_stream = runner.graph.stream(
            _build_initial_state(query),
            stream_mode=["updates", "custom"],
        )

        for event in langgraph_stream_to_sse(
            graph_stream, started_perf=started_perf
        ):
            sentinel = event.get("__final_state__")  # type: ignore[arg-type]
            if sentinel is not None:
                final_state = sentinel  # type: ignore[assignment]
                continue
            yield event

    except (AgentError, RAGError, LLMClientError) as exc:
        logger.exception("Agent layer raised during /chat")
        yield format_sse(
            StreamEventType.ERROR,
            {
                "kind": type(exc).__name__,
                "message": str(exc),
                "elapsed_ms": round(
                    (time.perf_counter() - started_perf) * 1000.0, 2
                ),
            },
        )
        return

    total_elapsed_ms = round((time.perf_counter() - started_perf) * 1000.0, 2)
    yield format_sse(
        StreamEventType.DONE,
        {
            "route": final_state.get("route"),
            "answer": final_state.get("answer") or "",
            "tool_calls": [
                {
                    "name": tc.name,
                    "input": tc.input,
                    "output": tc.output,
                    "elapsed_ms": tc.elapsed_ms,
                    "error": tc.error,
                }
                for tc in (final_state.get("tool_calls") or [])
            ],
            "trace": [
                {
                    "node": ev.node,
                    "timestamp": ev.timestamp,
                    "elapsed_ms": ev.elapsed_ms,
                    "inputs": ev.inputs,
                    "outputs": ev.outputs,
                }
                for ev in (final_state.get("trace") or [])
            ],
            "total_elapsed_ms": total_elapsed_ms,
        },
    )


async def _async_event_stream(
    runner: AgentRunner, query: str, started_perf: float
) -> AsyncIterator[dict[str, str]]:
    """Async wrapper that yields the sync generator's items off the event loop.

    ``OllamaClient.generate_stream`` uses blocking ``requests`` I/O, so we
    bridge through Starlette's ``iterate_in_threadpool`` to keep the event
    loop responsive (SSE keepalive pings, client-disconnect detection).
    """
    sync_iter = _sync_event_stream(runner, query, started_perf)
    async for event in iterate_in_threadpool(sync_iter):
        yield event


@router.post("/chat")
async def chat(
    payload: ChatRequest,
    runner: AgentRunner = Depends(get_runner),
) -> EventSourceResponse:
    """Stream the agent's response token-by-token via Server-Sent Events.

    Returns a ``text/event-stream`` response with the four event types
    documented in :class:`app.api.sse.StreamEventType`.
    """
    started_perf = time.perf_counter()
    return EventSourceResponse(
        _async_event_stream(runner, payload.query, started_perf),
        ping=settings.sse_keepalive_seconds,
    )


@router.get("/health", response_model=HealthResponse)
async def health(
    runner: AgentRunner = Depends(get_runner),
) -> HealthResponse:
    """Liveness probe - reports Ollama reachability and retriever readiness."""
    ollama_alive = runner.llm.health_check()
    retriever_ready = runner.retriever.is_ready
    chunks = runner.retriever.store.size if retriever_ready else 0
    return HealthResponse(
        status="ok" if (ollama_alive and retriever_ready) else "degraded",
        ollama_alive=ollama_alive,
        retriever_ready=retriever_ready,
        retriever_chunks=chunks,
        ollama_model=runner.llm.model,
        embed_model=settings.embed_model_name,
    )
