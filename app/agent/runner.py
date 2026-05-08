"""Public entry point for the agent + trace formatters.

``AgentRunner`` owns the LLM client, retriever, and compiled graph.
``format_trace_text`` and ``format_trace_json`` convert an
``AgentResponse`` into the two log artefacts shipped under ``tests/logs/``.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from app.agent.errors import AgentError
from app.agent.graph import build_agent_graph
from app.agent.types import AgentResponse, AgentState
from app.core.config import settings
from app.llm.factory import get_llm_client
from app.llm.ollama_client import OllamaClient
from app.rag.retriever import Retriever


class AgentRunner:
    """Runs the LangGraph agent over a single corpus.

    Ingests once (re-using Part 2's on-disk cache when possible) and
    compiles the graph once; both are reused across queries.
    """

    def __init__(
        self,
        retriever: Retriever | None = None,
        llm: OllamaClient | None = None,
    ) -> None:
        self._llm = llm or get_llm_client()
        if retriever is None:
            retriever = Retriever()
            retriever.ingest_directory(settings.data_dir, use_cache=True)
        self._retriever = retriever
        self._graph = build_agent_graph(llm=self._llm, retriever=self._retriever)

    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def llm(self) -> OllamaClient:
        return self._llm

    @property
    def graph(self) -> CompiledStateGraph[AgentState, Any, Any, Any]:
        """The compiled LangGraph (streaming via ``.stream``)."""
        return self._graph

    def run(self, query: str) -> AgentResponse:
        """Run the agent on a single query.

        Raises:
            AgentError: on empty query or if the graph terminates without
                producing a routing decision.
        """
        if not query or not query.strip():
            raise AgentError("Query must be a non-empty string")

        started_perf = time.perf_counter()
        initial_state: AgentState = {
            "query": query.strip(),
            "route": None,
            "retrieved": None,
            "answer": None,
            "tool_calls": [],
            "trace": [],
        }
        final_state: dict[str, Any] = self._graph.invoke(initial_state)
        total_elapsed_ms = (time.perf_counter() - started_perf) * 1000.0

        route = final_state.get("route")
        if route is None:
            raise AgentError("Agent graph terminated without a routing decision")

        return AgentResponse(
            query=query.strip(),
            route=route,
            answer=final_state.get("answer") or "",
            tool_calls=list(final_state.get("tool_calls", [])),
            trace=list(final_state.get("trace", [])),
            total_elapsed_ms=round(total_elapsed_ms, 2),
        )


# --- Trace formatting ------------------------------------------------------


def _format_chunk_excerpt(text: str, max_chars: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[: max_chars - 3] + "..."


def format_trace_text(response: AgentResponse) -> str:
    """Render an ``AgentResponse`` as a multi-line human-readable trace."""
    lines: list[str] = []
    lines.append("-" * 70)
    lines.append(f"QUERY: {response.query}")
    lines.append("-" * 70)
    lines.append(f"ROUTE: {response.route.upper()}")

    for event in response.trace:
        lines.append("")
        lines.append(f"[{event.node}] elapsed={event.elapsed_ms:.1f} ms")

        if event.node == "router":
            method = event.outputs.get("method", "llm")
            decided = event.outputs.get("decided_route", "")
            lines.append(f"         method={method}")
            if method == "llm":
                raw = event.outputs.get("raw_output", "")
                lines.append(f'         raw_output="{raw}"')
            lines.append(f"         decided_route={decided}")

        elif event.node == "rag_search":
            tc = next(
                (c for c in response.tool_calls if c.name == "rag_search"), None
            )
            if tc and tc.output:
                hits = tc.output.get("hits", [])
                lines.append(f'         query="{tc.input.get("query", "")}"')
                lines.append(f"         hits={len(hits)}")
                for rank, hit in enumerate(hits, start=1):
                    lines.append(
                        f"         #{rank}  score={hit['score']:.4f}  "
                        f"source={hit['source']}  "
                        f"chunk={hit['chunk_index']}"
                    )
                    lines.append(
                        f"               {_format_chunk_excerpt(hit['text'])}"
                    )

        elif event.node == "synthesis":
            prompt_chars = event.inputs.get("prompt_chars", 0)
            num_hits = event.inputs.get("num_hits", 0)
            answer_chars = event.outputs.get("answer_chars", 0)
            lines.append(
                f"         prompt={prompt_chars} chars  "
                f"context_hits={num_hits}  "
                f"answer={answer_chars} chars"
            )

        elif event.node == "direct":
            prompt_chars = event.inputs.get("prompt_chars", 0)
            answer_chars = event.outputs.get("answer_chars", 0)
            lines.append(
                f"         prompt={prompt_chars} chars  "
                f"answer={answer_chars} chars"
            )

        elif event.node == "fallback":
            lines.append("         (no LLM call - canonical no-context response)")

    lines.append("")
    lines.append("ANSWER:")
    for ans_line in response.answer.splitlines() or [""]:
        lines.append(f"  > {ans_line}")
    lines.append("")
    lines.append(f"TOTAL: {response.total_elapsed_ms:.1f} ms")
    return "\n".join(lines)


def format_trace_json(response: AgentResponse) -> dict[str, Any]:
    """Render an ``AgentResponse`` as a JSON-serialisable dict."""
    return {
        "query": response.query,
        "route": response.route,
        "answer": response.answer,
        "total_elapsed_ms": response.total_elapsed_ms,
        "tool_calls": [asdict(tc) for tc in response.tool_calls],
        "trace": [asdict(ev) for ev in response.trace],
    }
