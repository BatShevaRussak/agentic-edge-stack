"""End-to-end verification of the agent (Part 3).

Runs 7 demo queries (3 RAG, 3 DIRECT, 1 OOD->fallback) and writes both
``agent_run_<UTC>.txt`` and ``agent_run_<UTC>.json`` under ``tests/logs/``.
The query mix exercises every edge of the LangGraph.

Usage:
    python tests/verify_agent.py
    python tests/verify_agent.py --no-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.agent import (
    AgentError,
    AgentResponse,
    AgentRunner,
    format_trace_json,
    format_trace_text,
)
from app.agent.graph import render_mermaid
from app.core.config import settings
from app.rag.errors import RAGError
from app.rag.retriever import Retriever


DEMO_QUERIES: tuple[tuple[str, str], ...] = (
    (
        "What chunk size does this RAG pipeline use, and what is the overlap?",
        "rag",
    ),
    (
        "Why was BGE-small chosen over all-MiniLM-L6-v2?",
        "rag",
    ),
    (
        "How does FAISS compute cosine similarity in this project?",
        "rag",
    ),
    (
        "What is 17 multiplied by 24?",
        "direct",
    ),
    (
        "Translate 'good morning' to Spanish.",
        "direct",
    ),
    (
        "Who are you?",
        "direct",
    ),
    (
        "What is the recommended recipe for a chocolate cake?",
        "rag",  # router routes to RAG; RAG returns 0 hits; fallback fires.
    ),
)


class Trace:
    """Tee writer: prints to stdout while buffering for a log file."""

    def __init__(self) -> None:
        self._buffer: list[str] = []

    def write(self, line: str = "") -> None:
        print(line)
        self._buffer.append(line)

    def section(self, title: str) -> None:
        self.write("")
        self.write("=" * 70)
        self.write(title)
        self.write("=" * 70)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._buffer) + "\n", encoding="utf-8")


def _save_log(trace: Trace, log_dir: Path, timestamp: str) -> Path:
    log_path = log_dir / f"agent_run_{timestamp}.txt"
    trace.save(log_path)
    return log_path


def _save_json(
    log_dir: Path,
    timestamp: str,
    payload: dict,
) -> Path:
    log_path = log_dir / f"agent_run_{timestamp}.json"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return log_path


def _run_one(
    trace: Trace,
    runner: AgentRunner,
    query: str,
    expected_route: str,
) -> AgentResponse | None:
    """Run one query and append its trace section to the log."""
    try:
        response = runner.run(query)
    except AgentError as exc:
        trace.write("")
        trace.write(f"[FAIL] agent error: {exc}")
        return None

    trace.write("")
    trace.write(format_trace_text(response))

    actual = response.route
    status = "OK" if actual == expected_route else "WARN"
    trace.write(
        f"[{status}] expected_route={expected_route}  actual_route={actual}"
    )
    return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a fresh ingestion (ignore on-disk index cache).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("tests/logs"),
        help="Directory for run logs (default: tests/logs).",
    )
    args = parser.parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    trace = Trace()
    trace.section("AGENT PIPELINE VERIFICATION (Part 3)")
    trace.write(f"Timestamp:        {datetime.now(timezone.utc).isoformat()}")
    trace.write(f"Embed model:      {settings.embed_model_name}")
    trace.write(f"Ollama model:     {settings.ollama_model}")
    trace.write(f"Top-K:            {settings.rag_top_k}")
    trace.write(f"Score threshold:  {settings.rag_score_threshold}")
    trace.write(f"Cache:            {'fresh' if args.no_cache else 'reuse'}")

    trace.section("INGESTION")
    try:
        start = time.perf_counter()
        retriever = Retriever()
        num_chunks = retriever.ingest_directory(
            settings.data_dir, use_cache=not args.no_cache
        )
        ingest_seconds = time.perf_counter() - start
    except RAGError as exc:
        trace.write(f"[FAIL] {exc}")
        _save_log(trace, args.log_dir, timestamp)
        return 1

    trace.write(
        f"[OK] Indexed {num_chunks} chunks in {ingest_seconds:.2f}s "
        f"(cache_dir={settings.cache_dir})"
    )

    trace.section("AGENT GRAPH")
    runner = AgentRunner(retriever=retriever)
    try:
        mermaid = render_mermaid(runner._graph)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        mermaid = f"(mermaid rendering failed: {exc})"
    trace.write("Compiled LangGraph topology (Mermaid):")
    for line in mermaid.splitlines():
        trace.write(f"  {line}")

    trace.section("DEMO QUERIES")
    responses: list[AgentResponse] = []
    for query, expected_route in DEMO_QUERIES:
        response = _run_one(trace, runner, query, expected_route)
        if response is not None:
            responses.append(response)

    trace.section("SUMMARY")
    expected_routes = {q: r for q, r in DEMO_QUERIES}
    rag_count = sum(1 for r in responses if r.route == "rag")
    direct_count = sum(1 for r in responses if r.route == "direct")
    correct_count = sum(
        1 for r in responses if expected_routes.get(r.query) == r.route
    )
    trace.write(f"Queries executed:    {len(responses)} / {len(DEMO_QUERIES)}")
    trace.write(f"Routed to RAG:       {rag_count}")
    trace.write(f"Routed to DIRECT:    {direct_count}")
    trace.write(f"Routes as expected:  {correct_count} / {len(responses)}")

    trace.section("DONE")
    log_path = _save_log(trace, args.log_dir, timestamp)
    json_path = _save_json(
        args.log_dir,
        timestamp,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "embed_model_name": settings.embed_model_name,
                "ollama_model": settings.ollama_model,
                "rag_top_k": settings.rag_top_k,
                "rag_score_threshold": settings.rag_score_threshold,
            },
            "ingest": {
                "num_chunks": num_chunks,
                "elapsed_seconds": round(ingest_seconds, 3),
            },
            "queries": [format_trace_json(r) for r in responses],
        },
    )

    print(f"\nFull trace saved to: {log_path}")
    print(f"JSON trace saved to: {json_path}")
    return 0 if len(responses) == len(DEMO_QUERIES) else 1


if __name__ == "__main__":
    sys.exit(main())
