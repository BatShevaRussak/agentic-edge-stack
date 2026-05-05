"""End-to-end verification of the in-memory RAG pipeline (Part 2).

Runs a fixed set of demo queries through the full pipeline and writes a
per-query trace (query, retrieved chunks, augmented prompt, LLM answer)
to ``tests/logs/rag_run_<UTC timestamp>.txt``.

Usage:
    python tests/verify_rag.py
    python tests/verify_rag.py --skip-llm
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.llm.errors import LLMClientError
from app.llm.factory import get_llm_client
from app.rag.errors import RAGError
from app.rag.prompt_builder import build_rag_prompt
from app.rag.retriever import Retriever
from app.rag.types import RetrievalResult

DEMO_QUERIES: tuple[str, ...] = (
    "What quantization format is the deployed Llama 3.2 model shipped in, "
    "and what is the size/quality tradeoff?",
    "How does FAISS compute cosine similarity, and why does this project "
    "use IndexFlatIP?",
    "Why was BGE-small chosen over all-MiniLM-L6-v2 as the embedding model?",
    "What chunking strategy does this RAG pipeline use, and what chunk "
    "size and overlap are configured?",
    "What is the role of the Modelfile when running Llama 3.2 on Ollama?",
    "What is the recommended recipe for a chocolate cake?",  # out-of-domain
)


class Trace:
    """Tee-style writer that prints to stdout and buffers for a log file."""

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

    def subsection(self, title: str) -> None:
        self.write("")
        self.write("-" * 70)
        self.write(title)
        self.write("-" * 70)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self._buffer) + "\n", encoding="utf-8")


def _format_chunk_excerpt(text: str, max_chars: int = 240) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= max_chars else flat[: max_chars - 3] + "..."


def _log_retrieval(trace: Trace, result: RetrievalResult) -> None:
    if not result.hits:
        trace.write(
            "(no chunks passed the score threshold of "
            f"{settings.rag_score_threshold})"
        )
        return
    for rank, hit in enumerate(result.hits, start=1):
        trace.write(
            f"  #{rank}  score={hit.score:.4f}  "
            f"source={hit.chunk.source}  "
            f"chunk={hit.chunk.chunk_index}"
        )
        trace.write(f"        {_format_chunk_excerpt(hit.chunk.text)}")


def _run_query(
    trace: Trace,
    retriever: Retriever,
    query: str,
    skip_llm: bool,
) -> None:
    trace.subsection(f"QUERY: {query}")

    try:
        start = time.perf_counter()
        result = retriever.retrieve(query)
        retrieve_seconds = time.perf_counter() - start
    except RAGError as exc:
        trace.write(f"[FAIL] retrieval error: {exc}")
        return

    trace.write(
        f"Retrieved {len(result.hits)} chunks in "
        f"{retrieve_seconds * 1000:.1f} ms"
    )
    _log_retrieval(trace, result)

    prompt = build_rag_prompt(result)
    trace.write("")
    trace.write("AUGMENTED PROMPT (sent to LLM):")
    for line in prompt.splitlines():
        trace.write(f"  | {line}")

    if skip_llm:
        trace.write("")
        trace.write("(LLM call skipped via --skip-llm)")
        return

    try:
        client = get_llm_client()
        start = time.perf_counter()
        answer = client.generate(prompt)
        gen_seconds = time.perf_counter() - start
    except LLMClientError as exc:
        trace.write(f"[FAIL] LLM error: {exc}")
        return

    trace.write("")
    trace.write(f"LLM ANSWER ({gen_seconds:.2f}s):")
    for line in answer.strip().splitlines():
        trace.write(f"  > {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run retrieval only, do not call Ollama.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force a fresh ingestion (ignore on-disk index cache).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("tests/logs"),
        help="Directory for the run log (default: tests/logs).",
    )
    args = parser.parse_args(argv)

    trace = Trace()
    trace.section("RAG PIPELINE VERIFICATION")
    trace.write(f"Timestamp:        {datetime.now(timezone.utc).isoformat()}")
    trace.write(f"Embed model:      {settings.embed_model_name}")
    trace.write(f"Embed dimension:  {settings.embed_dim}")
    trace.write(f"Chunk size:       {settings.rag_chunk_size}")
    trace.write(f"Chunk overlap:    {settings.rag_chunk_overlap}")
    trace.write(f"Top-K:            {settings.rag_top_k}")
    trace.write(f"Score threshold:  {settings.rag_score_threshold}")
    trace.write(f"Ollama model:     {settings.ollama_model}")
    trace.write(f"Skip LLM:         {args.skip_llm}")

    retriever = Retriever()

    trace.section("INGESTION")
    try:
        start = time.perf_counter()
        num_chunks = retriever.ingest_directory(
            settings.data_dir, use_cache=not args.no_cache
        )
        elapsed = time.perf_counter() - start
    except RAGError as exc:
        trace.write(f"[FAIL] {exc}")
        _save_log(trace, args.log_dir)
        return 1

    trace.write(
        f"[OK] Indexed {num_chunks} chunks in {elapsed:.2f}s "
        f"(cache_dir={settings.cache_dir})"
    )

    trace.section("DEMO QUERIES")
    for query in DEMO_QUERIES:
        _run_query(trace, retriever, query, skip_llm=args.skip_llm)

    trace.section("DONE")
    log_path = _save_log(trace, args.log_dir)
    print(f"\nFull trace saved to: {log_path}")
    return 0


def _save_log(trace: Trace, log_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"rag_run_{timestamp}.txt"
    trace.save(log_path)
    return log_path


if __name__ == "__main__":
    sys.exit(main())
