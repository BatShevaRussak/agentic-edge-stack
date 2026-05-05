"""Build (or refresh) the FAISS index from the corpus on disk.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --data-dir data --no-cache
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from app.core.config import settings
from app.rag.errors import RAGError
from app.rag.retriever import Retriever


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=settings.data_dir,
        help=f"Corpus directory (default: {settings.data_dir})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=settings.cache_dir,
        help=f"On-disk cache directory (default: {settings.cache_dir})",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-embedding even if a valid cache exists",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print("=" * 60)
    print("RAG corpus ingestion")
    print("=" * 60)
    print(f"Data dir:       {args.data_dir}")
    print(f"Cache dir:      {args.cache_dir}")
    print(f"Embed model:    {settings.embed_model_name}")
    print(f"Chunk size:     {settings.rag_chunk_size}")
    print(f"Chunk overlap:  {settings.rag_chunk_overlap}")
    print(f"Use cache:      {not args.no_cache}")
    print()

    retriever = Retriever(cache_dir=args.cache_dir)

    try:
        start = time.perf_counter()
        num_chunks = retriever.ingest_directory(
            args.data_dir, use_cache=not args.no_cache
        )
        elapsed = time.perf_counter() - start
    except RAGError as exc:
        print(f"[FAIL] Ingestion failed: {exc}", file=sys.stderr)
        return 1

    print(f"[OK] Indexed {num_chunks} chunks in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
