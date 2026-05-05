"""High-level RAG retriever - orchestrates chunker, embedder, vector store."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.rag.chunker import Chunker
from app.rag.embeddings import Embedder
from app.rag.errors import EmptyCorpusError, IngestionError, RetrievalError
from app.rag.types import RetrievalHit, RetrievalResult
from app.rag.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"
_DEFAULT_GLOB_PATTERNS: tuple[str, ...] = ("*.md", "*.txt")


class Retriever:
    """Coordinates ingestion and retrieval over a single corpus.

    Embeddings are cached on disk under ``cache_dir``; the cache is keyed
    on a SHA-256 of the corpus contents plus the embedding model name and
    dimension, so any edit invalidates it automatically.
    """

    def __init__(
        self,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        cache_dir: Path | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.chunker = chunker or Chunker()
        self.embedder = embedder or Embedder()
        self.cache_dir = cache_dir or settings.cache_dir
        self.top_k = top_k if top_k is not None else settings.rag_top_k
        self.score_threshold = (
            score_threshold
            if score_threshold is not None
            else settings.rag_score_threshold
        )
        self._store: VectorStore | None = None

    @property
    def store(self) -> VectorStore:
        """The active vector store. Raises if ingestion has not run."""
        if self._store is None:
            raise RetrievalError(
                "Retriever has no index. Call ingest_directory(...) first."
            )
        return self._store

    @property
    def is_ready(self) -> bool:
        """True if an index is loaded and queryable."""
        return self._store is not None and self._store.size > 0

    def ingest_directory(
        self,
        directory: Path,
        patterns: tuple[str, ...] = _DEFAULT_GLOB_PATTERNS,
        use_cache: bool = True,
    ) -> int:
        """Ingest matching files from a directory and return chunk count.

        Raises:
            IngestionError: if the directory cannot be read.
            EmptyCorpusError: if no matching files exist.
        """
        if not directory.exists() or not directory.is_dir():
            raise IngestionError(f"Corpus directory not found: {directory}")

        files = sorted(
            {p for pattern in patterns for p in directory.glob(pattern)}
        )
        if not files:
            raise EmptyCorpusError(
                f"No files matching {patterns} under {directory}"
            )

        corpus_hash = _hash_files(files)
        manifest_path = self.cache_dir / _MANIFEST_FILENAME

        if use_cache and self._cache_is_valid(manifest_path, corpus_hash):
            logger.info("Loading vector index from cache: %s", self.cache_dir)
            self._store = VectorStore.load(self.cache_dir)
            return self._store.size

        logger.info("Ingesting %d files from %s", len(files), directory)
        chunks = []
        for path in files:
            chunks.extend(self.chunker.split_file(path))

        if not chunks:
            raise EmptyCorpusError(
                f"All files in {directory} produced zero chunks"
            )

        logger.info(
            "Embedding %d chunks with %s (dim=%d)",
            len(chunks),
            self.embedder.model_name,
            self.embedder.dim,
        )
        start = time.perf_counter()
        vectors = self.embedder.encode_documents(
            [chunk.text for chunk in chunks]
        )
        embed_seconds = time.perf_counter() - start
        logger.info(
            "Embedded %d chunks in %.2fs (%.1f chunks/s)",
            len(chunks),
            embed_seconds,
            len(chunks) / max(embed_seconds, 1e-6),
        )

        store = VectorStore(dim=self.embedder.dim)
        store.add(vectors, chunks)

        store.save(self.cache_dir)
        _write_manifest(
            manifest_path,
            corpus_hash=corpus_hash,
            embed_model=self.embedder.model_name,
            embed_dim=self.embedder.dim,
            num_chunks=len(chunks),
            files=[p.as_posix() for p in files],
        )

        self._store = store
        return store.size

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """Retrieve top-K chunks for a query, dropping hits below threshold.

        Raises:
            RetrievalError: on empty query or missing/empty index.
        """
        if not query or not query.strip():
            raise RetrievalError("Query must be a non-empty string")

        k = top_k if top_k is not None else self.top_k
        threshold = (
            score_threshold
            if score_threshold is not None
            else self.score_threshold
        )

        try:
            query_vector = self.embedder.encode_query(query)
            hits = self.store.search(query_vector, top_k=k)
        except VectorStoreError as exc:
            raise RetrievalError(str(exc)) from exc

        kept: list[RetrievalHit] = [h for h in hits if h.score >= threshold]
        return RetrievalResult(query=query, hits=kept)

    def _cache_is_valid(self, manifest_path: Path, expected_hash: str) -> bool:
        if not manifest_path.exists():
            return False
        try:
            with manifest_path.open("r", encoding="utf-8") as fh:
                manifest = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return False

        return (
            manifest.get("corpus_hash") == expected_hash
            and manifest.get("embed_model") == self.embedder.model_name
            and manifest.get("embed_dim") == self.embedder.dim
        )


def _hash_files(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.as_posix().encode("utf-8"))
        hasher.update(b"\0")
        try:
            hasher.update(path.read_bytes())
        except OSError as exc:
            raise IngestionError(f"Failed to hash {path}: {exc}") from exc
        hasher.update(b"\0")
    return hasher.hexdigest()


def _write_manifest(
    path: Path,
    *,
    corpus_hash: str,
    embed_model: str,
    embed_dim: int,
    num_chunks: int,
    files: list[str],
) -> None:
    payload = {
        "corpus_hash": corpus_hash,
        "embed_model": embed_model,
        "embed_dim": embed_dim,
        "num_chunks": num_chunks,
        "files": files,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
