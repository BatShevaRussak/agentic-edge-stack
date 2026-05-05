"""In-memory vector store backed by FAISS ``IndexFlatIP``.

Inner product on L2-normalized vectors equals cosine similarity, so this
single index family covers exact cosine search at the scale of an
in-memory RAG corpus (tens to a few hundred chunks).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.rag.errors import RAGError
from app.rag.types import Chunk, RetrievalHit


class VectorStoreError(RAGError):
    """Raised when the FAISS index cannot be built, queried, or persisted."""


_INDEX_FILENAME = "index.faiss"
_CHUNKS_FILENAME = "chunks.json"


class VectorStore:
    """FAISS-backed vector store with chunk metadata."""

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")

        self.dim = dim
        self._index: faiss.Index = faiss.IndexFlatIP(dim)
        self._chunks: list[Chunk] = []

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    def add(self, vectors: np.ndarray, chunks: list[Chunk]) -> None:
        """Insert vectors and their associated chunks.

        Vectors must be L2-normalized so inner product behaves as cosine.

        Raises:
            VectorStoreError: on shape mismatch or non-normalized vectors.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise VectorStoreError(
                f"Expected vectors of shape (n, {self.dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                f"Vectors / chunks length mismatch: {vectors.shape[0]} vs {len(chunks)}"
            )
        if vectors.shape[0] == 0:
            return

        sample_norm = float(np.linalg.norm(vectors[0]))
        if not (0.99 <= sample_norm <= 1.01):
            raise VectorStoreError(
                "Vectors must be L2-normalized for IndexFlatIP to behave as "
                f"cosine similarity (got norm={sample_norm:.4f})"
            )

        self._index.add(vectors.astype(np.float32, copy=False))
        self._chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalHit]:
        """Return the ``top_k`` chunks most similar to the query vector.

        Raises:
            VectorStoreError: on bad query shape or empty index.
        """
        if self.size == 0:
            raise VectorStoreError("Cannot search an empty index")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape != (1, self.dim):
            raise VectorStoreError(
                f"Expected query of shape (1, {self.dim}), got {query_vector.shape}"
            )

        k = min(top_k, self.size)
        scores, ids = self._index.search(
            query_vector.astype(np.float32, copy=False), k
        )

        return [
            RetrievalHit(chunk=self._chunks[int(idx)], score=float(score))
            for score, idx in zip(scores[0], ids[0], strict=True)
            if idx != -1
        ]

    def save(self, directory: Path) -> None:
        """Persist the FAISS index and chunk metadata to a directory."""
        try:
            directory.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(directory / _INDEX_FILENAME))
            with (directory / _CHUNKS_FILENAME).open("w", encoding="utf-8") as fh:
                json.dump(
                    [asdict(chunk) for chunk in self._chunks],
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except (OSError, TypeError) as exc:
            raise VectorStoreError(
                f"Failed to save vector store to {directory}: {exc}"
            ) from exc

    @classmethod
    def load(cls, directory: Path) -> VectorStore:
        """Restore a previously saved store."""
        index_path = directory / _INDEX_FILENAME
        chunks_path = directory / _CHUNKS_FILENAME
        if not index_path.exists() or not chunks_path.exists():
            raise VectorStoreError(f"Missing index or chunks file under {directory}")

        try:
            index = faiss.read_index(str(index_path))
            with chunks_path.open("r", encoding="utf-8") as fh:
                chunk_dicts = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise VectorStoreError(
                f"Failed to load vector store from {directory}: {exc}"
            ) from exc

        store = cls(dim=index.d)
        store._index = index
        store._chunks = [Chunk(**data) for data in chunk_dicts]
        return store
