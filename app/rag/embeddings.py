"""Sentence embedding model wrapper."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from app.core.config import settings
from app.rag.errors import RAGError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingError(RAGError):
    """Raised when the embedding model cannot encode a batch."""


class Embedder:
    """Loads a sentence embedding model and produces unit-norm vectors.

    The model is loaded lazily (on first ``encode``) and outputs are
    L2-normalized so that inner product equals cosine similarity.
    """

    def __init__(
        self,
        model_name: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name or settings.embed_model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """Embedding dimension reported by the loaded model."""
        return int(self.model.get_embedding_dimension())

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of documents to L2-normalized float32 vectors.

        Returns an array of shape ``(len(texts), dim)``.

        Raises:
            EmbeddingError: if the underlying model raises.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:
            vectors = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to encode {len(texts)} documents: {exc}"
            ) from exc

        return vectors.astype(np.float32, copy=False)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query to a unit-norm vector of shape ``(1, dim)``."""
        if self.query_instruction:
            query = f"{self.query_instruction}{query}"
        return self.encode_documents([query])
