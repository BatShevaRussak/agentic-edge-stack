"""Domain types passed between RAG components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage with source attribution."""

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A chunk paired with its cosine similarity score (in ``[-1, 1]``)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """The output of a single retrieval call, hits ranked best-first."""

    query: str
    hits: list[RetrievalHit]
