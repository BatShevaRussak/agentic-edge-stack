"""Pydantic models for the public HTTP surface of Part 4.

These intentionally live in the API layer (not next to the agent layer):
they describe the *wire* contract, not the agent's internal types. This
keeps the agent free of HTTP / OpenAPI concerns and lets us evolve the
two surfaces independently.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for ``POST /chat``."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's question. Routed to RAG or answered directly.",
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: Literal["ok", "degraded"] = Field(
        ..., description="Overall service status."
    )
    ollama_alive: bool = Field(
        ..., description="Whether the local Ollama daemon answered the probe."
    )
    retriever_ready: bool = Field(
        ..., description="Whether the FAISS index is loaded and queryable."
    )
    retriever_chunks: int = Field(
        ..., description="Number of chunks indexed in the active retriever."
    )
    ollama_model: str = Field(..., description="Tag of the active LLM model.")
    embed_model: str = Field(..., description="Embedding model name or path.")
