"""Tool definitions exposed to the agent.

The agent has a single tool, ``rag_search``, declared with ``@tool`` from
``langchain_core``. The function docstring is what an LLM-driven router
sees as the tool description.
"""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.core.config import settings
from app.rag.retriever import Retriever
from app.rag.types import RetrievalResult


class RagSearchInput(BaseModel):
    """Input schema for ``rag_search``."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description=(
            "A focused search query, ideally a rewritten version of the "
            "user's question that captures the key technical terms."
        ),
    )


_RETRIEVER: Retriever | None = None


def set_retriever(retriever: Retriever) -> None:
    """Inject the ``Retriever`` used by ``rag_search`` (DI seam for tests)."""
    global _RETRIEVER
    _RETRIEVER = retriever


def get_retriever() -> Retriever:
    """Return the active ``Retriever``, ingesting on first use."""
    global _RETRIEVER
    if _RETRIEVER is None:
        retriever = Retriever()
        retriever.ingest_directory(settings.data_dir, use_cache=True)
        _RETRIEVER = retriever
    return _RETRIEVER


@tool("rag_search", args_schema=RagSearchInput)
def rag_search(query: str) -> RetrievalResult:
    """Search the local technical knowledge base for facts about the
    deployed Llama 3.2 model, the RAG pipeline (FAISS IndexFlatIP),
    BGE-small embeddings, the Ollama runtime, or this project's chunking
    strategy. Use this for project-specific or technical questions.
    Do NOT use for general knowledge, math, translation, or unrelated
    topics. Returns the top-K matching chunks ranked by cosine similarity."""
    retriever = get_retriever()
    return retriever.retrieve(query)
