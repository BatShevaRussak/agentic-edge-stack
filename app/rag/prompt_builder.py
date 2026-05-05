"""Builds the RAG-augmented prompt sent to the LLM."""

from __future__ import annotations

from app.rag.types import RetrievalResult

_SYSTEM_INSTRUCTION = (
    "You are a precise technical assistant. Answer the user's question "
    "using ONLY the information in the CONTEXT below. "
    "Quote facts faithfully and cite sources by their bracketed number "
    "(e.g. [1]). If the answer is not present in the context, reply "
    'exactly: "I don\'t have that information in my knowledge base." '
    "Do not invent facts."
)

_NO_CONTEXT_INSTRUCTION = (
    "You are a precise technical assistant. The retriever found no "
    "relevant context for the user's question. Reply with exactly: "
    '"I don\'t have that information in my knowledge base."'
)


def build_rag_prompt(result: RetrievalResult) -> str:
    """Assemble the final prompt string from a retrieval result."""
    if not result.hits:
        return f"{_NO_CONTEXT_INSTRUCTION}\n\nQuestion: {result.query}"

    context_blocks: list[str] = []
    for index, hit in enumerate(result.hits, start=1):
        context_blocks.append(
            f"[{index}] {hit.chunk.source}\n{hit.chunk.text.strip()}"
        )
    context = "\n\n".join(context_blocks)

    return (
        f"{_SYSTEM_INSTRUCTION}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {result.query}\n\n"
        f"ANSWER:"
    )
