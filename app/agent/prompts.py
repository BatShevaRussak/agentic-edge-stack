"""Prompt templates and the router output parser.

Every query is classified by a single deterministic LLM call -- no
heuristic shortcuts. See ``app.agent.nodes.router_node`` for the wiring.
"""

from __future__ import annotations

from app.agent.types import Route

_ROUTER_PROMPT_TEMPLATE = """You are a routing classifier for an AI assistant.
Reply with EXACTLY ONE WORD on a single line. No explanation. No punctuation.
Allowed answers: DIRECT  or  RAG.

Use DIRECT for general knowledge, math, translation, casual conversation,
self-identity ("who are you"), greetings, definitions of common concepts, or
ANY question that does not depend on private project documentation. This is
the default whenever the question can be answered without project-specific
context.

Use RAG ONLY when the question is about THIS specific project's
configuration, components, embeddings, vector store, retrieval, model card,
deployment, or chunking strategy.

Examples:
Question: What is 17 multiplied by 24?
Answer: DIRECT

Question: Translate "good morning" to Spanish.
Answer: DIRECT

Question: Who are you?
Answer: DIRECT

Question: What is the capital of France?
Answer: DIRECT

Question: Explain HTTP in one sentence.
Answer: DIRECT

Question: What chunk size does this RAG pipeline use?
Answer: RAG

Question: Why was BGE-small chosen over MiniLM?
Answer: RAG

Question: How does FAISS compute cosine similarity in this project?
Answer: RAG

Question: {query}
Answer:"""


_DIRECT_SYSTEM_PROMPT = """You are a concise, helpful assistant. Answer the
user's question directly in 1-3 sentences. Do not invent project-specific
facts. If the question is unclear, ask one short clarifying question.

Question: {query}
Answer:"""


# Kept identical to ``app.rag.prompt_builder._NO_CONTEXT_INSTRUCTION``.
FALLBACK_MESSAGE = "I don't have that information in my knowledge base."


def build_router_prompt(query: str) -> str:
    """Render the router prompt for ``query``."""
    return _ROUTER_PROMPT_TEMPLATE.format(query=query.strip())


def build_direct_prompt(query: str) -> str:
    """Render the direct-answer prompt for ``query``."""
    return _DIRECT_SYSTEM_PROMPT.format(query=query.strip())


def parse_route(raw_output: str) -> Route:
    """Map the router LLM's output to a ``Route``.

    Permissive: if both labels appear the first wins; if neither, default
    to ``"direct"`` -- a wrong ``direct`` gives a generic answer, a wrong
    ``rag`` returns the "no information" fallback on every off-topic query.
    """
    text = raw_output.strip().upper()
    has_rag = "RAG" in text
    has_direct = "DIRECT" in text

    if has_rag and not has_direct:
        return "rag"
    if has_direct and not has_rag:
        return "direct"
    if has_rag and has_direct:
        return "rag" if text.find("RAG") < text.find("DIRECT") else "direct"
    return "direct"

