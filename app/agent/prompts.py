"""Prompt templates, router output parser, and a heuristic pre-filter.

Routing is a two-tier cascade: ``quick_classify`` resolves obvious cases
(arithmetic, translation, identity) without an LLM call, the LLM router
handles everything else.
"""

from __future__ import annotations

import re

from app.agent.types import Route

_ROUTER_PROMPT_TEMPLATE = """You are a routing classifier for an AI assistant.
Decide whether the user's question requires looking up TECHNICAL FACTS in a
local knowledge base about a specific project (a Llama 3.2 deployment, a RAG
pipeline using FAISS, BGE embeddings, the Ollama runtime, and a chunking
strategy) -- or whether it can be answered directly without any external
context.

Reply with EXACTLY ONE WORD on a single line. No explanation. No punctuation.
Allowed answers: RAG  or  DIRECT.

Use RAG when the question is about: this project's components, configuration,
embeddings, vector store, retrieval, model card, deployment, chunking, or any
specific technical fact that might be documented.

Use DIRECT when the question is general knowledge, math, translation, casual
conversation, identity ("who are you"), or unrelated to the project.

Examples:
Question: What chunk size does this RAG pipeline use?
Answer: RAG

Question: Why was BGE-small chosen over MiniLM?
Answer: RAG

Question: How does FAISS compute cosine similarity in this project?
Answer: RAG

Question: What is 17 multiplied by 24?
Answer: DIRECT

Question: Translate "good morning" to Spanish.
Answer: DIRECT

Question: Who are you?
Answer: DIRECT

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
    """Map the router LLM's free-form output to a ``Route``.

    Permissive: 1B models often emit extra tokens. If both labels appear
    we pick the one that came first; if neither, we default to ``"rag"``
    (safer to over-search than to answer a project question without
    context).
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
    return "rag"


# --- Heuristic pre-filter --------------------------------------------------

_MATH_PATTERN = re.compile(
    r"\b\d+\s*(?:\+|-|\*|/|x|×|÷|times|plus|minus|divided\s+by|multiplied\s+by)"
    r"\s*\d+\b",
    re.IGNORECASE,
)

_DIRECT_PHRASES: tuple[str, ...] = (
    "translate ",
    "translation of ",
    "in spanish",
    "in french",
    "in german",
    "in hebrew",
    "in italian",
    "who are you",
    "what are you",
    "how are you",
    "your name",
    "your purpose",
    "what's your name",
    "introduce yourself",
)


def quick_classify(query: str) -> Route | None:
    """Return ``"direct"`` for unambiguous general queries, ``None`` otherwise.

    Catches arithmetic, translation requests, and self-identity questions.
    Never returns ``"rag"`` -- project-specific detection is fuzzier and
    is left to the LLM router so the agentic decision stays in the trace.
    """
    text = query.lower().strip()

    if _MATH_PATTERN.search(text):
        return "direct"

    for phrase in _DIRECT_PHRASES:
        if phrase in text:
            return "direct"

    return None
