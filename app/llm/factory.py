"""Resolve the active LLM client.

Currently there is a single backend (Ollama). The factory function is kept
to centralize client construction so future code (RAG, Agent, API) can
depend on a single seam for dependency injection / testing.
"""

from app.llm.ollama_client import OllamaClient


def get_llm_client() -> OllamaClient:
    """Return the configured local Ollama client."""
    return OllamaClient()
