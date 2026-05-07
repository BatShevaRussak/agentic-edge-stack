"""FastAPI layer (Part 4) - exposes the agent over HTTP with SSE streaming."""

from app.api.main import create_app

__all__ = ["create_app"]
