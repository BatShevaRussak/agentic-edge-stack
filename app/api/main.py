"""FastAPI application factory + lifespan wiring.

The lifespan handler builds the (expensive) ``AgentRunner`` singleton
once at startup, before the server accepts traffic, and stashes it on
``app.state``. Routes obtain it via ``Depends(get_runner)``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agent import AgentRunner
from app.api.routes import router as api_router
from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build the ``AgentRunner`` once and stash it on ``app.state``.

    The runner ingests the corpus (re-using Part 2's on-disk cache when
    valid), so a warm restart is ~0.4s while a cold restart is ~5s.
    """
    logger.info("Starting up: initializing AgentRunner ...")
    runner = AgentRunner()
    app.state.runner = runner
    logger.info(
        "AgentRunner ready (model=%s, chunks=%d).",
        runner.llm.model,
        runner.retriever.store.size if runner.retriever.is_ready else 0,
    )
    try:
        yield
    finally:
        logger.info("Shutting down.")


def create_app() -> FastAPI:
    """Build and return the configured FastAPI app instance."""
    app = FastAPI(
        title="Agentic Edge Stack",
        version="0.4.0",
        description=(
            "Locally hosted agentic AI assistant with RAG and SSE streaming. "
            "POST /chat streams Server-Sent Events; GET /health probes liveness."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)
    return app


app = create_app()
