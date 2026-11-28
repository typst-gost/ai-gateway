from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from ai_gateway.api.rate_limit import SlidingWindowRateLimiter
from ai_gateway.api.routers import admin_router, assistant_router, health_router
from ai_gateway.config import Settings, load_settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.logging_setup import setup_logging
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.eval_service import EvalService
from ai_gateway.rag.guard import TopicGuard
from ai_gateway.rag.indexer import DocsIndexer
from ai_gateway.rag.retriever import DocsRetriever
from ai_gateway.rag.service import RagService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    db = GatewayDatabase(settings.sqlite_path, settings)
    await db.connect()

    polza = PolzaClient(settings)
    retriever = DocsRetriever(settings.db_path)
    guard = TopicGuard(polza)
    indexer = DocsIndexer(settings, db, retriever)
    rag_service = RagService(settings, db, polza, retriever, guard)
    eval_service = EvalService(settings, db, polza, retriever)
    rate_limiter = SlidingWindowRateLimiter()

    app.state.db = db
    app.state.polza = polza
    app.state.retriever = retriever
    app.state.guard = guard
    app.state.indexer = indexer
    app.state.rag_service = rag_service
    app.state.eval_service = eval_service
    app.state.rate_limiter = rate_limiter

    if not retriever.is_ready():
        if settings.sync_docs_on_start:
            logger.info("Vector DB not found on startup; running initial sync & index...")
            try:
                await indexer.reindex(trigger="startup", sync_github=True)
            except Exception:
                logger.exception("Initial startup reindex failed; service started degraded")
        else:
            logger.warning("Vector DB not found; run POST /api/v1/admin/rag/reindex")

    logger.info(
        f"AI Gateway ready on {settings.host}:{settings.port} "
        f"(docs_ready={retriever.is_ready()}, rows={retriever.count_rows()})"
    )

    try:
        yield
    finally:
        await polza.close()
        await db.close()


def create_app() -> FastAPI:
    settings = load_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    app = FastAPI(
        title="Typst GOST AI Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    allowed_origins = list(dict.fromkeys([*settings.cors_allowed_origins, settings.website_base_url]))
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health_router)
    app.include_router(assistant_router)
    app.include_router(admin_router)
    return app


app = create_app()


def run() -> None:
    settings: Settings = app.state.settings
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )


if __name__ == "__main__":
    run()