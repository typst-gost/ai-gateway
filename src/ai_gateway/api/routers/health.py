from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Response, status

from ai_gateway.api.deps import get_db, get_indexer, get_retriever, get_settings
from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.rag.indexer import DocsIndexer
from ai_gateway.rag.retriever import DocsRetriever

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready(
    response: Response,
    settings: Settings = Depends(get_settings),
    db: GatewayDatabase = Depends(get_db),
    retriever: DocsRetriever = Depends(get_retriever),
    indexer: DocsIndexer = Depends(get_indexer),
) -> dict[str, Any]:
    docs_ready = retriever.is_ready()
    docs_count = retriever.count_rows() if docs_ready else 0
    gw = await db.get_gateway_settings()
    profiles = await db.list_profiles()

    if not docs_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if docs_ready else "degraded",
        "docs_ready": docs_ready,
        "docs_count": docs_count,
        "reindex_running": indexer.is_running,
        "profiles_loaded": len(profiles),
        "clients": {
            "gateway_public_url": gw.get("gateway_public_url", settings.gateway_public_url),
            "website_base_url": gw.get("website_base_url", settings.website_base_url),
            "telegram_bot_url": gw.get("telegram_bot_url", settings.telegram_bot_url),
        },
    }