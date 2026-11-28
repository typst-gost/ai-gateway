from __future__ import annotations

from fastapi import Request

from ai_gateway.api.rate_limit import SlidingWindowRateLimiter
from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.eval_service import EvalService
from ai_gateway.rag.guard import TopicGuard
from ai_gateway.rag.indexer import DocsIndexer
from ai_gateway.rag.retriever import DocsRetriever
from ai_gateway.rag.service import RagService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_db(request: Request) -> GatewayDatabase:
    return request.app.state.db


def get_polza(request: Request) -> PolzaClient:
    return request.app.state.polza


def get_retriever(request: Request) -> DocsRetriever:
    return request.app.state.retriever


def get_guard(request: Request) -> TopicGuard:
    return request.app.state.guard


def get_indexer(request: Request) -> DocsIndexer:
    return request.app.state.indexer


def get_rag_service(request: Request) -> RagService:
    return request.app.state.rag_service


def get_eval_service(request: Request) -> EvalService:
    return request.app.state.eval_service


def get_rate_limiter(request: Request) -> SlidingWindowRateLimiter:
    return request.app.state.rate_limiter


def extract_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client.host if request.client else "anonymous"