from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from ai_gateway.api.deps import (
    get_db,
    get_eval_service,
    get_guard,
    get_indexer,
    get_polza,
    get_retriever,
    get_settings,
)
from ai_gateway.auth.security import (
    AdminIdentity,
    create_access_token,
    require_admin,
    verify_admin_credentials,
)
from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.assets import list_compile_assets
from ai_gateway.rag.eval_service import EvalService
from ai_gateway.rag.guard import TopicGuard
from ai_gateway.rag.indexer import DocsIndexer
from ai_gateway.rag.retriever import DocsRetriever
from ai_gateway.schemas import (
    AdminLoginRequest,
    AdminTokenResponse,
    ChannelName,
    EvalChunkLookupRequest,
    EvalDatasetUploadRequest,
    EvalRunRequest,
    GatewaySettingsUpdate,
    ModelCategory,
    ProfileConfigUpdate,
    ReindexTriggerRequest,
    TelegramBulkUpdate,
    TestGuardRequest,
)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/auth/login", response_model=AdminTokenResponse)
async def admin_login(
    req: AdminLoginRequest,
    settings: Settings = Depends(get_settings),
) -> AdminTokenResponse:
    if not verify_admin_credentials(req.username, req.password, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    token = create_access_token(
        req.username,
        settings.jwt_secret,
        settings.jwt_ttl_seconds,
    )
    return AdminTokenResponse(
        access_token=token,
        expires_in=settings.jwt_ttl_seconds,
        username=req.username,
    )


@router.get("/auth/me")
async def admin_me(admin: AdminIdentity = Depends(require_admin)) -> dict[str, str]:
    return {"username": admin.username, "auth_type": admin.auth_type}


@router.get("/polza/balance")
async def polza_balance(
    _admin: AdminIdentity = Depends(require_admin),
    polza: PolzaClient = Depends(get_polza),
) -> dict[str, Any]:
    try:
        return await polza.get_balance()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Polza AI balance request failed: HTTP {exc.response.status_code}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Polza AI balance request failed: {exc}",
        ) from exc


@router.get("/polza/models")
async def polza_models(
    category: ModelCategory = "all",
    search: str | None = None,
    include_providers: bool = False,
    include_capabilities: bool = False,
    only_fz152: bool = False,
    refresh: bool = False,
    limit: int | None = Query(default=None, ge=1, le=500),
    _admin: AdminIdentity = Depends(require_admin),
    polza: PolzaClient = Depends(get_polza),
) -> dict[str, Any]:
    try:
        items = await polza.list_models(
            category=category,
            include_providers=include_providers,
            include_capabilities=include_capabilities,
            refresh=refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Polza AI models request failed: HTTP {exc.response.status_code}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Polza AI models request failed: {exc}",
        ) from exc

    filtered = items
    if only_fz152:
        filtered = [m for m in filtered if m.get("is_fz152_compliant") is True]

    if search and search.strip():
        needle = search.strip().lower()
        filtered = [
            m
            for m in filtered
            if needle in str(m.get("id", "")).lower()
            or needle in str(m.get("name", "")).lower()
            or needle in str(m.get("short_description", "")).lower()
        ]

    total = len(filtered)
    if limit is not None:
        filtered = filtered[:limit]

    return {
        "category": category,
        "total": total,
        "models": filtered,
    }


@router.get("/settings")
async def get_gateway_settings(
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    return await db.get_gateway_settings()


@router.patch("/settings")
async def update_gateway_settings(
    payload: GatewaySettingsUpdate,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    return await db.update_gateway_settings(updates)


@router.get("/profiles")
async def list_profiles(
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    return {"profiles": await db.list_profiles()}


@router.patch("/profiles/telegram/bulk")
async def update_telegram_profiles_bulk(
    payload: TelegramBulkUpdate,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    profiles = await db.update_telegram_profiles(updates)
    return {"updated": len(profiles), "profiles": profiles}


@router.get("/profiles/{profile_id}")
async def get_profile(
    profile_id: ChannelName,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    profile = await db.get_profile(profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")
    return profile


@router.patch("/profiles/{profile_id}")
async def update_profile(
    profile_id: ChannelName,
    payload: ProfileConfigUpdate,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    updates = payload.model_dump(exclude_unset=True)
    try:
        return await db.update_profile(profile_id, updates)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found") from exc


@router.post("/profiles/{profile_id}/reset")
async def reset_profile(
    profile_id: ChannelName,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    try:
        return await db.reset_profile(profile_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found") from exc


@router.post("/test-guard")
async def test_guard(
    req: TestGuardRequest,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
    guard: TopicGuard = Depends(get_guard),
) -> dict[str, Any]:
    profile = await db.get_profile(req.profile_id)
    if profile is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Profile not found")

    history = (
        [{"role": m.role, "content": m.content} for m in req.history]
        if req.history
        else None
    )
    state = guard.build_query_state(req.query, history=history, page_context=req.page_context)
    verdict = await guard.evaluate(
        state=state,
        questions=req.decision_questions_override or profile["decision_questions"],
        model=req.decision_model_override or profile["decision_model"],
        enabled=True,
        offtopic_threshold=(
            req.offtopic_threshold_override
            if req.offtopic_threshold_override is not None
            else profile["guard_offtopic_threshold"]
        ),
        noul_threshold=(
            req.noul_threshold_override
            if req.noul_threshold_override is not None
            else profile["guard_noul_threshold"]
        ),
        timeout_seconds=profile["guard_timeout_seconds"],
        preview_label=f"admin_test profile={req.profile_id}",
    )
    return {"verdict": verdict.model_dump(), "evaluated_state": state}


@router.get("/rag/status")
async def rag_status(
    _admin: AdminIdentity = Depends(require_admin),
    settings: Settings = Depends(get_settings),
    db: GatewayDatabase = Depends(get_db),
    retriever: DocsRetriever = Depends(get_retriever),
    indexer: DocsIndexer = Depends(get_indexer),
) -> dict[str, Any]:
    assets = list_compile_assets(settings.assets_dir)
    docs_ready = retriever.is_ready()
    return {
        "docs_ready": docs_ready,
        "docs_count": retriever.count_rows() if docs_ready else 0,
        "reindex_running": indexer.is_running,
        "compile_assets_count": len(assets),
        "compile_assets": assets,
        "recent_jobs": await db.list_reindex_jobs(limit=10),
    }


async def _background_reindex(
    indexer: DocsIndexer,
    *,
    trigger: str,
    sync_github: bool,
    force_reembed: bool,
) -> None:
    try:
        await indexer.reindex(
            trigger=trigger,
            sync_github=sync_github,
            force_reembed=force_reembed,
        )
    except Exception:
        logger.exception("Background reindex task failed")


@router.post("/rag/reindex")
async def trigger_reindex(
    req: ReindexTriggerRequest,
    admin: AdminIdentity = Depends(require_admin),
    indexer: DocsIndexer = Depends(get_indexer),
) -> dict[str, Any]:
    if indexer.is_running:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Reindex job is already running",
        )

    trigger_label = f"admin:{admin.username}"
    if req.background:
        asyncio.create_task(
            _background_reindex(
                indexer,
                trigger=trigger_label,
                sync_github=req.sync_github,
                force_reembed=req.force_reembed,
            )
        )
        return {
            "status": "started",
            "background": True,
            "sync_github": req.sync_github,
            "force_reembed": req.force_reembed,
        }

    job_id, result = await indexer.reindex(
        trigger=trigger_label,
        sync_github=req.sync_github,
        force_reembed=req.force_reembed,
    )
    return {
        "status": "completed",
        "background": False,
        "job_id": job_id,
        "files_processed": result.files_processed,
        "chunks_count": result.chunks_count,
        "embeddings_fetched": result.embeddings_fetched,
        "assets_synced": result.assets_synced,
    }


@router.get("/rag/jobs")
async def list_reindex_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    return {"jobs": await db.list_reindex_jobs(limit=limit)}


@router.get("/eval/dataset")
async def get_eval_dataset(
    path: str | None = None,
    include_questions: bool = False,
    _admin: AdminIdentity = Depends(require_admin),
    eval_service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    try:
        return await eval_service.inspect_dataset(
            custom_path=path,
            include_questions=include_questions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.put("/eval/dataset")
async def upload_eval_dataset(
    payload: EvalDatasetUploadRequest,
    path: str | None = None,
    _admin: AdminIdentity = Depends(require_admin),
    eval_service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    try:
        saved_path = eval_service.save_dataset(payload.questions, custom_path=path)
        info = await eval_service.inspect_dataset(custom_path=path, include_questions=False)
        return {"saved_to": str(saved_path), **info}
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/eval/chunks/lookup")
async def lookup_eval_chunks(
    req: EvalChunkLookupRequest,
    _admin: AdminIdentity = Depends(require_admin),
    retriever: DocsRetriever = Depends(get_retriever),
) -> dict[str, Any]:
    if not retriever.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not indexed yet",
        )
    chunks = await retriever.get_chunks_by_ids(req.chunk_ids)
    return {"chunks": chunks}


@router.post("/eval/run")
async def run_rag_eval(
    req: EvalRunRequest,
    admin: AdminIdentity = Depends(require_admin),
    eval_service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    trigger = f"admin:{admin.username}"
    try:
        if req.background:
            run_id = await eval_service.start_background(req, trigger=trigger)
            return {
                "status": "started",
                "background": True,
                "run_id": run_id,
            }
        return await eval_service.run_sync(req, trigger=trigger)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        msg = str(exc)
        code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if "Knowledge base" in msg
            else status.HTTP_404_NOT_FOUND
        )
        raise HTTPException(status_code=code, detail=msg) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/eval/runs")
async def list_eval_runs(
    limit: int = Query(default=20, ge=1, le=100),
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
    eval_service: EvalService = Depends(get_eval_service),
) -> dict[str, Any]:
    return {
        "eval_running": eval_service.is_running,
        "runs": await db.list_eval_runs(limit=limit),
    }


@router.get("/eval/runs/{run_id}")
async def get_eval_run(
    run_id: int,
    include_details: bool = True,
    only_misses: bool = False,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    run = await db.get_eval_run(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found")
    if only_misses:
        run["details"] = [
            d for d in run.get("details", []) if float(d.get("reciprocal_rank", 0.0)) == 0.0
        ]
    if not include_details:
        run.pop("details", None)
    return run


@router.delete("/eval/runs/{run_id}")
async def delete_eval_run(
    run_id: int,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    deleted = await db.delete_eval_run(run_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Eval run not found")
    return {"run_id": run_id, "deleted": True}


@router.get("/conversations")
async def list_conversations(
    channel: ChannelName | None = None,
    search: str | None = None,
    external_user_id: str | None = None,
    guard_allowed: bool | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    items, total = await db.list_conversations(
        channel=channel,
        search=search,
        external_user_id=external_user_id,
        guard_allowed=guard_allowed,
        limit=limit,
        offset=offset,
    )
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "conversations": items,
    }


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    detail = await db.get_conversation_detail(conversation_id)
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return detail


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    deleted = await db.delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return {"conversation_id": conversation_id, "deleted": True}


@router.get("/analytics")
async def get_analytics(
    days: int | None = Query(default=30, ge=1, le=3650),
    _admin: AdminIdentity = Depends(require_admin),
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    return await db.get_analytics(days=days)