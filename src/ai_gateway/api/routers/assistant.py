from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from loguru import logger

from ai_gateway.api.deps import (
    extract_client_ip,
    get_db,
    get_rag_service,
    get_rate_limiter,
)
from ai_gateway.api.rate_limit import SlidingWindowRateLimiter
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.rag.service import RagService
from ai_gateway.schemas import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantFixRequest,
    AssistantFixResponse,
    AssistantSearchRequest,
    AssistantSearchResponse,
    ClearConversationRequest,
)

router = APIRouter(prefix="/api/v1/assistant", tags=["assistant"])


async def _enforce_rate_limit(
    request: Request,
    db: GatewayDatabase,
    rate_limiter: SlidingWindowRateLimiter,
    *,
    channel: str,
    external_user_id: str | None,
) -> None:
    gw = await db.get_gateway_settings()
    if not gw.get("rate_limit_enabled", True):
        return

    client_key = f"{channel}:{external_user_id or extract_client_ip(request)}"
    wait_seconds = rate_limiter.check(
        client_key,
        max_requests=int(gw.get("rate_limit_requests", 20)),
        window_seconds=int(gw.get("rate_limit_window_seconds", 60)),
    )
    if wait_seconds > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Retry in {wait_seconds:.1f}s.",
            headers={"Retry-After": str(int(wait_seconds) + 1)},
        )


def _format_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@router.post("/chat", response_model=None)
async def assistant_chat(
    req: AssistantChatRequest,
    request: Request,
    db: GatewayDatabase = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    rate_limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
) -> AssistantChatResponse | StreamingResponse:
    await _enforce_rate_limit(
        request,
        db,
        rate_limiter,
        channel=req.channel,
        external_user_id=req.external_user_id,
    )

    if not req.stream:
        try:
            return await rag_service.chat_sync(req)
        except PermissionError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Knowledge base is not indexed yet",
            ) from exc

    agen = rag_service.chat_stream(req)
    try:
        first_event = await anext(agen)
    except PermissionError as exc:
        await agen.aclose()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        await agen.aclose()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        await agen.aclose()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not indexed yet",
        ) from exc
    except StopAsyncIteration:
        await agen.aclose()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Empty stream produced",
        )

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield _format_sse(first_event[0], first_event[1])
            async for event_name, payload in agen:
                if await request.is_disconnected():
                    logger.debug(f"Client disconnected during SSE stream ({req.channel})")
                    break
                yield _format_sse(event_name, payload)
        finally:
            await agen.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/fix", response_model=AssistantFixResponse)
async def assistant_fix(
    req: AssistantFixRequest,
    request: Request,
    db: GatewayDatabase = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    rate_limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
) -> AssistantFixResponse:
    await _enforce_rate_limit(
        request,
        db,
        rate_limiter,
        channel="telegram_fix",
        external_user_id=req.external_user_id,
    )
    try:
        return await rag_service.fix_code(req)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not indexed yet",
        ) from exc


@router.post("/search", response_model=AssistantSearchResponse)
async def assistant_search(
    req: AssistantSearchRequest,
    request: Request,
    db: GatewayDatabase = Depends(get_db),
    rag_service: RagService = Depends(get_rag_service),
    rate_limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
) -> AssistantSearchResponse:
    await _enforce_rate_limit(
        request,
        db,
        rate_limiter,
        channel="search",
        external_user_id=None,
    )
    try:
        return await rag_service.search(req)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base is not indexed yet",
        ) from exc


@router.post("/conversations/clear")
async def clear_conversation(
    req: ClearConversationRequest,
    db: GatewayDatabase = Depends(get_db),
) -> dict[str, Any]:
    cleared = await db.reset_conversation_context(req.conversation_id)
    return {"conversation_id": req.conversation_id, "cleared": cleared}