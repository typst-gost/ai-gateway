from __future__ import annotations

import re
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from loguru import logger

from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.assets import list_compile_assets
from ai_gateway.rag.guard import TopicGuard, format_guard_rag_debug
from ai_gateway.rag.query_builder import expand_query
from ai_gateway.rag.retriever import DocsRetriever, normalize_active_doc_url
from ai_gateway.schemas import (
    AssistantChatRequest,
    AssistantChatResponse,
    AssistantFixRequest,
    AssistantFixResponse,
    AssistantSearchRequest,
    AssistantSearchResponse,
    DocSourceSchema,
    GuardVerdictSchema,
    PageContext,
    RetrievalStatsSchema,
)

CODE_SEGMENT_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
CITATION_GROUP_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
FENCE_OPEN_RE = re.compile(r"^[ \t]*(`{3,}|~{3,})([^\n]*)$")


def filter_and_renumber_sources(
    markdown_text: str,
    sources: list[DocSourceSchema],
) -> tuple[str, list[DocSourceSchema]]:
    if not sources or not markdown_text.strip():
        return markdown_text, []

    source_by_label = {src.label: src for src in sources}
    new_label_by_old: dict[int, int] = {}
    segments = CODE_SEGMENT_RE.split(markdown_text)

    for idx in range(0, len(segments), 2):
        for match in CITATION_GROUP_RE.finditer(segments[idx]):
            for part in match.group(1).split(","):
                num = int(part.strip())
                if num in source_by_label and num not in new_label_by_old:
                    new_label_by_old[num] = len(new_label_by_old) + 1

    if not new_label_by_old:
        return markdown_text, []

    def _remap(match: re.Match[str]) -> str:
        raw_nums = [int(part.strip()) for part in match.group(1).split(",")]
        remapped: list[str] = []
        for num in raw_nums:
            new_num = new_label_by_old.get(num)
            remapped.append(f"[{new_num if new_num is not None else num}]")
        return ", ".join(remapped) if len(remapped) > 1 else remapped[0]

    for idx in range(0, len(segments), 2):
        segments[idx] = CITATION_GROUP_RE.sub(_remap, segments[idx])

    used_sources = [
        source_by_label[old_label].model_copy(update={"label": new_label})
        for old_label, new_label in new_label_by_old.items()
    ]
    return "".join(segments), used_sources


def strip_typst_code_fences(text: str) -> str:
    cleaned = text.strip()
    match = re.search(r"```(?:typst|typ)?\s*\n([\s\S]*?)\n```", cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return cleaned


def _format_source_for_profile(
    src: DocSourceSchema,
    *,
    url_mode: str,
    include_content: bool = False,
) -> DocSourceSchema:
    target_url = src.path if url_mode == "relative" else src.url
    return src.model_copy(
        update={
            "url": target_url,
            "content": src.content if include_content else None,
        }
    )


class RagService:
    def __init__(
        self,
        settings: Settings,
        db: GatewayDatabase,
        polza: PolzaClient,
        retriever: DocsRetriever,
        guard: TopicGuard,
    ) -> None:
        self._settings = settings
        self._db = db
        self._polza = polza
        self._retriever = retriever
        self._guard = guard

    def _build_assets_section(self, available_assets: list[str] | None) -> str:
        assets = (
            available_assets
            if available_assets is not None
            else list_compile_assets(self._settings.assets_dir)
        )
        if not assets:
            return (
                "ДОСТУПНЫЕ АССЕТЫ ДЛЯ КОМПИЛЯЦИИ:\n"
                "Локальные файлы изображений отсутствуют. В примерах кода не указывай вымышленные пути в `#image(...)` или `#read(...)` — вместо этого используй встроенную графику Typst (`rect`, `circle`, `polygon`, `table`)."
            )
        listed = "\n".join(f"- `{item}`" for item in assets[:80])
        return (
            "ДОСТУПНЫЕ АССЕТЫ ДЛЯ КОМПИЛЯЦИИ:\n"
            "Для примеров с изображениями или внешними данными используй ТОЛЬКО реальные пути из списка ниже (если подходящего файла нет — рисуй средствами Typst):\n"
            f"{listed}"
        )

    def _render_system_prompt(
        self,
        template: str,
        *,
        sources: list[DocSourceSchema],
        include_assets: bool,
        available_assets: list[str] | None,
        page_context: PageContext | None,
    ) -> str:
        context_text = (
            "\n---\n".join(
                f"[Источник {src.label}] ({src.title}):\n{src.content or ''}"
                for src in sources
            )
            if sources
            else "Дополнительный контекст не найден."
        )
        assets_section = (
            self._build_assets_section(available_assets) if include_assets else ""
        )
        current_page_section = ""
        if page_context and page_context.title and page_context.location:
            current_page_section = (
                f"\nПользователь сейчас читает страницу: «{page_context.title}» "
                f"({page_context.location})."
            )

        rendered = (
            template.replace("{context}", context_text)
            .replace("{assets_section}", assets_section)
            .replace("{current_page_section}", current_page_section)
        )
        return re.sub(r"\n{3,}", "\n\n", rendered).strip()

    async def _retrieve_sources(
        self,
        *,
        raw_query: str,
        history: list[dict[str, str]] | None,
        page_context: PageContext | None,
        max_sources_override: int | None = None,
    ) -> tuple[list[DocSourceSchema], RetrievalStatsSchema]:
        gw = await self._db.get_gateway_settings()
        docs_base_url = str(gw["docs_base_url"])
        active_doc_url = normalize_active_doc_url(
            page_context.location if page_context else None,
            docs_base_url,
        )
        embedding_query, fts_query = expand_query(
            raw_query,
            history=history,
            page_title=page_context.title if page_context else None,
        )
        query_vector = await self._polza.create_embedding(
            str(gw["embedding_model"]),
            embedding_query,
        )
        return await self._retriever.retrieve_with_stats(
            raw_query=raw_query,
            query_vector=query_vector,
            fts_query=fts_query,
            docs_base_url=docs_base_url,
            min_score=float(gw["rag_min_score"]),
            rel_ratio=float(gw["rag_rel_ratio"]),
            limit=int(gw["rag_search_limit"]),
            max_sources=max_sources_override or int(gw["rag_max_sources"]),
            max_page_chars=int(gw["rag_max_page_chars"]),
            max_total_context_chars=int(gw["rag_max_total_context_chars"]),
            active_doc_url=active_doc_url,
        )

    def _resolve_query_and_history(
        self,
        req: AssistantChatRequest,
    ) -> tuple[str, list[dict[str, str]] | None]:
        if req.messages:
            cleaned = [
                {"role": m.role, "content": m.content.strip()}
                for m in req.messages
                if m.content.strip()
            ]
            user_msgs = [m for m in cleaned if m["role"] == "user"]
            if not user_msgs:
                raise ValueError("No user message provided")
            last_user = user_msgs[-1]["content"]
            prior = cleaned[:-1] if len(cleaned) > 1 else []
            return last_user, prior

        query = (req.query or "").strip()
        if not query:
            raise ValueError("Either 'query' or 'messages' must be provided")
        return query, None

    async def _prepare_chat_context(
        self,
        req: AssistantChatRequest,
    ) -> dict[str, Any]:
        profile = await self._db.get_profile(req.channel)
        if profile is None:
            raise ValueError(f"Unknown channel profile: {req.channel}")
        if not profile["enabled"]:
            raise PermissionError(f"Channel '{req.channel}' is disabled")

        query, client_history = self._resolve_query_and_history(req)
        max_chars = int(profile["max_query_chars"])
        if len(query) > max_chars:
            raise ValueError(f"Query exceeds maximum length ({max_chars} chars)")

        conversation_id = req.conversation_id or f"{req.channel}:{uuid.uuid4().hex}"
        if req.save_history:
            await self._db.ensure_conversation(
                conversation_id=conversation_id,
                channel=req.channel,
                external_user_id=req.external_user_id,
                external_chat_id=req.external_chat_id,
                title=query,
                metadata={
                    **(req.metadata or {}),
                    **(
                        {"page_title": req.page_context.title, "page_location": req.page_context.location}
                        if req.page_context
                        else {}
                    ),
                },
            )

        history: list[dict[str, str]] = []
        if profile["use_history"]:
            if client_history is not None:
                limit = int(profile["history_max_messages"])
                history = client_history[-limit:] if limit > 0 else []
            elif req.conversation_id:
                history = await self._db.get_conversation_history(
                    conversation_id,
                    max_messages=int(profile["history_max_messages"]),
                    ttl_seconds=int(profile["history_ttl_seconds"]),
                )

        return {
            "profile": profile,
            "query": query,
            "history": history,
            "conversation_id": conversation_id,
        }

    async def search(self, req: AssistantSearchRequest) -> AssistantSearchResponse:
        history = (
            [{"role": m.role, "content": m.content} for m in req.history]
            if req.history
            else None
        )
        sources, stats = await self._retrieve_sources(
            raw_query=req.query.strip(),
            history=history,
            page_context=req.page_context,
            max_sources_override=req.max_sources,
        )
        formatted = [
            src if req.include_content else src.model_copy(update={"content": None})
            for src in sources
        ]
        return AssistantSearchResponse(sources=formatted, stats=stats)

    async def chat_sync(self, req: AssistantChatRequest) -> AssistantChatResponse:
        started = time.monotonic()
        ctx = await self._prepare_chat_context(req)
        profile: dict[str, Any] = ctx["profile"]
        query: str = ctx["query"]
        history: list[dict[str, str]] = ctx["history"]
        conversation_id: str = ctx["conversation_id"]

        verdict = await self._guard.check_query(
            query=query,
            profile=profile,
            history=history,
            page_context=req.page_context,
        )

        if not verdict.allowed:
            latency_ms = (time.monotonic() - started) * 1000.0
            refusal = profile["refusal_reply"]
            debug_md = format_guard_rag_debug(verdict) if profile["guard_debug"] else None
            if req.save_history:
                await self._db.append_turn_messages(
                    conversation_id=conversation_id,
                    user_content=query,
                    assistant_content=refusal,
                    llm_model=profile["llm_model"],
                    decision_model=verdict.model,
                    guard_allowed=False,
                    guard_intent=verdict.intent,
                    guard_offtopic_prob=verdict.offtopic_prob,
                    guard_noul_prob=verdict.abuse_prob,
                    guard_latency_ms=verdict.latency_ms,
                    rag_candidates=None,
                    rag_selected=None,
                    rag_cited=None,
                    rag_top_score=None,
                    sources=[],
                    usage=None,
                    latency_ms=latency_ms,
                )
            return AssistantChatResponse(
                conversation_id=conversation_id,
                channel=req.channel,
                answer=refusal,
                raw_answer=refusal,
                sources=[],
                guard=verdict,
                stats=None,
                blocked_by_guard=True,
                llm_model=profile["llm_model"],
                decision_model=verdict.model,
                usage=None,
                latency_ms=latency_ms,
                debug_markdown=debug_md,
            )

        sources, stats = await self._retrieve_sources(
            raw_query=query,
            history=history,
            page_context=req.page_context,
        )
        if not sources:
            stats = stats.model_copy(update={"cited_count": 0})
            latency_ms = (time.monotonic() - started) * 1000.0
            no_docs = profile["no_docs_reply"]
            debug_md = format_guard_rag_debug(verdict, stats) if profile["guard_debug"] else None
            if req.save_history:
                await self._db.append_turn_messages(
                    conversation_id=conversation_id,
                    user_content=query,
                    assistant_content=no_docs,
                    llm_model=profile["llm_model"],
                    decision_model=verdict.model,
                    guard_allowed=True,
                    guard_intent=verdict.intent,
                    guard_offtopic_prob=verdict.offtopic_prob,
                    guard_noul_prob=verdict.abuse_prob,
                    guard_latency_ms=verdict.latency_ms,
                    rag_candidates=stats.candidates_count,
                    rag_selected=0,
                    rag_cited=0,
                    rag_top_score=stats.top_score,
                    sources=[],
                    usage=None,
                    latency_ms=latency_ms,
                )
            return AssistantChatResponse(
                conversation_id=conversation_id,
                channel=req.channel,
                answer=no_docs,
                raw_answer=no_docs,
                sources=[],
                guard=verdict,
                stats=stats,
                no_docs_found=True,
                llm_model=profile["llm_model"],
                decision_model=verdict.model,
                usage=None,
                latency_ms=latency_ms,
                debug_markdown=debug_md,
            )

        system_prompt = self._render_system_prompt(
            profile["system_prompt"],
            sources=sources,
            include_assets=profile["include_assets_in_prompt"],
            available_assets=req.available_assets,
            page_context=req.page_context,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query},
        ]

        raw_answer, usage = await self._polza.create_chat_completion(
            model=profile["llm_model"],
            messages=messages,
            temperature=float(profile["temperature"]),
            max_tokens=profile["max_tokens"],
        )
        if not raw_answer:
            raw_answer = "Не удалось сформировать ответ по найденной документации."

        clean_answer, used_sources = filter_and_renumber_sources(raw_answer, sources)
        stats = stats.model_copy(update={"cited_count": len(used_sources)})
        formatted_sources = [
            _format_source_for_profile(src, url_mode=profile["source_url_mode"])
            for src in used_sources
        ]
        latency_ms = (time.monotonic() - started) * 1000.0
        debug_md = format_guard_rag_debug(verdict, stats) if profile["guard_debug"] else None

        if req.save_history:
            await self._db.append_turn_messages(
                conversation_id=conversation_id,
                user_content=query,
                assistant_content=clean_answer,
                llm_model=profile["llm_model"],
                decision_model=verdict.model,
                guard_allowed=True,
                guard_intent=verdict.intent,
                guard_offtopic_prob=verdict.offtopic_prob,
                guard_noul_prob=verdict.abuse_prob,
                guard_latency_ms=verdict.latency_ms,
                rag_candidates=stats.candidates_count,
                rag_selected=stats.selected_count,
                rag_cited=len(used_sources),
                rag_top_score=stats.top_score,
                sources=[s.model_dump(exclude={"content"}) for s in formatted_sources],
                usage=usage,
                latency_ms=latency_ms,
            )

        return AssistantChatResponse(
            conversation_id=conversation_id,
            channel=req.channel,
            answer=clean_answer,
            raw_answer=raw_answer,
            sources=formatted_sources,
            guard=verdict,
            stats=stats,
            llm_model=profile["llm_model"],
            decision_model=verdict.model,
            usage=usage,
            latency_ms=latency_ms,
            debug_markdown=debug_md,
        )

    async def chat_stream(self, req: AssistantChatRequest) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        started = time.monotonic()
        ctx = await self._prepare_chat_context(req)
        profile: dict[str, Any] = ctx["profile"]
        query: str = ctx["query"]
        history: list[dict[str, str]] = ctx["history"]
        conversation_id: str = ctx["conversation_id"]
        url_mode = profile["source_url_mode"]

        verdict = await self._guard.check_query(
            query=query,
            profile=profile,
            history=history,
            page_context=req.page_context,
        )

        if not verdict.allowed:
            latency_ms = (time.monotonic() - started) * 1000.0
            refusal = profile["refusal_reply"]
            debug_md = format_guard_rag_debug(verdict) if profile["guard_debug"] else None

            yield (
                "meta",
                {
                    "conversation_id": conversation_id,
                    "channel": req.channel,
                    "blocked_by_guard": True,
                    "no_docs_found": False,
                    "guard": verdict.model_dump(),
                    "sources": [],
                    "stats": None,
                    "llm_model": profile["llm_model"],
                    "decision_model": verdict.model,
                },
            )
            yield ("delta", {"text": refusal})
            if req.save_history:
                await self._db.append_turn_messages(
                    conversation_id=conversation_id,
                    user_content=query,
                    assistant_content=refusal,
                    llm_model=profile["llm_model"],
                    decision_model=verdict.model,
                    guard_allowed=False,
                    guard_intent=verdict.intent,
                    guard_offtopic_prob=verdict.offtopic_prob,
                    guard_noul_prob=verdict.abuse_prob,
                    guard_latency_ms=verdict.latency_ms,
                    rag_candidates=None,
                    rag_selected=None,
                    rag_cited=None,
                    rag_top_score=None,
                    sources=[],
                    usage=None,
                    latency_ms=latency_ms,
                )
            yield (
                "done",
                {
                    "conversation_id": conversation_id,
                    "channel": req.channel,
                    "answer": refusal,
                    "raw_answer": refusal,
                    "sources": [],
                    "guard": verdict.model_dump(),
                    "stats": None,
                    "blocked_by_guard": True,
                    "no_docs_found": False,
                    "usage": None,
                    "latency_ms": latency_ms,
                    "debug_markdown": debug_md,
                },
            )
            return

        sources, stats = await self._retrieve_sources(
            raw_query=query,
            history=history,
            page_context=req.page_context,
        )
        initial_formatted_sources = [
            _format_source_for_profile(src, url_mode=url_mode) for src in sources
        ]

        if not sources:
            stats = stats.model_copy(update={"cited_count": 0})
            latency_ms = (time.monotonic() - started) * 1000.0
            no_docs = profile["no_docs_reply"]
            debug_md = format_guard_rag_debug(verdict, stats) if profile["guard_debug"] else None

            yield (
                "meta",
                {
                    "conversation_id": conversation_id,
                    "channel": req.channel,
                    "blocked_by_guard": False,
                    "no_docs_found": True,
                    "guard": verdict.model_dump(),
                    "sources": [],
                    "stats": stats.model_dump(),
                    "llm_model": profile["llm_model"],
                    "decision_model": verdict.model,
                },
            )
            yield ("delta", {"text": no_docs})
            if req.save_history:
                await self._db.append_turn_messages(
                    conversation_id=conversation_id,
                    user_content=query,
                    assistant_content=no_docs,
                    llm_model=profile["llm_model"],
                    decision_model=verdict.model,
                    guard_allowed=True,
                    guard_intent=verdict.intent,
                    guard_offtopic_prob=verdict.offtopic_prob,
                    guard_noul_prob=verdict.abuse_prob,
                    guard_latency_ms=verdict.latency_ms,
                    rag_candidates=stats.candidates_count,
                    rag_selected=0,
                    rag_cited=0,
                    rag_top_score=stats.top_score,
                    sources=[],
                    usage=None,
                    latency_ms=latency_ms,
                )
            yield (
                "done",
                {
                    "conversation_id": conversation_id,
                    "channel": req.channel,
                    "answer": no_docs,
                    "raw_answer": no_docs,
                    "sources": [],
                    "guard": verdict.model_dump(),
                    "stats": stats.model_dump(),
                    "blocked_by_guard": False,
                    "no_docs_found": True,
                    "usage": None,
                    "latency_ms": latency_ms,
                    "debug_markdown": debug_md,
                },
            )
            return

        yield (
            "meta",
            {
                "conversation_id": conversation_id,
                "channel": req.channel,
                "blocked_by_guard": False,
                "no_docs_found": False,
                "guard": verdict.model_dump(),
                "sources": [s.model_dump(exclude={"content"}) for s in initial_formatted_sources],
                "stats": stats.model_dump(),
                "llm_model": profile["llm_model"],
                "decision_model": verdict.model,
            },
        )

        system_prompt = self._render_system_prompt(
            profile["system_prompt"],
            sources=sources,
            include_assets=profile["include_assets_in_prompt"],
            available_assets=req.available_assets,
            page_context=req.page_context,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": query},
        ]

        parts: list[str] = []
        usage: dict[str, Any] | None = None
        stream_error: str | None = None
        completed = False

        try:
            async for kind, payload in self._polza.stream_chat_completion(
                model=profile["llm_model"],
                messages=messages,
                temperature=float(profile["temperature"]),
                max_tokens=profile["max_tokens"],
            ):
                if kind == "delta":
                    parts.append(payload)
                    yield ("delta", {"text": payload})
                elif kind == "usage":
                    usage = payload
            completed = True
        except Exception as exc:
            stream_error = str(exc)
            logger.exception("Error while streaming LLM response")
            yield ("error", {"detail": stream_error})
        finally:
            raw_answer = "".join(parts).strip()
            if completed and not raw_answer:
                raw_answer = "Не удалось сформировать ответ по найденной документации."

            clean_answer, used_sources = filter_and_renumber_sources(raw_answer, sources)
            final_stats = stats.model_copy(update={"cited_count": len(used_sources)})
            used_formatted = [
                _format_source_for_profile(src, url_mode=url_mode) for src in used_sources
            ]
            latency_ms = (time.monotonic() - started) * 1000.0

            if req.save_history and (raw_answer or stream_error):
                try:
                    await self._db.append_turn_messages(
                        conversation_id=conversation_id,
                        user_content=query,
                        assistant_content=clean_answer or (stream_error or ""),
                        llm_model=profile["llm_model"],
                        decision_model=verdict.model,
                        guard_allowed=True,
                        guard_intent=verdict.intent,
                        guard_offtopic_prob=verdict.offtopic_prob,
                        guard_noul_prob=verdict.abuse_prob,
                        guard_latency_ms=verdict.latency_ms,
                        rag_candidates=final_stats.candidates_count,
                        rag_selected=final_stats.selected_count,
                        rag_cited=len(used_formatted),
                        rag_top_score=final_stats.top_score,
                        sources=[s.model_dump(exclude={"content"}) for s in used_formatted],
                        usage=usage,
                        latency_ms=latency_ms,
                        stopped=not completed and stream_error is None,
                        error=stream_error,
                    )
                except Exception:
                    logger.exception("Failed to persist streamed conversation turn")

        if completed:
            debug_md = (
                format_guard_rag_debug(verdict, final_stats)
                if profile["guard_debug"]
                else None
            )
            yield (
                "done",
                {
                    "conversation_id": conversation_id,
                    "channel": req.channel,
                    "answer": clean_answer,
                    "raw_answer": raw_answer,
                    "sources": [s.model_dump(exclude={"content"}) for s in used_formatted],
                    "guard": verdict.model_dump(),
                    "stats": final_stats.model_dump(),
                    "blocked_by_guard": False,
                    "no_docs_found": False,
                    "usage": usage,
                    "latency_ms": latency_ms,
                    "debug_markdown": debug_md,
                },
            )

    async def fix_code(self, req: AssistantFixRequest) -> AssistantFixResponse:
        started = time.monotonic()
        profile = await self._db.get_profile("telegram_fix")
        if profile is None:
            raise ValueError("Profile 'telegram_fix' not found")
        if not profile["enabled"]:
            raise PermissionError("Channel 'telegram_fix' is disabled")

        conversation_id = req.conversation_id or f"telegram_fix:{uuid.uuid4().hex}"
        user_summary = f"Ошибка:\n{req.error_text[:500]}\n\nКод:\n{req.source_code[:2000]}"

        if req.save_history:
            await self._db.ensure_conversation(
                conversation_id=conversation_id,
                channel="telegram_fix",
                external_user_id=req.external_user_id,
                external_chat_id=req.external_chat_id,
                title=f"Исправление кода: {req.error_text[:80]}",
                metadata=req.metadata,
            )

        verdict = await self._guard.check_code_fix(
            source_code=req.source_code,
            error_text=req.error_text,
            profile=profile,
        )

        if not verdict.allowed:
            latency_ms = (time.monotonic() - started) * 1000.0
            refusal = profile["refusal_reply"]
            debug_md = (
                format_guard_rag_debug(verdict, is_fix=True)
                if profile["guard_debug"]
                else None
            )
            if req.save_history:
                await self._db.append_turn_messages(
                    conversation_id=conversation_id,
                    user_content=user_summary,
                    assistant_content=refusal,
                    llm_model=profile["llm_model"],
                    decision_model=verdict.model,
                    guard_allowed=False,
                    guard_intent=verdict.intent,
                    guard_offtopic_prob=verdict.offtopic_prob,
                    guard_noul_prob=verdict.abuse_prob,
                    guard_latency_ms=verdict.latency_ms,
                    rag_candidates=None,
                    rag_selected=None,
                    rag_cited=None,
                    rag_top_score=None,
                    sources=[],
                    usage=None,
                    latency_ms=latency_ms,
                )
            return AssistantFixResponse(
                conversation_id=conversation_id,
                fixed_code=req.source_code,
                blocked_by_guard=True,
                refusal_message=refusal,
                guard=verdict,
                stats=None,
                llm_model=profile["llm_model"],
                decision_model=verdict.model,
                usage=None,
                latency_ms=latency_ms,
                debug_markdown=debug_md,
            )

        retrieval_query = f"ошибка typst {req.error_text[:300]}\n{req.source_code[:400]}"
        sources, stats = await self._retrieve_sources(
            raw_query=retrieval_query,
            history=None,
            page_context=None,
            max_sources_override=4,
        )
        system_prompt = self._render_system_prompt(
            profile["system_prompt"],
            sources=sources,
            include_assets=profile["include_assets_in_prompt"],
            available_assets=req.available_assets,
            page_context=None,
        )
        user_prompt = (
            f"Исходный код:\n```typst\n{req.source_code}\n```\n\n"
            f"Ошибка компилятора:\n```text\n{req.error_text}\n```"
        )
        raw_reply, usage = await self._polza.create_chat_completion(
            model=profile["llm_model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=float(profile["temperature"]),
            max_tokens=profile["max_tokens"],
        )
        fixed_code = strip_typst_code_fences(raw_reply) or req.source_code
        latency_ms = (time.monotonic() - started) * 1000.0
        debug_md = (
            format_guard_rag_debug(verdict, stats, is_fix=True)
            if profile["guard_debug"]
            else None
        )

        if req.save_history:
            await self._db.append_turn_messages(
                conversation_id=conversation_id,
                user_content=user_summary,
                assistant_content=f"```typst\n{fixed_code}\n```",
                llm_model=profile["llm_model"],
                decision_model=verdict.model,
                guard_allowed=True,
                guard_intent=verdict.intent,
                guard_offtopic_prob=verdict.offtopic_prob,
                guard_noul_prob=verdict.abuse_prob,
                guard_latency_ms=verdict.latency_ms,
                rag_candidates=stats.candidates_count,
                rag_selected=stats.selected_count,
                rag_cited=None,
                rag_top_score=stats.top_score,
                sources=[s.model_dump(exclude={"content"}) for s in sources],
                usage=usage,
                latency_ms=latency_ms,
            )

        return AssistantFixResponse(
            conversation_id=conversation_id,
            fixed_code=fixed_code,
            blocked_by_guard=False,
            guard=verdict,
            stats=stats,
            llm_model=profile["llm_model"],
            decision_model=verdict.model,
            usage=usage,
            latency_ms=latency_ms,
            debug_markdown=debug_md,
        )