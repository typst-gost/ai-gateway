from __future__ import annotations

import time
from typing import Any

from loguru import logger

from ai_gateway.polza.client import PolzaClient
from ai_gateway.schemas import (
    ChatMessageInput,
    GuardVerdictSchema,
    PageContext,
    RetrievalStatsSchema,
)


def format_guard_rag_debug(
    verdict: GuardVerdictSchema,
    stats: RetrievalStatsSchema | None = None,
    *,
    is_fix: bool = False,
) -> str:
    decision = "РАЗРЕШЕНО" if verdict.allowed else "ЗАБЛОКИРОВАНО"
    p_domain = verdict.probabilities.get("domain", 0.0)
    p_ambig = verdict.probabilities.get("ambiguous", 0.0)
    p_offtopic = verdict.probabilities.get("offtopic", verdict.offtopic_prob)

    lines = [
        "> **Guard & RAG Debug**",
        (
            f"> **Guard:** **{decision}** (`{verdict.reason}`) · "
            f"`{verdict.model}` (`{verdict.latency_ms:.0f} мс`)"
        ),
        (
            f"> Класс: `{verdict.intent}` · "
            f"`domain={p_domain:.3f}` · `ambiguous={p_ambig:.3f}` · "
            f"`offtopic={p_offtopic:.3f}` · `noul={verdict.abuse_prob:.3f}`"
        ),
        (
            f"> Пороги Guard: `offtopic >= {verdict.offtopic_threshold:.2f}` "
            f"и `noul >= {verdict.noul_threshold:.2f}`"
        ),
    ]
    if verdict.error:
        safe_err = verdict.error.replace("\n", " ")[:180]
        lines.append(f"> Ошибка Guard (fail-open): `{safe_err}`")

    if not verdict.allowed:
        lines.append("> **RAG:** не запускался (запрос отклонён Guard)")
    elif stats is not None:
        prefix = "**RAG (fix):**" if is_fix else "**RAG:**"
        cited_part = (
            f" -> процитировано `{stats.cited_count}`"
            if stats.cited_count is not None
            else ""
        )
        lines.append(
            f"> {prefix} кандидатов `{stats.candidates_count}` -> "
            f"отобрано `{stats.selected_count}`{cited_part}"
        )
        lines.append(
            f"> `top_score={stats.top_score:.3f}` (`cos={stats.top_cosine:.3f}`) · "
            f"отсечка `{stats.cutoff_score:.3f}` "
            f"(`min={stats.min_score:.2f}`, `rel={stats.rel_ratio:.2f}`)"
        )
    else:
        lines.append("> **RAG:** статистика недоступна")

    return "\n".join(lines)


class TopicGuard:
    def __init__(self, polza_client: PolzaClient) -> None:
        self._polza = polza_client

    def build_query_state(
        self,
        query: str,
        *,
        history: list[dict[str, str]] | list[ChatMessageInput] | None = None,
        page_context: PageContext | None = None,
    ) -> str:
        parts: list[str] = []
        if page_context and (page_context.title or page_context.location):
            parts.append(
                f"Открытая страница документации: {page_context.title or ''} "
                f"({page_context.location or ''})".strip()
            )

        if history:
            recent_turns: list[str] = []
            for msg in history[-4:]:
                if isinstance(msg, ChatMessageInput):
                    role_raw, content = msg.role, msg.content
                else:
                    role_raw, content = msg.get("role", "user"), msg.get("content", "")
                role = "Пользователь" if role_raw == "user" else "Ассистент"
                snippet = content.strip()[:300]
                if snippet:
                    recent_turns.append(f"{role}: {snippet}")
            if recent_turns:
                parts.append("Контекст предыдущего диалога:\n" + "\n".join(recent_turns))

        parts.append(f"Последний запрос пользователя:\n{query}")
        return "\n\n".join(parts)

    async def evaluate(
        self,
        *,
        state: str,
        questions: dict[str, Any],
        model: str,
        enabled: bool,
        offtopic_threshold: float,
        noul_threshold: float,
        timeout_seconds: float,
        preview_label: str,
    ) -> GuardVerdictSchema:
        if not enabled:
            return GuardVerdictSchema(
                allowed=True,
                reason="disabled",
                model=model,
                intent="bypass",
                offtopic_threshold=offtopic_threshold,
                noul_threshold=noul_threshold,
            )

        started = time.monotonic()
        try:
            data = await self._polza.evaluate_decision(
                model=model,
                state=state,
                questions=questions,
                timeout_seconds=timeout_seconds,
            )
            latency_ms = (time.monotonic() - started) * 1000.0

            answers = data.get("answers") or {}
            intent_ans = answers.get("intent") or {}
            chosen_intent = str(intent_ans.get("choice") or "ambiguous")
            raw_probs = intent_ans.get("probabilities") or {}
            probabilities = {
                str(k): float(v)
                for k, v in raw_probs.items()
                if isinstance(v, (int, float))
            }
            offtopic_prob = probabilities.get("offtopic", 0.0)

            noul_ans = answers.get("definitely_abuse") or {}
            abuse_prob = float(noul_ans.get("noul", 0.0))

            blocked = (
                chosen_intent == "offtopic"
                and offtopic_prob >= offtopic_threshold
                and abuse_prob >= noul_threshold
            )
            reason = "blocked_thresholds" if blocked else "allowed"

            log_msg = (
                f"TopicGuard [{reason}] model={model} intent={chosen_intent} "
                f"probs={probabilities} noul={abuse_prob:.3f} "
                f"latency={latency_ms:.0f}ms {preview_label}"
            )
            if blocked:
                logger.info(log_msg)
            else:
                logger.debug(log_msg)

            return GuardVerdictSchema(
                allowed=not blocked,
                reason=reason,
                model=str(data.get("model") or model),
                intent=chosen_intent,
                probabilities=probabilities,
                offtopic_prob=offtopic_prob,
                abuse_prob=abuse_prob,
                offtopic_threshold=offtopic_threshold,
                noul_threshold=noul_threshold,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - started) * 1000.0
            logger.warning(f"TopicGuard failed (fail-open, {latency_ms:.0f}ms): {exc}")
            return GuardVerdictSchema(
                allowed=True,
                reason="fail_open_error",
                model=model,
                intent="error",
                offtopic_threshold=offtopic_threshold,
                noul_threshold=noul_threshold,
                latency_ms=latency_ms,
                error=str(exc),
            )

    async def check_query(
        self,
        *,
        query: str,
        profile: dict[str, Any],
        history: list[dict[str, str]] | None = None,
        page_context: PageContext | None = None,
    ) -> GuardVerdictSchema:
        if not query.strip():
            return GuardVerdictSchema(
                allowed=True,
                reason="empty_query",
                model=profile["decision_model"],
                intent="bypass",
                offtopic_threshold=profile["guard_offtopic_threshold"],
                noul_threshold=profile["guard_noul_threshold"],
            )
        state = self.build_query_state(query, history=history, page_context=page_context)
        return await self.evaluate(
            state=state,
            questions=profile["decision_questions"],
            model=profile["decision_model"],
            enabled=profile["guard_enabled"],
            offtopic_threshold=profile["guard_offtopic_threshold"],
            noul_threshold=profile["guard_noul_threshold"],
            timeout_seconds=profile["guard_timeout_seconds"],
            preview_label=f"profile={profile['profile_id']} query={query[:120]!r}",
        )

    async def check_code_fix(
        self,
        *,
        source_code: str,
        error_text: str,
        profile: dict[str, Any],
    ) -> GuardVerdictSchema:
        state = (
            f"Код, отправленный на исправление:\n{source_code[:2000]}\n\n"
            f"Ошибка компилятора Typst:\n{error_text[:500]}"
        )
        return await self.evaluate(
            state=state,
            questions=profile["decision_questions"],
            model=profile["decision_model"],
            enabled=profile["guard_enabled"],
            offtopic_threshold=profile["guard_offtopic_threshold"],
            noul_threshold=profile["guard_noul_threshold"],
            timeout_seconds=profile["guard_timeout_seconds"],
            preview_label=f"profile={profile['profile_id']} fix_code={source_code[:80]!r}",
        )