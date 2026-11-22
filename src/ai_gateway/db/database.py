from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import aiosqlite
from loguru import logger

from ai_gateway.config import Settings
from ai_gateway.db.defaults import build_default_gateway_settings, build_default_profiles


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        return None


class GatewayDatabase:
    def __init__(self, sqlite_path: Path, settings: Settings) -> None:
        self._path = sqlite_path
        self._settings = settings
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA foreign_keys=ON;")
        await self._init_schema()
        await self._seed_defaults()
        logger.info(f"SQLite database connected: {self._path}")

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database is not connected")
        return self._conn

    async def _init_schema(self) -> None:
        await self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS gateway_settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profile_configs (
                profile_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                llm_model TEXT NOT NULL,
                decision_model TEXT NOT NULL,
                temperature REAL NOT NULL,
                max_tokens INTEGER,
                system_prompt TEXT NOT NULL,
                decision_questions_json TEXT NOT NULL,
                refusal_reply TEXT NOT NULL,
                no_docs_reply TEXT NOT NULL,
                guard_enabled INTEGER NOT NULL,
                guard_offtopic_threshold REAL NOT NULL,
                guard_noul_threshold REAL NOT NULL,
                guard_timeout_seconds REAL NOT NULL,
                guard_debug INTEGER NOT NULL,
                max_query_chars INTEGER NOT NULL,
                use_history INTEGER NOT NULL,
                history_max_messages INTEGER NOT NULL,
                history_ttl_seconds INTEGER NOT NULL,
                include_assets_in_prompt INTEGER NOT NULL,
                source_url_mode TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                channel TEXT NOT NULL,
                external_user_id TEXT,
                external_chat_id TEXT,
                title TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                context_reset_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_conversations_channel_updated
                ON conversations(channel, updated_at DESC);
            CREATE INDEX IF NOT EXISTS idx_conversations_external_user
                ON conversations(external_user_id);

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                llm_model TEXT,
                decision_model TEXT,
                guard_allowed INTEGER,
                guard_intent TEXT,
                guard_offtopic_prob REAL,
                guard_noul_prob REAL,
                guard_latency_ms REAL,
                rag_candidates INTEGER,
                rag_selected INTEGER,
                rag_cited INTEGER,
                rag_top_score REAL,
                sources_json TEXT,
                usage_json TEXT,
                latency_ms REAL,
                stopped INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv_created
                ON messages(conversation_id, id ASC);

            CREATE TABLE IF NOT EXISTS reindex_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT NOT NULL,
                trigger TEXT NOT NULL,
                files_processed INTEGER NOT NULL DEFAULT 0,
                chunks_count INTEGER NOT NULL DEFAULT 0,
                embeddings_fetched INTEGER NOT NULL DEFAULT 0,
                assets_synced INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT
            );
            """
        )
        await self.conn.commit()

    async def _seed_defaults(self) -> None:
        now = _utc_now_iso()
        default_settings = build_default_gateway_settings(self._settings)
        for key, value in default_settings.items():
            await self.conn.execute(
                """
                INSERT OR IGNORE INTO gateway_settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                """,
                (key, json.dumps(value, ensure_ascii=False), now),
            )

        default_profiles = build_default_profiles(self._settings)
        for profile in default_profiles.values():
            await self._insert_profile_if_missing(profile, now)

        await self.conn.commit()

    async def _insert_profile_if_missing(self, profile: dict[str, Any], now: str) -> None:
        await self.conn.execute(
            """
            INSERT OR IGNORE INTO profile_configs (
                profile_id, display_name, description, enabled,
                llm_model, decision_model, temperature, max_tokens,
                system_prompt, decision_questions_json,
                refusal_reply, no_docs_reply,
                guard_enabled, guard_offtopic_threshold, guard_noul_threshold,
                guard_timeout_seconds, guard_debug,
                max_query_chars, use_history, history_max_messages, history_ttl_seconds,
                include_assets_in_prompt, source_url_mode, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile["profile_id"],
                profile["display_name"],
                profile["description"],
                int(profile["enabled"]),
                profile["llm_model"],
                profile["decision_model"],
                float(profile["temperature"]),
                profile["max_tokens"],
                profile["system_prompt"],
                json.dumps(profile["decision_questions"], ensure_ascii=False),
                profile["refusal_reply"],
                profile["no_docs_reply"],
                int(profile["guard_enabled"]),
                float(profile["guard_offtopic_threshold"]),
                float(profile["guard_noul_threshold"]),
                float(profile["guard_timeout_seconds"]),
                int(profile["guard_debug"]),
                int(profile["max_query_chars"]),
                int(profile["use_history"]),
                int(profile["history_max_messages"]),
                int(profile["history_ttl_seconds"]),
                int(profile["include_assets_in_prompt"]),
                profile["source_url_mode"],
                now,
            ),
        )

    async def get_gateway_settings(self) -> dict[str, Any]:
        defaults = build_default_gateway_settings(self._settings)
        async with self.conn.execute("SELECT key, value_json FROM gateway_settings") as cur:
            rows = await cur.fetchall()
        for row in rows:
            defaults[row["key"]] = json.loads(row["value_json"])
        return defaults

    async def update_gateway_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        now = _utc_now_iso()
        for key, value in updates.items():
            await self.conn.execute(
                """
                INSERT INTO gateway_settings (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), now),
            )
        await self.conn.commit()
        return await self.get_gateway_settings()

    def _row_to_profile(self, row: aiosqlite.Row) -> dict[str, Any]:
        return {
            "profile_id": row["profile_id"],
            "display_name": row["display_name"],
            "description": row["description"],
            "enabled": bool(row["enabled"]),
            "llm_model": row["llm_model"],
            "decision_model": row["decision_model"],
            "temperature": float(row["temperature"]),
            "max_tokens": row["max_tokens"],
            "system_prompt": row["system_prompt"],
            "decision_questions": json.loads(row["decision_questions_json"]),
            "refusal_reply": row["refusal_reply"],
            "no_docs_reply": row["no_docs_reply"],
            "guard_enabled": bool(row["guard_enabled"]),
            "guard_offtopic_threshold": float(row["guard_offtopic_threshold"]),
            "guard_noul_threshold": float(row["guard_noul_threshold"]),
            "guard_timeout_seconds": float(row["guard_timeout_seconds"]),
            "guard_debug": bool(row["guard_debug"]),
            "max_query_chars": int(row["max_query_chars"]),
            "use_history": bool(row["use_history"]),
            "history_max_messages": int(row["history_max_messages"]),
            "history_ttl_seconds": int(row["history_ttl_seconds"]),
            "include_assets_in_prompt": bool(row["include_assets_in_prompt"]),
            "source_url_mode": row["source_url_mode"],
            "updated_at": row["updated_at"],
        }

    async def list_profiles(self) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM profile_configs ORDER BY profile_id ASC"
        ) as cur:
            rows = await cur.fetchall()
        return [self._row_to_profile(row) for row in rows]

    async def get_profile(self, profile_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM profile_configs WHERE profile_id = ?",
            (profile_id,),
        ) as cur:
            row = await cur.fetchone()
        return self._row_to_profile(row) if row is not None else None

    async def update_profile(self, profile_id: str, updates: dict[str, Any]) -> dict[str, Any]:
        if not updates:
            existing = await self.get_profile(profile_id)
            if existing is None:
                raise KeyError(profile_id)
            return existing

        col_map: dict[str, Any] = {}
        bool_cols = {
            "enabled",
            "guard_enabled",
            "guard_debug",
            "use_history",
            "include_assets_in_prompt",
        }
        for key, value in updates.items():
            if key == "decision_questions":
                col_map["decision_questions_json"] = json.dumps(value, ensure_ascii=False)
            elif key in bool_cols:
                col_map[key] = int(bool(value))
            else:
                col_map[key] = value

        col_map["updated_at"] = _utc_now_iso()
        set_clause = ", ".join(f"{col} = ?" for col in col_map)
        params = list(col_map.values()) + [profile_id]

        cur = await self.conn.execute(
            f"UPDATE profile_configs SET {set_clause} WHERE profile_id = ?",
            params,
        )
        await self.conn.commit()
        if cur.rowcount == 0:
            raise KeyError(profile_id)

        updated = await self.get_profile(profile_id)
        if updated is None:
            raise KeyError(profile_id)
        return updated

    async def update_telegram_profiles(self, updates: dict[str, Any]) -> list[dict[str, Any]]:
        tg_ids = ("telegram_private", "telegram_group", "telegram_fix")
        results: list[dict[str, Any]] = []
        for pid in tg_ids:
            results.append(await self.update_profile(pid, updates))
        return results

    async def reset_profile(self, profile_id: str) -> dict[str, Any]:
        defaults = build_default_profiles(self._settings)
        if profile_id not in defaults:
            raise KeyError(profile_id)
        profile = defaults[profile_id]
        now = _utc_now_iso()
        await self.conn.execute(
            """
            INSERT OR REPLACE INTO profile_configs (
                profile_id, display_name, description, enabled,
                llm_model, decision_model, temperature, max_tokens,
                system_prompt, decision_questions_json,
                refusal_reply, no_docs_reply,
                guard_enabled, guard_offtopic_threshold, guard_noul_threshold,
                guard_timeout_seconds, guard_debug,
                max_query_chars, use_history, history_max_messages, history_ttl_seconds,
                include_assets_in_prompt, source_url_mode, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile["profile_id"],
                profile["display_name"],
                profile["description"],
                int(profile["enabled"]),
                profile["llm_model"],
                profile["decision_model"],
                float(profile["temperature"]),
                profile["max_tokens"],
                profile["system_prompt"],
                json.dumps(profile["decision_questions"], ensure_ascii=False),
                profile["refusal_reply"],
                profile["no_docs_reply"],
                int(profile["guard_enabled"]),
                float(profile["guard_offtopic_threshold"]),
                float(profile["guard_noul_threshold"]),
                float(profile["guard_timeout_seconds"]),
                int(profile["guard_debug"]),
                int(profile["max_query_chars"]),
                int(profile["use_history"]),
                int(profile["history_max_messages"]),
                int(profile["history_ttl_seconds"]),
                int(profile["include_assets_in_prompt"]),
                profile["source_url_mode"],
                now,
            ),
        )
        await self.conn.commit()
        restored = await self.get_profile(profile_id)
        assert restored is not None
        return restored

    async def ensure_conversation(
        self,
        *,
        conversation_id: str,
        channel: str,
        external_user_id: str | None = None,
        external_chat_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = _utc_now_iso()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        clean_title = title[:120].strip() if title else None

        await self.conn.execute(
            """
            INSERT INTO conversations (
                id, channel, external_user_id, external_chat_id, title,
                metadata_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                external_user_id = COALESCE(excluded.external_user_id, conversations.external_user_id),
                external_chat_id = COALESCE(excluded.external_chat_id, conversations.external_chat_id),
                title = COALESCE(conversations.title, excluded.title),
                metadata_json = CASE
                    WHEN excluded.metadata_json != '{}' THEN excluded.metadata_json
                    ELSE conversations.metadata_json
                END,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                channel,
                external_user_id,
                external_chat_id,
                clean_title,
                meta_json,
                now,
                now,
            ),
        )
        await self.conn.commit()

    async def get_conversation_history(
        self,
        conversation_id: str,
        *,
        max_messages: int,
        ttl_seconds: int,
    ) -> list[dict[str, str]]:
        if max_messages <= 0:
            return []

        async with self.conn.execute(
            "SELECT context_reset_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cur:
            conv_row = await cur.fetchone()

        if conv_row is None:
            return []

        reset_at = conv_row["context_reset_at"]
        query = (
            "SELECT role, content, created_at FROM messages "
            "WHERE conversation_id = ? AND role IN ('user', 'assistant')"
        )
        params: list[Any] = [conversation_id]
        if reset_at:
            query += " AND created_at > ?"
            params.append(reset_at)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(max_messages)

        async with self.conn.execute(query, params) as cur:
            rows = await cur.fetchall()

        if not rows:
            return []

        latest_dt = _parse_iso(rows[0]["created_at"])
        if latest_dt is not None:
            if (datetime.now(UTC) - latest_dt).total_seconds() > ttl_seconds:
                return []

        ordered = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in ordered]

    async def reset_conversation_context(self, conversation_id: str) -> bool:
        now = _utc_now_iso()
        cur = await self.conn.execute(
            "UPDATE conversations SET context_reset_at = ?, updated_at = ? WHERE id = ?",
            (now, now, conversation_id),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def append_turn_messages(
        self,
        *,
        conversation_id: str,
        user_content: str,
        assistant_content: str,
        llm_model: str,
        decision_model: str,
        guard_allowed: bool,
        guard_intent: str,
        guard_offtopic_prob: float,
        guard_noul_prob: float,
        guard_latency_ms: float,
        rag_candidates: int | None,
        rag_selected: int | None,
        rag_cited: int | None,
        rag_top_score: float | None,
        sources: list[dict[str, Any]],
        usage: dict[str, Any] | None,
        latency_ms: float,
        stopped: bool = False,
        error: str | None = None,
    ) -> None:
        now_dt = datetime.now(UTC)
        user_ts = now_dt.isoformat()
        assistant_ts = (now_dt + timedelta(microseconds=1)).isoformat()

        await self.conn.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, created_at
            ) VALUES (?, 'user', ?, ?)
            """,
            (conversation_id, user_content, user_ts),
        )
        await self.conn.execute(
            """
            INSERT INTO messages (
                conversation_id, role, content, llm_model, decision_model,
                guard_allowed, guard_intent, guard_offtopic_prob, guard_noul_prob,
                guard_latency_ms, rag_candidates, rag_selected, rag_cited,
                rag_top_score, sources_json, usage_json, latency_ms,
                stopped, error, created_at
            ) VALUES (?, 'assistant', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                assistant_content,
                llm_model,
                decision_model,
                int(guard_allowed),
                guard_intent,
                guard_offtopic_prob,
                guard_noul_prob,
                guard_latency_ms,
                rag_candidates,
                rag_selected,
                rag_cited,
                rag_top_score,
                json.dumps(sources, ensure_ascii=False),
                json.dumps(usage, ensure_ascii=False) if usage else None,
                latency_ms,
                int(stopped),
                error,
                assistant_ts,
            ),
        )
        await self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (assistant_ts, conversation_id),
        )
        await self.conn.commit()

    async def list_conversations(
        self,
        *,
        channel: str | None = None,
        search: str | None = None,
        external_user_id: str | None = None,
        guard_allowed: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        where_clauses: list[str] = []
        params: list[Any] = []

        if channel:
            where_clauses.append("c.channel = ?")
            params.append(channel)
        if external_user_id:
            where_clauses.append("c.external_user_id = ?")
            params.append(external_user_id)
        if search:
            like = f"%{search}%"
            where_clauses.append(
                "(c.title LIKE ? OR EXISTS (SELECT 1 FROM messages m2 "
                "WHERE m2.conversation_id = c.id AND m2.content LIKE ?))"
            )
            params.extend([like, like])
        if guard_allowed is not None:
            where_clauses.append(
                "EXISTS (SELECT 1 FROM messages m3 WHERE m3.conversation_id = c.id "
                "AND m3.role = 'assistant' AND m3.guard_allowed = ?)"
            )
            params.append(int(guard_allowed))

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        count_sql = f"SELECT COUNT(*) AS cnt FROM conversations c {where_sql}"
        async with self.conn.execute(count_sql, params) as cur:
            total_row = await cur.fetchone()
        total = int(total_row["cnt"]) if total_row else 0

        list_sql = f"""
            SELECT
                c.*,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                (
                    SELECT m.content FROM messages m
                    WHERE m.conversation_id = c.id
                    ORDER BY m.id DESC LIMIT 1
                ) AS last_message_preview
            FROM conversations c
            {where_sql}
            ORDER BY c.updated_at DESC
            LIMIT ? OFFSET ?
        """
        async with self.conn.execute(list_sql, [*params, limit, offset]) as cur:
            rows = await cur.fetchall()

        items = [
            {
                "id": row["id"],
                "channel": row["channel"],
                "external_user_id": row["external_user_id"],
                "external_chat_id": row["external_chat_id"],
                "title": row["title"],
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "message_count": int(row["message_count"] or 0),
                "last_message_preview": (row["last_message_preview"] or "")[:200],
                "context_reset_at": row["context_reset_at"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]
        return items, total

    async def get_conversation_detail(self, conversation_id: str) -> dict[str, Any] | None:
        async with self.conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,),
        ) as cur:
            conv = await cur.fetchone()
        if conv is None:
            return None

        async with self.conn.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ) as cur:
            msg_rows = await cur.fetchall()

        messages = [
            {
                "id": row["id"],
                "role": row["role"],
                "content": row["content"],
                "llm_model": row["llm_model"],
                "decision_model": row["decision_model"],
                "guard_allowed": (
                    bool(row["guard_allowed"]) if row["guard_allowed"] is not None else None
                ),
                "guard_intent": row["guard_intent"],
                "guard_offtopic_prob": row["guard_offtopic_prob"],
                "guard_noul_prob": row["guard_noul_prob"],
                "guard_latency_ms": row["guard_latency_ms"],
                "rag_candidates": row["rag_candidates"],
                "rag_selected": row["rag_selected"],
                "rag_cited": row["rag_cited"],
                "rag_top_score": row["rag_top_score"],
                "sources": json.loads(row["sources_json"]) if row["sources_json"] else [],
                "usage": json.loads(row["usage_json"]) if row["usage_json"] else None,
                "latency_ms": row["latency_ms"],
                "stopped": bool(row["stopped"]),
                "error": row["error"],
                "created_at": row["created_at"],
            }
            for row in msg_rows
        ]
        return {
            "id": conv["id"],
            "channel": conv["channel"],
            "external_user_id": conv["external_user_id"],
            "external_chat_id": conv["external_chat_id"],
            "title": conv["title"],
            "metadata": json.loads(conv["metadata_json"] or "{}"),
            "context_reset_at": conv["context_reset_at"],
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "messages": messages,
        }

    async def delete_conversation(self, conversation_id: str) -> bool:
        cur = await self.conn.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )
        await self.conn.commit()
        return cur.rowcount > 0

    async def get_analytics(self, *, days: int | None = None) -> dict[str, Any]:
        since_clause = ""
        params: list[Any] = []
        if days is not None and days > 0:
            since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            since_clause = "WHERE created_at >= ?"
            params.append(since)

        async with self.conn.execute(
            f"SELECT COUNT(*) AS cnt FROM conversations {since_clause}",
            params,
        ) as cur:
            total_convs = int((await cur.fetchone())["cnt"])

        msg_where = "WHERE role = 'assistant'"
        if days is not None and days > 0:
            msg_where += " AND created_at >= ?"

        async with self.conn.execute(
            f"""
            SELECT
                COUNT(*) AS total_turns,
                SUM(CASE WHEN guard_allowed = 0 THEN 1 ELSE 0 END) AS blocked_count,
                SUM(CASE WHEN guard_allowed = 1 AND rag_selected = 0 THEN 1 ELSE 0 END) AS no_docs_count,
                AVG(latency_ms) AS avg_latency_ms
            FROM messages
            {msg_where}
            """,
            params,
        ) as cur:
            summary = await cur.fetchone()

        async with self.conn.execute(
            f"""
            SELECT c.channel, COUNT(m.id) AS turns
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            {msg_where.replace('created_at', 'm.created_at')}
            GROUP BY c.channel
            """,
            params,
        ) as cur:
            channel_rows = await cur.fetchall()

        async with self.conn.execute(
            f"SELECT usage_json FROM messages {msg_where} AND usage_json IS NOT NULL",
            params,
        ) as cur:
            usage_rows = await cur.fetchall()

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        for row in usage_rows:
            try:
                u = json.loads(row["usage_json"])
            except ValueError:
                continue
            prompt_tokens += int(u.get("prompt_tokens") or 0)
            completion_tokens += int(u.get("completion_tokens") or 0)
            total_tokens += int(u.get("total_tokens") or 0)
            cost_val = u.get("cost_rub") if u.get("cost_rub") is not None else u.get("cost")
            if isinstance(cost_val, (int, float)):
                total_cost += float(cost_val)

        return {
            "period_days": days,
            "total_conversations": total_convs,
            "total_turns": int(summary["total_turns"] or 0),
            "guard_blocked_count": int(summary["blocked_count"] or 0),
            "no_docs_count": int(summary["no_docs_count"] or 0),
            "avg_latency_ms": round(float(summary["avg_latency_ms"] or 0.0), 1),
            "by_channel": {row["channel"]: int(row["turns"]) for row in channel_rows},
            "tokens": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated_cost": round(total_cost, 6),
            },
        }

    async def create_reindex_job(self, trigger: str) -> int:
        now = _utc_now_iso()
        cur = await self.conn.execute(
            "INSERT INTO reindex_jobs (status, trigger, started_at) VALUES ('running', ?, ?)",
            (trigger, now),
        )
        await self.conn.commit()
        assert cur.lastrowid is not None
        return int(cur.lastrowid)

    async def finish_reindex_job(
        self,
        job_id: int,
        *,
        status: str,
        files_processed: int = 0,
        chunks_count: int = 0,
        embeddings_fetched: int = 0,
        assets_synced: int = 0,
        error: str | None = None,
    ) -> None:
        now = _utc_now_iso()
        await self.conn.execute(
            """
            UPDATE reindex_jobs SET
                status = ?,
                files_processed = ?,
                chunks_count = ?,
                embeddings_fetched = ?,
                assets_synced = ?,
                error = ?,
                finished_at = ?
            WHERE id = ?
            """,
            (
                status,
                files_processed,
                chunks_count,
                embeddings_fetched,
                assets_synced,
                error,
                now,
                job_id,
            ),
        )
        await self.conn.commit()

    async def list_reindex_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self.conn.execute(
            "SELECT * FROM reindex_jobs ORDER BY id DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
        return [dict(row) for row in rows]