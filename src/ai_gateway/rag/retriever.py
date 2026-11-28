from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import lancedb
from lancedb.rerankers import RRFReranker
from loguru import logger

from ai_gateway.rag.query_builder import (
    GOST_HINT_RE,
    PARAMS_HINT_RE,
    PARAMS_SECTION_RE,
    SYMBOL_QUERY_RE,
    VERSION_QUERY_RE,
)
from ai_gateway.schemas import DocSourceSchema, RetrievalStatsSchema

RRF_MAX_THEORETICAL = 2.0 / 61.0
PART_SUFFIX_RE = re.compile(r"\s*\(часть \d+ из \d+\)$")


def _cosine_similarity(a: list[float], b: Any) -> float | None:
    if b is None:
        return None
    try:
        if len(b) != len(a) or not a:
            return None
        dot = 0.0
        norm_a = 0.0
        norm_b = 0.0
        for va, vb in zip(a, b):
            fa = float(va)
            fb = float(vb)
            dot += fa * fb
            norm_a += fa * fa
            norm_b += fb * fb
        denom = math.sqrt(norm_a) * math.sqrt(norm_b)
        if denom <= 0.0:
            return None
        return dot / denom
    except (TypeError, ValueError):
        return None


def normalize_active_doc_url(location: str | None, docs_base_url: str) -> str | None:
    if not location:
        return None
    parsed = urlparse(location)
    if parsed.path:
        return f"{docs_base_url.rstrip('/')}{parsed.path.rstrip('/')}"
    return None


@dataclass
class _SourceAccumulator:
    label: int
    url: str
    path: str
    doc_url: str
    title: str
    category: str
    score: float
    cosine: float
    matched_orders: set[int] = field(default_factory=set)
    fallback_chunks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalRetrievalResult:
    retrieved_chunk_ids: list[str]
    context_chunk_ids: set[str] | None
    candidates_count: int
    selected_sources_count: int | None
    top_score: float | None
    cutoff_score: float | None
    latency_ms: float


class DocsRetriever:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._table: lancedb.table.Table | None = None
        self._reranker = RRFReranker()
        self._lock = threading.RLock()

    def _get_table(self) -> lancedb.table.Table:
        with self._lock:
            if self._table is not None:
                return self._table
            if not (self._db_path / "docs.lance").exists():
                raise FileNotFoundError(
                    f"Vector database table 'docs.lance' not found in: {self._db_path}"
                )
            db = lancedb.connect(str(self._db_path))
            self._table = db.open_table("docs")
            return self._table

    def is_ready(self) -> bool:
        try:
            self._get_table()
            return True
        except Exception:
            return False

    def count_rows(self) -> int:
        try:
            table = self._get_table()
            return int(table.count_rows())
        except Exception:
            return 0

    def unload(self) -> None:
        with self._lock:
            self._table = None
            logger.debug("DocsRetriever table reference unloaded")

    def reload(self) -> None:
        with self._lock:
            self._table = None
            if (self._db_path / "docs.lance").exists():
                db = lancedb.connect(str(self._db_path))
                self._table = db.open_table("docs")
                logger.info(f"DocsRetriever reloaded table from {self._db_path}")

    def _list_chunk_ids_sync(self) -> set[str]:
        table = self._get_table()
        total = max(1, int(table.count_rows()))
        rows = table.search().select(["chunk_id"]).limit(total).to_list()
        return {str(row["chunk_id"]).strip() for row in rows if row.get("chunk_id")}

    async def list_chunk_ids(self) -> set[str]:
        return await asyncio.to_thread(self._list_chunk_ids_sync)

    def _get_chunks_by_ids_sync(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        unique_ids = list(
            dict.fromkeys(str(cid).strip() for cid in chunk_ids if str(cid).strip())
        )[:200]
        if not unique_ids:
            return {}
        table = self._get_table()
        escaped = ", ".join("'" + cid.replace("'", "''") + "'" for cid in unique_ids)
        try:
            rows = (
                table.search()
                .where(f"chunk_id IN ({escaped})")
                .select(
                    [
                        "chunk_id",
                        "title",
                        "section",
                        "heading_path",
                        "category",
                        "url",
                        "text",
                    ]
                )
                .limit(len(unique_ids))
                .to_list()
            )
        except Exception as exc:
            logger.warning(f"Chunk lookup failed: {exc}")
            return {}

        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            cid = str(row.get("chunk_id") or "").strip()
            if not cid:
                continue
            text = str(row.get("text") or "")
            result[cid] = {
                "chunk_id": cid,
                "title": str(row.get("title") or ""),
                "section": PART_SUFFIX_RE.sub("", str(row.get("section") or "")),
                "heading_path": str(row.get("heading_path") or ""),
                "category": self._infer_category(row),
                "url": str(row.get("url") or ""),
                "text_preview": text[:600],
            }
        return result

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
        return await asyncio.to_thread(self._get_chunks_by_ids_sync, chunk_ids)

    async def retrieve_with_stats(
        self,
        *,
        raw_query: str,
        query_vector: list[float],
        fts_query: str,
        docs_base_url: str,
        min_score: float,
        rel_ratio: float,
        limit: int = 24,
        max_sources: int = 5,
        max_page_chars: int = 10000,
        max_total_context_chars: int = 24000,
        active_doc_url: str | None = None,
    ) -> tuple[list[DocSourceSchema], RetrievalStatsSchema]:
        candidates = await asyncio.to_thread(
            self._search_hybrid_sync,
            query_vector,
            fts_query,
            limit,
        )
        scored_rows = self._score_candidates(
            candidates,
            raw_query=raw_query,
            query_vector=query_vector,
            active_doc_url=active_doc_url,
        )
        sources, stats, _ = await asyncio.to_thread(
            self._filter_and_expand_sources_with_chunks_sync,
            scored_rows,
            docs_base_url=docs_base_url,
            min_score=min_score,
            rel_ratio=rel_ratio,
            max_sources=max_sources,
            max_page_chars=max_page_chars,
            max_total_context_chars=max_total_context_chars,
        )
        return sources, stats

    def _search_hybrid_sync(
        self,
        query_vector: list[float],
        fts_query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        table = self._get_table()
        if fts_query:
            try:
                return (
                    table.search(
                        query_type="hybrid",
                        vector_column_name="vector",
                        fts_columns="text",
                    )
                    .vector(query_vector)
                    .text(fts_query)
                    .rerank(self._reranker)
                    .limit(limit)
                    .to_list()
                )
            except Exception as exc:
                logger.warning(f"Hybrid search failed, falling back to vector search: {exc}")

        return table.search(query_vector).limit(limit).to_list()

    def _search_vector_sync(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[dict[str, Any]]:
        table = self._get_table()
        return table.search(query_vector).limit(limit).to_list()

    def _search_fts_sync(
        self,
        fts_query: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        if not fts_query.strip():
            return []
        table = self._get_table()
        try:
            return (
                table.search(fts_query, query_type="fts", fts_columns="text")
                .limit(limit)
                .to_list()
            )
        except Exception as exc:
            logger.warning(f"FTS search failed: {exc}")
            return []

    def _infer_category(self, row: dict[str, Any]) -> str:
        if row.get("category"):
            return str(row["category"])
        url = str(row.get("url", ""))
        if "/docs/typst-gost" in url:
            return "typst-gost"
        if "/docs/typst/changelog" in url:
            return "typst-changelog"
        if "/docs/typst/symbols" in url:
            return "typst-symbols"
        return "typst-reference"

    def _score_candidates(
        self,
        rows: list[dict[str, Any]],
        *,
        raw_query: str,
        query_vector: list[float],
        active_doc_url: str | None,
    ) -> list[tuple[float, float, dict[str, Any]]]:
        wants_changelog = bool(VERSION_QUERY_RE.search(raw_query))
        wants_symbols = bool(SYMBOL_QUERY_RE.search(raw_query))
        reading_typst_gost = bool(active_doc_url and "/docs/typst-gost" in active_doc_url)
        mentions_gost = bool(GOST_HINT_RE.search(raw_query))
        wants_params = bool(PARAMS_HINT_RE.search(raw_query))

        scored: list[tuple[float, float, dict[str, Any]]] = []
        for idx, row in enumerate(rows):
            category = self._infer_category(row)
            doc_url = str(row.get("doc_url") or str(row.get("url", "")).split("#")[0])
            relevance = row.get("_relevance_score")
            if isinstance(relevance, (int, float)):
                rrf_norm = min(1.0, max(0.0, float(relevance) / RRF_MAX_THEORETICAL))
            else:
                rrf_norm = 1.0 / (1.0 + idx * 0.15)

            cos_sim = _cosine_similarity(query_vector, row.get("vector"))
            if cos_sim is not None:
                clamped_cos = max(0.0, cos_sim)
                rrf_effective = rrf_norm * min(1.0, clamped_cos / 0.45)
                score = 0.72 * clamped_cos + 0.28 * rrf_effective
                cos_value = cos_sim
            else:
                score = rrf_norm
                cos_value = 0.0

            if category == "typst-changelog" and not wants_changelog:
                score *= 0.35
            if category == "typst-symbols" and not wants_symbols:
                score *= 0.40
            if category == "typst-gost" and (mentions_gost or reading_typst_gost):
                score *= 1.15
            if active_doc_url and doc_url == active_doc_url:
                score *= 1.12
            if wants_params and PARAMS_SECTION_RE.search(str(row.get("section", ""))):
                score *= 1.10

            scored.append((score, cos_value, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored

    def _fetch_siblings_sync(self, doc_url: str) -> list[dict[str, Any]]:
        table = self._get_table()
        safe_url = doc_url.replace("'", "''")
        try:
            rows = table.search().where(f"doc_url = '{safe_url}'").to_list()
            rows.sort(key=lambda r: int(r.get("chunk_order", 0)))
            return rows
        except Exception:
            return []

    def _normalize_urls(self, raw_url: str, docs_base_url: str) -> tuple[str, str]:
        base = docs_base_url.rstrip("/")
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            parsed = urlparse(raw_url)
            rel_path = parsed.path or "/"
            if parsed.fragment:
                rel_path = f"{rel_path}#{parsed.fragment}"
            return raw_url, rel_path
        rel_path = f"/{raw_url.lstrip('/')}"
        return f"{base}{rel_path}", rel_path

    def _filter_and_expand_sources_with_chunks_sync(
        self,
        scored_rows: list[tuple[float, float, dict[str, Any]]],
        *,
        docs_base_url: str,
        min_score: float,
        rel_ratio: float,
        max_sources: int,
        max_page_chars: int,
        max_total_context_chars: int,
    ) -> tuple[list[DocSourceSchema], RetrievalStatsSchema, set[str]]:
        groups: dict[str, _SourceAccumulator] = {}

        for score, cos_val, row in scored_rows:
            raw_url = str(row.get("url", ""))
            doc_url = str(row.get("doc_url") or raw_url.split("#")[0])
            group = groups.get(doc_url)

            if group is None:
                title = str(row.get("title", "Документация"))
                section = PART_SUFFIX_RE.sub("", str(row.get("section", "")))
                display_title = (
                    title
                    if not section or section in {title, "Введение"}
                    else f"{title} · {section}"
                )
                abs_url, rel_path = self._normalize_urls(raw_url, docs_base_url)
                group = _SourceAccumulator(
                    label=0,
                    url=abs_url,
                    path=rel_path,
                    doc_url=doc_url,
                    title=display_title,
                    category=self._infer_category(row),
                    score=score,
                    cosine=cos_val,
                )
                groups[doc_url] = group
            else:
                if score > group.score:
                    group.score = score
                    group.cosine = max(group.cosine, cos_val)

            order = row.get("chunk_order")
            if isinstance(order, int):
                group.matched_orders.add(order)
            group.fallback_chunks.append(row)

        ranked_groups = sorted(groups.values(), key=lambda g: g.score, reverse=True)
        candidates_count = len(ranked_groups)
        top_score = ranked_groups[0].score if ranked_groups else 0.0
        top_cosine = ranked_groups[0].cosine if ranked_groups else 0.0
        cutoff_score = max(min_score, top_score * rel_ratio)

        selected_groups = [g for g in ranked_groups if g.score >= cutoff_score][:max_sources]
        for idx, group in enumerate(selected_groups, start=1):
            group.label = idx

        stats = RetrievalStatsSchema(
            candidates_count=candidates_count,
            selected_count=len(selected_groups),
            top_score=top_score,
            top_cosine=top_cosine,
            cutoff_score=cutoff_score,
            min_score=min_score,
            rel_ratio=rel_ratio,
        )

        if not selected_groups:
            return [], stats, set()

        total_budget = max_total_context_chars
        sources: list[DocSourceSchema] = []
        included_chunk_ids: set[str] = set()

        for group in selected_groups:
            siblings = self._fetch_siblings_sync(group.doc_url)
            if not siblings:
                chosen = group.fallback_chunks
                text = "\n\n".join(str(c.get("text", "")) for c in chosen)
            else:
                full_text = "\n\n".join(str(c.get("text", "")) for c in siblings)
                if (
                    group.category == "typst-gost" or len(siblings) <= 6
                ) and len(full_text) <= max_page_chars:
                    chosen = siblings
                    text = full_text
                else:
                    wanted_orders: set[int] = set()
                    for order in group.matched_orders:
                        wanted_orders.update((order - 1, order, order + 1))
                    selected = [
                        c
                        for c in siblings
                        if not isinstance(c.get("chunk_order"), int)
                        or c["chunk_order"] in wanted_orders
                    ]
                    chosen = selected or group.fallback_chunks
                    text = "\n\n".join(str(c.get("text", "")) for c in chosen)[:max_page_chars]

            if len(text) > total_budget:
                text = text[: max(800, total_budget)]
            total_budget = max(0, total_budget - len(text))

            cursor = 0
            text_len = len(text)
            for chunk in chosen:
                chunk_text = str(chunk.get("text", ""))
                if cursor < text_len:
                    cid = str(chunk.get("chunk_id") or "").strip()
                    if cid:
                        included_chunk_ids.add(cid)
                cursor += len(chunk_text) + 2

            sources.append(
                DocSourceSchema(
                    label=group.label,
                    title=group.title,
                    url=group.url,
                    path=group.path,
                    doc_url=group.doc_url,
                    category=group.category,
                    score=group.score,
                    cosine=group.cosine,
                    content=text,
                )
            )

        return sources, stats, included_chunk_ids

    def _retrieve_for_eval_sync(
        self,
        *,
        raw_query: str,
        query_vector: list[float] | None,
        fts_query: str,
        search_mode: str,
        limit: int,
        docs_base_url: str,
        min_score: float,
        rel_ratio: float,
        max_sources: int,
        max_page_chars: int,
        max_total_context_chars: int,
        include_context_metrics: bool,
    ) -> EvalRetrievalResult:
        started = time.monotonic()
        if search_mode in ("hybrid_scored", "hybrid_raw"):
            if query_vector is None:
                raise ValueError("query_vector is required for hybrid search")
            rows = self._search_hybrid_sync(query_vector, fts_query, limit)
        elif search_mode == "vector":
            if query_vector is None:
                raise ValueError("query_vector is required for vector search")
            rows = self._search_vector_sync(query_vector, limit)
        elif search_mode == "fts":
            rows = self._search_fts_sync(fts_query, limit)
        else:
            raise ValueError(f"Unsupported search_mode: {search_mode}")

        scored_rows: list[tuple[float, float, dict[str, Any]]] | None = None
        if search_mode == "hybrid_scored" and query_vector is not None:
            scored_rows = self._score_candidates(
                rows,
                raw_query=raw_query,
                query_vector=query_vector,
                active_doc_url=None,
            )
            ranked_rows = [row for _, _, row in scored_rows]
        else:
            ranked_rows = rows

        retrieved_chunk_ids: list[str] = []
        seen_ids: set[str] = set()
        for row in ranked_rows:
            cid = str(row.get("chunk_id") or "").strip()
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                retrieved_chunk_ids.append(cid)

        context_chunk_ids: set[str] | None = None
        selected_sources_count: int | None = None
        top_score: float | None = None
        cutoff_score: float | None = None
        candidates_count = len(retrieved_chunk_ids)

        if include_context_metrics and scored_rows is not None:
            _, stats, context_chunk_ids = self._filter_and_expand_sources_with_chunks_sync(
                scored_rows,
                docs_base_url=docs_base_url,
                min_score=min_score,
                rel_ratio=rel_ratio,
                max_sources=max_sources,
                max_page_chars=max_page_chars,
                max_total_context_chars=max_total_context_chars,
            )
            selected_sources_count = stats.selected_count
            top_score = stats.top_score
            cutoff_score = stats.cutoff_score
            candidates_count = stats.candidates_count

        latency_ms = (time.monotonic() - started) * 1000.0
        return EvalRetrievalResult(
            retrieved_chunk_ids=retrieved_chunk_ids,
            context_chunk_ids=context_chunk_ids,
            candidates_count=candidates_count,
            selected_sources_count=selected_sources_count,
            top_score=top_score,
            cutoff_score=cutoff_score,
            latency_ms=latency_ms,
        )

    async def retrieve_for_eval(
        self,
        *,
        raw_query: str,
        query_vector: list[float] | None,
        fts_query: str,
        search_mode: str = "hybrid_scored",
        limit: int = 24,
        docs_base_url: str = "https://typst-gost.ru",
        min_score: float = 0.40,
        rel_ratio: float = 0.72,
        max_sources: int = 5,
        max_page_chars: int = 10000,
        max_total_context_chars: int = 24000,
        include_context_metrics: bool = True,
    ) -> EvalRetrievalResult:
        return await asyncio.to_thread(
            self._retrieve_for_eval_sync,
            raw_query=raw_query,
            query_vector=query_vector,
            fts_query=fts_query,
            search_mode=search_mode,
            limit=limit,
            docs_base_url=docs_base_url,
            min_score=min_score,
            rel_ratio=rel_ratio,
            max_sources=max_sources,
            max_page_chars=max_page_chars,
            max_total_context_chars=max_total_context_chars,
            include_context_metrics=include_context_metrics,
        )