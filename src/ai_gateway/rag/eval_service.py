from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from loguru import logger
from openai import BadRequestError

from ai_gateway.config import Settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.query_builder import expand_query, sanitize_fts_query
from ai_gateway.rag.retriever import DocsRetriever
from ai_gateway.schemas import EvalMetricsBlock, EvalQuestionInput, EvalRunRequest


def _hash_query(embedding_model: str, text: str) -> str:
    return hashlib.sha256(f"{embedding_model}\0{text}".encode("utf-8")).hexdigest()


def _normalize_ks(raw_ks: list[int]) -> list[int]:
    cleaned = sorted({k for k in raw_ks if 1 <= k <= 200})
    if not cleaned:
        raise ValueError("ks must contain at least one integer in range [1, 200]")
    return cleaned


def _dedup_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for raw in items:
        val = str(raw).strip()
        if val and val not in seen:
            seen.add(val)
            result.append(val)
    return result


def parse_eval_dataset(raw_payload: Any) -> list[EvalQuestionInput]:
    if isinstance(raw_payload, dict):
        items = raw_payload.get("questions")
    elif isinstance(raw_payload, list):
        items = raw_payload
    else:
        raise ValueError("Dataset JSON must be an object with 'questions' list or a JSON array")

    if not isinstance(items, list) or not items:
        raise ValueError("Dataset contains no questions")

    questions: list[EvalQuestionInput] = []
    for idx, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question #{idx} must be a JSON object")
        rel_ids = _dedup_keep_order(list(item.get("relevant_chunk_ids") or []))
        if not rel_ids:
            raise ValueError(f"Question #{idx} ({item.get('id', '?')}) has empty relevant_chunk_ids")
        q_obj = EvalQuestionInput(
            id=str(item.get("id") or f"q_{idx:03d}").strip(),
            query=str(item.get("query") or "").strip(),
            relevant_chunk_ids=rel_ids,
            category=str(item.get("category") or "unknown").strip(),
            query_type=str(item.get("query_type") or "unknown").strip(),
            difficulty=str(item.get("difficulty") or "unknown").strip(),
        )
        questions.append(q_obj)
    return questions


def _aggregate_metrics(records: list[dict[str, Any]], ks: list[int]) -> EvalMetricsBlock:
    n = len(records)
    if n == 0:
        zeros = {str(k): 0.0 for k in ks}
        return EvalMetricsBlock(
            count=0,
            mrr=0.0,
            mrr_at_k=dict(zeros),
            recall_at_k=dict(zeros),
            hit_rate_at_k=dict(zeros),
            avg_latency_ms=0.0,
        )

    mrr = sum(r["reciprocal_rank"] for r in records) / n
    mrr_at_k = {
        str(k): round(sum(r["mrr_at_k"][str(k)] for r in records) / n, 4)
        for k in ks
    }
    recall_at_k = {
        str(k): round(sum(r["recall_at_k"][str(k)] for r in records) / n, 4)
        for k in ks
    }
    hit_rate_at_k = {
        str(k): round(sum(r["hit_at_k"][str(k)] for r in records) / n, 4)
        for k in ks
    }
    avg_latency_ms = round(sum(r["latency_ms"] for r in records) / n, 2)

    ctx_records = [r for r in records if r.get("context_recall") is not None]
    if ctx_records:
        cn = len(ctx_records)
        context_recall = round(sum(r["context_recall"] for r in ctx_records) / cn, 4)
        context_hit_rate = round(sum(r["context_hit"] for r in ctx_records) / cn, 4)
        no_docs_rate = round(sum(1 for r in ctx_records if r.get("no_docs")) / cn, 4)
        avg_selected_sources = round(
            sum(r["selected_sources_count"] for r in ctx_records) / cn, 2
        )
    else:
        context_recall = None
        context_hit_rate = None
        no_docs_rate = None
        avg_selected_sources = None

    return EvalMetricsBlock(
        count=n,
        mrr=round(mrr, 4),
        mrr_at_k=mrr_at_k,
        recall_at_k=recall_at_k,
        hit_rate_at_k=hit_rate_at_k,
        context_recall=context_recall,
        context_hit_rate=context_hit_rate,
        no_docs_rate=no_docs_rate,
        avg_selected_sources=avg_selected_sources,
        avg_latency_ms=avg_latency_ms,
    )


class EvalService:
    def __init__(
        self,
        settings: Settings,
        db: GatewayDatabase,
        polza: PolzaClient,
        retriever: DocsRetriever,
    ) -> None:
        self._settings = settings
        self._db = db
        self._polza = polza
        self._retriever = retriever
        self._is_running = False
        self._cache_lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._is_running

    def resolve_dataset_path(self, custom_path: str | None = None) -> Path:
        if not custom_path:
            return self._settings.eval_dataset_path
        candidate = Path(custom_path)
        if not candidate.is_absolute():
            candidate = self._settings.data_dir / candidate
        resolved = candidate.resolve()
        data_root = self._settings.data_dir.resolve()
        default_resolved = self._settings.eval_dataset_path.resolve()
        if not resolved.is_relative_to(data_root) and resolved != default_resolved:
            raise ValueError(f"Dataset path must stay inside DATA_DIR ({data_root})")
        return resolved

    def load_dataset(self, custom_path: str | None = None) -> tuple[Path, list[EvalQuestionInput]]:
        path = self.resolve_dataset_path(custom_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found: {path}")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise ValueError(f"Invalid JSON in evaluation dataset {path}: {exc}") from exc
        return path, parse_eval_dataset(raw)

    def save_dataset(
        self,
        questions: list[EvalQuestionInput],
        custom_path: str | None = None,
    ) -> Path:
        path = self.resolve_dataset_path(custom_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        normalized: list[EvalQuestionInput] = []
        for q in questions:
            rel_ids = _dedup_keep_order(q.relevant_chunk_ids)
            if not rel_ids:
                raise ValueError(f"Question '{q.id}' has empty relevant_chunk_ids")
            normalized.append(q.model_copy(update={"relevant_chunk_ids": rel_ids}))
        payload = {"questions": [q.model_dump() for q in normalized]}
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
                tmp_file.write("\n")
            os.replace(tmp_name, path)
        except Exception:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
            raise
        return path

    async def inspect_dataset(
        self,
        *,
        custom_path: str | None = None,
        include_questions: bool = False,
    ) -> dict[str, Any]:
        path = self.resolve_dataset_path(custom_path)
        docs_ready = self._retriever.is_ready()
        index_chunks_count = self._retriever.count_rows() if docs_ready else 0

        if not path.exists():
            return {
                "exists": False,
                "path": str(path),
                "total_questions": 0,
                "docs_ready": docs_ready,
                "index_chunks_count": index_chunks_count,
            }

        _, questions = self.load_dataset(custom_path)
        by_category = dict(Counter(q.category for q in questions))
        by_query_type = dict(Counter(q.query_type for q in questions))
        by_difficulty = dict(Counter(q.difficulty for q in questions))

        missing_total = 0
        questions_with_missing = 0
        sample_missing: list[str] = []

        if docs_ready:
            existing_ids = await self._retriever.list_chunk_ids()
            seen_missing: set[str] = set()
            for q in questions:
                q_missing = [cid for cid in q.relevant_chunk_ids if cid not in existing_ids]
                if q_missing:
                    questions_with_missing += 1
                    missing_total += len(q_missing)
                    for cid in q_missing:
                        if cid not in seen_missing:
                            seen_missing.add(cid)
                            if len(sample_missing) < 20:
                                sample_missing.append(cid)

        result: dict[str, Any] = {
            "exists": True,
            "path": str(path),
            "total_questions": len(questions),
            "by_category": by_category,
            "by_query_type": by_query_type,
            "by_difficulty": by_difficulty,
            "docs_ready": docs_ready,
            "index_chunks_count": index_chunks_count,
            "missing_chunk_ids_total": missing_total,
            "questions_with_missing_chunks": questions_with_missing,
            "sample_missing_chunk_ids": sample_missing,
        }
        if include_questions:
            result["questions"] = [q.model_dump() for q in questions]
        return result

    def _load_cache_sync(self) -> dict[str, list[float]]:
        cache: dict[str, list[float]] = {}
        cache_path = self._settings.eval_embeddings_cache_path
        if not cache_path.exists():
            return cache
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    cache[str(item["hash"])] = [float(x) for x in item["vector"]]
                except (ValueError, KeyError, TypeError):
                    continue
        return cache

    def _append_cache_sync(self, entries: list[tuple[str, list[float]]]) -> None:
        if not entries:
            return
        cache_path = self._settings.eval_embeddings_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8") as file:
            for hash_val, vector in entries:
                file.write(json.dumps({"hash": hash_val, "vector": vector}) + "\n")

    async def _resolve_query_embeddings(
        self,
        *,
        embedding_model: str,
        query_texts: list[str],
        batch_size: int,
        force_reembed: bool,
    ) -> tuple[dict[str, list[float]], int, int]:
        async with self._cache_lock:
            cache = {} if force_reembed else await asyncio.to_thread(self._load_cache_sync)

        unique_by_hash: dict[str, str] = {}
        for text in query_texts:
            h = _hash_query(embedding_model, text)
            if h not in unique_by_hash:
                unique_by_hash[h] = text

        missing = [(h, txt) for h, txt in unique_by_hash.items() if h not in cache]
        cached_hits = len(unique_by_hash) - len(missing)
        fetched = 0

        if missing:
            new_entries: list[tuple[str, list[float]]] = []
            for i in range(0, len(missing), batch_size):
                batch = missing[i : i + batch_size]
                batch_texts = [txt for _, txt in batch]
                try:
                    vectors = await self._polza.create_embeddings(embedding_model, batch_texts)
                except BadRequestError as exc:
                    logger.warning(f"Eval batch embedding error ({exc}), retrying one by one")
                    vectors = []
                    for txt in batch_texts:
                        vectors.append(await self._polza.create_embedding(embedding_model, txt))
                for (h, _), vec in zip(batch, vectors):
                    cache[h] = vec
                    new_entries.append((h, vec))
                    fetched += 1

            async with self._cache_lock:
                await asyncio.to_thread(self._append_cache_sync, new_entries)

        return cache, cached_hits, fetched

    def _prepare_run_inputs(
        self,
        req: EvalRunRequest,
    ) -> tuple[str, list[EvalQuestionInput], list[int]]:
        if not self._retriever.is_ready():
            raise FileNotFoundError("Knowledge base is not indexed yet")

        ks = _normalize_ks(req.ks)

        if req.questions is not None:
            questions: list[EvalQuestionInput] = []
            for q in req.questions:
                rel_ids = _dedup_keep_order(q.relevant_chunk_ids)
                if not rel_ids:
                    raise ValueError(f"Question '{q.id}' has empty relevant_chunk_ids")
                questions.append(q.model_copy(update={"relevant_chunk_ids": rel_ids}))
            dataset_name = "inline"
        else:
            path, questions = self.load_dataset(req.dataset_path)
            dataset_name = path.name

        if req.categories:
            allowed = {c.strip() for c in req.categories if c.strip()}
            questions = [q for q in questions if q.category in allowed]
        if req.query_types:
            allowed = {t.strip() for t in req.query_types if t.strip()}
            questions = [q for q in questions if q.query_type in allowed]
        if req.difficulties:
            allowed = {d.strip() for d in req.difficulties if d.strip()}
            questions = [q for q in questions if q.difficulty in allowed]
        if req.limit_questions is not None:
            questions = questions[: req.limit_questions]

        if not questions:
            raise ValueError("No evaluation questions left after filtering")

        return dataset_name, questions, ks

    async def _execute_eval(
        self,
        *,
        run_id: int,
        questions: list[EvalQuestionInput],
        ks: list[int],
        req: EvalRunRequest,
        gw: dict[str, Any],
    ) -> dict[str, Any]:
        search_limit = req.search_limit or int(gw["rag_search_limit"])
        min_score = (
            req.min_score_override
            if req.min_score_override is not None
            else float(gw["rag_min_score"])
        )
        rel_ratio = (
            req.rel_ratio_override
            if req.rel_ratio_override is not None
            else float(gw["rag_rel_ratio"])
        )
        max_sources = req.max_sources_override or int(gw["rag_max_sources"])
        docs_base_url = str(gw["docs_base_url"])
        max_page_chars = int(gw["rag_max_page_chars"])
        max_total_context_chars = int(gw["rag_max_total_context_chars"])
        embedding_model = str(gw["embedding_model"])
        batch_size = int(gw["index_batch_size"])

        existing_chunk_ids = await self._retriever.list_chunk_ids()

        prepared_queries: list[tuple[EvalQuestionInput, str, str, list[str]]] = []
        skipped_questions = 0
        for q in questions:
            missing_ids = [cid for cid in q.relevant_chunk_ids if cid not in existing_chunk_ids]
            if req.skip_missing_chunks and len(missing_ids) == len(q.relevant_chunk_ids):
                skipped_questions += 1
                continue
            if req.use_query_expansion:
                emb_q, fts_q = expand_query(q.query)
            else:
                emb_q, fts_q = q.query, sanitize_fts_query(q.query)
            prepared_queries.append((q, emb_q, fts_q, missing_ids))

        if not prepared_queries:
            raise ValueError("All questions were skipped because their chunk_ids are missing in index")

        embeddings_cache: dict[str, list[float]] = {}
        cached_hits = 0
        fetched_embeddings = 0
        if req.search_mode != "fts":
            embeddings_cache, cached_hits, fetched_embeddings = await self._resolve_query_embeddings(
                embedding_model=embedding_model,
                query_texts=[emb_q for _, emb_q, _, _ in prepared_queries],
                batch_size=batch_size,
                force_reembed=req.force_reembed,
            )

        records: list[dict[str, Any]] = []
        questions_with_missing = 0

        for q, emb_q, fts_q, missing_ids in prepared_queries:
            if missing_ids:
                questions_with_missing += 1
            query_vector = (
                embeddings_cache.get(_hash_query(embedding_model, emb_q))
                if req.search_mode != "fts"
                else None
            )

            res = await self._retriever.retrieve_for_eval(
                raw_query=q.query,
                query_vector=query_vector,
                fts_query=fts_q,
                search_mode=req.search_mode,
                limit=search_limit,
                docs_base_url=docs_base_url,
                min_score=min_score,
                rel_ratio=rel_ratio,
                max_sources=max_sources,
                max_page_chars=max_page_chars,
                max_total_context_chars=max_total_context_chars,
                include_context_metrics=req.include_context_metrics,
            )

            relevant_set = set(q.relevant_chunk_ids)
            retrieved = res.retrieved_chunk_ids

            first_rank: int | None = None
            for idx, cid in enumerate(retrieved, start=1):
                if cid in relevant_set:
                    first_rank = idx
                    break

            rr = (1.0 / first_rank) if first_rank is not None else 0.0
            mrr_at_k = {
                str(k): round((1.0 / first_rank) if (first_rank is not None and first_rank <= k) else 0.0, 4)
                for k in ks
            }
            recall_at_k = {
                str(k): round(len(set(retrieved[:k]) & relevant_set) / len(relevant_set), 4)
                for k in ks
            }
            hit_at_k = {
                str(k): 1.0 if (set(retrieved[:k]) & relevant_set) else 0.0
                for k in ks
            }

            if res.context_chunk_ids is not None:
                context_recall: float | None = round(
                    len(res.context_chunk_ids & relevant_set) / len(relevant_set), 4
                )
                context_hit: float | None = (
                    1.0 if (res.context_chunk_ids & relevant_set) else 0.0
                )
                no_docs: bool | None = res.selected_sources_count == 0
            else:
                context_recall = None
                context_hit = None
                no_docs = None

            records.append(
                {
                    "id": q.id,
                    "query": q.query,
                    "category": q.category,
                    "query_type": q.query_type,
                    "difficulty": q.difficulty,
                    "relevant_chunk_ids": q.relevant_chunk_ids,
                    "retrieved_chunk_ids": retrieved[: max(ks)],
                    "first_relevant_rank": first_rank,
                    "reciprocal_rank": round(rr, 4),
                    "mrr_at_k": mrr_at_k,
                    "recall_at_k": recall_at_k,
                    "hit_at_k": hit_at_k,
                    "context_recall": context_recall,
                    "context_hit": context_hit,
                    "no_docs": no_docs,
                    "selected_sources_count": res.selected_sources_count,
                    "top_score": round(res.top_score, 4) if res.top_score is not None else None,
                    "missing_chunk_ids": missing_ids,
                    "latency_ms": round(res.latency_ms, 2),
                }
            )

        overall = _aggregate_metrics(records, ks)
        by_category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_type_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_diff_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_category_groups[rec["category"]].append(rec)
            by_type_groups[rec["query_type"]].append(rec)
            by_diff_groups[rec["difficulty"]].append(rec)

        metrics_payload = {
            "overall": overall.model_dump(),
            "by_category": {
                k: _aggregate_metrics(v, ks).model_dump()
                for k, v in sorted(by_category_groups.items())
            },
            "by_query_type": {
                k: _aggregate_metrics(v, ks).model_dump()
                for k, v in sorted(by_type_groups.items())
            },
            "by_difficulty": {
                k: _aggregate_metrics(v, ks).model_dump()
                for k, v in sorted(by_diff_groups.items())
            },
            "questions_with_missing_chunks": questions_with_missing,
            "embeddings_cached_hits": cached_hits,
            "embeddings_fetched": fetched_embeddings,
        }

        await self._db.finish_eval_run(
            run_id,
            status="completed",
            evaluated_questions=len(records),
            skipped_questions=skipped_questions,
            metrics=metrics_payload,
            details=records,
        )

        worst_queries = sorted(
            records,
            key=lambda r: (r["reciprocal_rank"], r["recall_at_k"][str(ks[-1])]),
        )[:20]

        logger.info(
            f"Eval run #{run_id} finished: mode={req.search_mode} "
            f"n={len(records)} MRR={overall.mrr:.4f} Recall@{ks[-1]}={overall.recall_at_k[str(ks[-1])]:.4f}"
        )

        response: dict[str, Any] = {
            "status": "completed",
            "run_id": run_id,
            "total_questions": len(questions),
            "evaluated_questions": len(records),
            "skipped_questions": skipped_questions,
            "metrics": metrics_payload,
            "worst_queries": worst_queries,
        }
        if req.include_details:
            response["details"] = records
        return response

    def _build_config_snapshot(
        self,
        req: EvalRunRequest,
        ks: list[int],
        gw: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ks": ks,
            "search_mode": req.search_mode,
            "search_limit": req.search_limit or int(gw["rag_search_limit"]),
            "use_query_expansion": req.use_query_expansion,
            "include_context_metrics": req.include_context_metrics,
            "embedding_model": str(gw["embedding_model"]),
            "min_score": (
                req.min_score_override
                if req.min_score_override is not None
                else float(gw["rag_min_score"])
            ),
            "rel_ratio": (
                req.rel_ratio_override
                if req.rel_ratio_override is not None
                else float(gw["rag_rel_ratio"])
            ),
            "max_sources": req.max_sources_override or int(gw["rag_max_sources"]),
            "skip_missing_chunks": req.skip_missing_chunks,
        }

    async def run_sync(
        self,
        req: EvalRunRequest,
        *,
        trigger: str = "manual",
    ) -> dict[str, Any]:
        if self._is_running:
            raise RuntimeError("Evaluation run is already running")

        self._is_running = True
        run_id: int | None = None
        try:
            dataset_name, questions, ks = self._prepare_run_inputs(req)
            gw = await self._db.get_gateway_settings()
            config_snapshot = self._build_config_snapshot(req, ks, gw)
            run_id = await self._db.create_eval_run(
                trigger=trigger,
                dataset_name=dataset_name,
                total_questions=len(questions),
                config=config_snapshot,
            )
            return await self._execute_eval(
                run_id=run_id,
                questions=questions,
                ks=ks,
                req=req,
                gw=gw,
            )
        except Exception as exc:
            if run_id is not None:
                logger.exception(f"Eval run #{run_id} failed")
                await self._db.finish_eval_run(run_id, status="failed", error=str(exc))
            raise
        finally:
            self._is_running = False

    async def start_background(
        self,
        req: EvalRunRequest,
        *,
        trigger: str = "manual",
    ) -> int:
        if self._is_running:
            raise RuntimeError("Evaluation run is already running")

        self._is_running = True
        try:
            dataset_name, questions, ks = self._prepare_run_inputs(req)
            gw = await self._db.get_gateway_settings()
            config_snapshot = self._build_config_snapshot(req, ks, gw)
            run_id = await self._db.create_eval_run(
                trigger=trigger,
                dataset_name=dataset_name,
                total_questions=len(questions),
                config=config_snapshot,
            )
        except Exception:
            self._is_running = False
            raise

        async def _runner() -> None:
            try:
                await self._execute_eval(
                    run_id=run_id,
                    questions=questions,
                    ks=ks,
                    req=req,
                    gw=gw,
                )
            except Exception as exc:
                logger.exception(f"Background eval run #{run_id} failed")
                await self._db.finish_eval_run(run_id, status="failed", error=str(exc))
            finally:
                self._is_running = False

        asyncio.create_task(_runner())
        return run_id