from __future__ import annotations

import argparse
import asyncio
import json

from ai_gateway.config import load_settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.logging_setup import setup_logging
from ai_gateway.polza.client import PolzaClient
from ai_gateway.rag.eval_service import EvalService
from ai_gateway.rag.retriever import DocsRetriever
from ai_gateway.schemas import EvalRunRequest


async def _run(args: argparse.Namespace) -> None:
    settings = load_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    db = GatewayDatabase(settings.sqlite_path, settings)
    await db.connect()
    polza = PolzaClient(settings)
    retriever = DocsRetriever(settings.db_path)
    eval_service = EvalService(settings, db, polza, retriever)

    try:
        ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        req = EvalRunRequest(
            dataset_path=args.dataset,
            ks=ks,
            search_mode=args.mode,
            search_limit=args.limit,
            use_query_expansion=not args.no_expand,
            skip_missing_chunks=args.skip_missing,
            force_reembed=args.force_reembed,
            include_context_metrics=not args.no_context_metrics,
        )
        result = await eval_service.run_sync(req, trigger="cli")
        summary = {
            "run_id": result["run_id"],
            "evaluated_questions": result["evaluated_questions"],
            "skipped_questions": result["skipped_questions"],
            "overall": result["metrics"]["overall"],
            "by_category": result["metrics"]["by_category"],
            "by_query_type": result["metrics"]["by_query_type"],
            "by_difficulty": result["metrics"]["by_difficulty"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        await polza.close()
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG Recall@k, HitRate@k, and MRR evaluation on golden benchmark",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to dataset JSON (default: EVAL_DATASET_PATH or data/eval_dataset.json)",
    )
    parser.add_argument(
        "--ks",
        default="1,3,5,10,20",
        help="Comma-separated K values for Recall@k and HitRate@k (default: 1,3,5,10,20)",
    )
    parser.add_argument(
        "--mode",
        default="hybrid_scored",
        choices=["hybrid_scored", "hybrid_raw", "vector", "fts"],
        help="Search mode to evaluate (default: hybrid_scored)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="LanceDB candidates search limit",
    )
    parser.add_argument(
        "--no-expand",
        action="store_true",
        help="Disable SYNONYM_MAP query expansion",
    )
    parser.add_argument(
        "--no-context-metrics",
        action="store_true",
        help="Do not calculate final context inclusion metrics (faster)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip questions whose relevant chunk IDs are entirely absent from index",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Recompute query embeddings instead of using cache",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()