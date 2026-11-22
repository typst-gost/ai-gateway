from __future__ import annotations

import argparse
import asyncio

from loguru import logger

from ai_gateway.config import load_settings
from ai_gateway.db.database import GatewayDatabase
from ai_gateway.logging_setup import setup_logging
from ai_gateway.rag.indexer import DocsIndexer
from ai_gateway.rag.retriever import DocsRetriever


async def _run_reindex(*, sync_github: bool, force_reembed: bool) -> None:
    settings = load_settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)

    db = GatewayDatabase(settings.sqlite_path, settings)
    await db.connect()
    retriever = DocsRetriever(settings.db_path)
    indexer = DocsIndexer(settings, db, retriever)

    try:
        job_id, result = await indexer.reindex(
            trigger="cli",
            sync_github=sync_github,
            force_reembed=force_reembed,
        )
        logger.info(
            f"CLI reindex finished (job_id={job_id}): "
            f"files={result.files_processed}, chunks={result.chunks_count}, "
            f"embeddings={result.embeddings_fetched}, assets={result.assets_synced}"
        )
    finally:
        await db.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync documentation from GitHub and rebuild LanceDB index",
    )
    parser.add_argument(
        "--no-sync-github",
        action="store_true",
        help="Reindex from already downloaded docs_content directory",
    )
    parser.add_argument(
        "--force-reembed",
        action="store_true",
        help="Ignore embeddings cache and regenerate all vectors",
    )
    args = parser.parse_args()

    asyncio.run(
        _run_reindex(
            sync_github=not args.no_sync_github,
            force_reembed=args.force_reembed,
        )
    )


if __name__ == "__main__":
    main()