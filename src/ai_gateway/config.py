from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    host: str
    port: int
    gateway_public_url: str
    website_base_url: str
    telegram_bot_url: str
    cors_allowed_origins: tuple[str, ...]

    admin_username: str
    admin_password: str
    admin_api_key: str | None
    jwt_secret: str
    jwt_ttl_seconds: int

    polza_api_key: str
    polza_base_url: str
    polza_api_root: str

    embedding_model: str
    website_chat_model: str
    website_guard_model: str
    telegram_chat_model: str
    telegram_guard_model: str
    external_chat_model: str
    external_guard_model: str

    guard_enabled: bool
    guard_offtopic_threshold: float
    guard_noul_threshold: float
    guard_timeout_seconds: float
    guard_debug: bool

    rag_min_score: float
    rag_rel_ratio: float
    rag_max_sources: int
    rag_search_limit: int
    rag_max_page_chars: int
    rag_max_total_context_chars: int

    data_dir: Path
    sqlite_path: Path
    db_path: Path
    assets_dir: Path
    docs_cache_dir: Path
    embeddings_cache_path: Path

    docs_base_url: str
    docs_repo_slug: str
    docs_repo_branch: str
    docs_repo_docs_subdir: str
    docs_repo_assets_subdir: str
    chunk_max_chars: int
    chunk_overlap: int
    index_batch_size: int
    sync_docs_on_start: bool

    rate_limit_enabled: bool
    rate_limit_requests: int
    rate_limit_window_seconds: int

    log_level: str
    log_file: str


def load_settings() -> Settings:
    load_dotenv()

    polza_api_key = os.getenv("POLZA_AI_API_KEY", "").strip()
    if not polza_api_key:
        raise ValueError("POLZA_AI_API_KEY is required")

    admin_password = os.getenv("ADMIN_PASSWORD", "").strip()
    if not admin_password:
        raise ValueError("ADMIN_PASSWORD is required")

    jwt_secret = os.getenv("JWT_SECRET_KEY", "").strip()
    if not jwt_secret:
        raise ValueError("JWT_SECRET_KEY is required")

    polza_base_url = os.getenv("POLZA_BASE_URL", "https://polza.ai/api/v1").rstrip("/")
    default_api_root = polza_base_url.removesuffix("/v1")
    polza_api_root = os.getenv("POLZA_API_ROOT", default_api_root).rstrip("/")

    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    sqlite_path = Path(os.getenv("SQLITE_DB_PATH", str(data_dir / "gateway.db"))).resolve()
    db_path = Path(os.getenv("DOCS_DB_PATH", str(data_dir / "docs_vector_db"))).resolve()
    assets_dir = Path(os.getenv("COMPILE_ASSETS_DIR", str(data_dir / "compile_assets"))).resolve()
    docs_cache_dir = Path(os.getenv("DOCS_CACHE_DIR", str(data_dir / "docs_content"))).resolve()
    embeddings_cache_path = Path(
        os.getenv("EMBEDDINGS_CACHE_PATH", str(data_dir / "embeddings_cache.jsonl"))
    ).resolve()

    default_chat = os.getenv("RAG_CHAT_MODEL", "deepseek/deepseek-v4.1-flash-1017").strip()
    default_guard = os.getenv("RAG_GUARD_MODEL", "google/gemini-decision-1").strip()

    admin_api_key = os.getenv("ADMIN_API_KEY", "").strip() or None

    return Settings(
        host=os.getenv("GATEWAY_HOST", "0.0.0.0").strip(),
        port=int(os.getenv("GATEWAY_PORT", "8085")),
        gateway_public_url=os.getenv("GATEWAY_PUBLIC_URL", "http://localhost:8085").rstrip("/"),
        website_base_url=os.getenv("WEBSITE_BASE_URL", "https://typst-gost.ru").rstrip("/"),
        telegram_bot_url=os.getenv("TELEGRAM_BOT_URL", "http://localhost:8081").rstrip("/"),
        cors_allowed_origins=_parse_csv(os.getenv("CORS_ALLOWED_ORIGINS")),
        admin_username=os.getenv("ADMIN_USERNAME", "admin").strip(),
        admin_password=admin_password,
        admin_api_key=admin_api_key,
        jwt_secret=jwt_secret,
        jwt_ttl_seconds=int(os.getenv("JWT_TTL_SECONDS", "86400")),
        polza_api_key=polza_api_key,
        polza_base_url=polza_base_url,
        polza_api_root=polza_api_root,
        embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "qwen/qwen3-embedding-8b").strip(),
        website_chat_model=os.getenv("WEBSITE_CHAT_MODEL", default_chat).strip(),
        website_guard_model=os.getenv("WEBSITE_GUARD_MODEL", default_guard).strip(),
        telegram_chat_model=os.getenv("TELEGRAM_CHAT_MODEL", default_chat).strip(),
        telegram_guard_model=os.getenv("TELEGRAM_GUARD_MODEL", default_guard).strip(),
        external_chat_model=os.getenv("EXTERNAL_CHAT_MODEL", default_chat).strip(),
        external_guard_model=os.getenv("EXTERNAL_GUARD_MODEL", default_guard).strip(),
        guard_enabled=_parse_bool(os.getenv("RAG_GUARD_ENABLED"), default=True),
        guard_offtopic_threshold=float(os.getenv("RAG_GUARD_OFFTOPIC_THRESHOLD", "0.85")),
        guard_noul_threshold=float(os.getenv("RAG_GUARD_NOUL_THRESHOLD", "0.80")),
        guard_timeout_seconds=float(os.getenv("RAG_GUARD_TIMEOUT", "3.0")),
        guard_debug=_parse_bool(os.getenv("RAG_GUARD_DEBUG"), default=False),
        rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.40")),
        rag_rel_ratio=float(os.getenv("RAG_REL_RATIO", "0.72")),
        rag_max_sources=int(os.getenv("RAG_MAX_SOURCES", "5")),
        rag_search_limit=int(os.getenv("RAG_SEARCH_LIMIT", "24")),
        rag_max_page_chars=int(os.getenv("RAG_MAX_PAGE_CHARS", "10000")),
        rag_max_total_context_chars=int(os.getenv("RAG_MAX_TOTAL_CONTEXT_CHARS", "24000")),
        data_dir=data_dir,
        sqlite_path=sqlite_path,
        db_path=db_path,
        assets_dir=assets_dir,
        docs_cache_dir=docs_cache_dir,
        embeddings_cache_path=embeddings_cache_path,
        docs_base_url=os.getenv("DOCS_BASE_URL", "https://typst-gost.ru").rstrip("/"),
        docs_repo_slug=os.getenv("DOCS_REPO_SLUG", "typst-gost/website").strip(),
        docs_repo_branch=os.getenv("DOCS_REPO_BRANCH", "main").strip(),
        docs_repo_docs_subdir=os.getenv("DOCS_REPO_DOCS_SUBDIR", "content/docs").strip("/"),
        docs_repo_assets_subdir=os.getenv(
            "DOCS_REPO_ASSETS_SUBDIR",
            "public/docs/compile_assets",
        ).strip("/"),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "2500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "250")),
        index_batch_size=int(os.getenv("INDEX_BATCH_SIZE", "16")),
        sync_docs_on_start=_parse_bool(os.getenv("SYNC_DOCS_ON_START"), default=True),
        rate_limit_enabled=_parse_bool(os.getenv("RATE_LIMIT_ENABLED"), default=True),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "20")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_file=os.getenv("LOG_FILE", "").strip(),
    )