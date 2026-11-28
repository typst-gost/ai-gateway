from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ChannelName = Literal[
    "website",
    "telegram_private",
    "telegram_group",
    "telegram_fix",
    "external",
]
ChatChannelName = Literal[
    "website",
    "telegram_private",
    "telegram_group",
    "external",
]
ModelCategory = Literal[
    "all",
    "llm",
    "decision",
    "chat",
    "classification",
    "image",
    "embedding",
    "audio",
    "video",
    "tts",
    "stt",
]
EvalSearchMode = Literal["hybrid_scored", "hybrid_raw", "vector", "fts"]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class PageContext(BaseModel):
    title: str | None = None
    location: str | None = None


class DocSourceSchema(BaseModel):
    label: int
    title: str
    url: str
    path: str
    doc_url: str
    category: str
    score: float = 0.0
    cosine: float = 0.0
    content: str | None = None


class RetrievalStatsSchema(BaseModel):
    candidates_count: int
    selected_count: int
    top_score: float
    top_cosine: float
    cutoff_score: float
    min_score: float
    rel_ratio: float
    cited_count: int | None = None


class GuardVerdictSchema(BaseModel):
    allowed: bool
    reason: str
    model: str
    intent: str
    probabilities: dict[str, float] = Field(default_factory=dict)
    offtopic_prob: float = 0.0
    abuse_prob: float = 0.0
    offtopic_threshold: float = 0.85
    noul_threshold: float = 0.80
    latency_ms: float = 0.0
    error: str | None = None


class AssistantChatRequest(BaseModel):
    channel: ChatChannelName = "external"
    conversation_id: str | None = None
    query: str | None = None
    messages: list[ChatMessageInput] | None = None
    page_context: PageContext | None = None
    available_assets: list[str] | None = None
    external_user_id: str | None = None
    external_chat_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    stream: bool = True
    save_history: bool = True


class AssistantChatResponse(BaseModel):
    conversation_id: str
    channel: str
    answer: str
    raw_answer: str
    sources: list[DocSourceSchema]
    guard: GuardVerdictSchema
    stats: RetrievalStatsSchema | None = None
    blocked_by_guard: bool = False
    no_docs_found: bool = False
    llm_model: str
    decision_model: str
    usage: dict[str, Any] | None = None
    latency_ms: float
    debug_markdown: str | None = None


class AssistantFixRequest(BaseModel):
    source_code: str = Field(min_length=1, max_length=100000)
    error_text: str = Field(min_length=1, max_length=20000)
    conversation_id: str | None = None
    available_assets: list[str] | None = None
    external_user_id: str | None = None
    external_chat_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    save_history: bool = True


class AssistantFixResponse(BaseModel):
    conversation_id: str
    fixed_code: str
    blocked_by_guard: bool = False
    refusal_message: str | None = None
    guard: GuardVerdictSchema
    stats: RetrievalStatsSchema | None = None
    llm_model: str
    decision_model: str
    usage: dict[str, Any] | None = None
    latency_ms: float
    debug_markdown: str | None = None


class AssistantSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1500)
    history: list[ChatMessageInput] | None = None
    page_context: PageContext | None = None
    max_sources: int | None = Field(default=None, ge=1, le=20)
    include_content: bool = False


class AssistantSearchResponse(BaseModel):
    sources: list[DocSourceSchema]
    stats: RetrievalStatsSchema


class ClearConversationRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=256)


class AdminLoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


class AdminTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str


class GatewaySettingsUpdate(BaseModel):
    embedding_model: str | None = None
    rag_min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    rag_rel_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    rag_max_sources: int | None = Field(default=None, ge=1, le=20)
    rag_search_limit: int | None = Field(default=None, ge=5, le=100)
    rag_max_page_chars: int | None = Field(default=None, ge=1000, le=50000)
    rag_max_total_context_chars: int | None = Field(default=None, ge=2000, le=100000)
    docs_base_url: str | None = None
    docs_repo_slug: str | None = None
    docs_repo_branch: str | None = None
    docs_repo_docs_subdir: str | None = None
    docs_repo_assets_subdir: str | None = None
    chunk_max_chars: int | None = Field(default=None, ge=500, le=8000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=2000)
    index_batch_size: int | None = Field(default=None, ge=1, le=64)
    rate_limit_enabled: bool | None = None
    rate_limit_requests: int | None = Field(default=None, ge=1, le=1000)
    rate_limit_window_seconds: int | None = Field(default=None, ge=1, le=3600)
    website_base_url: str | None = None
    telegram_bot_url: str | None = None
    gateway_public_url: str | None = None


class ProfileConfigUpdate(BaseModel):
    enabled: bool | None = None
    llm_model: str | None = None
    decision_model: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=64, le=32768)
    system_prompt: str | None = None
    decision_questions: dict[str, Any] | None = None
    refusal_reply: str | None = None
    no_docs_reply: str | None = None
    guard_enabled: bool | None = None
    guard_offtopic_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    guard_noul_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    guard_timeout_seconds: float | None = Field(default=None, ge=0.5, le=30.0)
    guard_debug: bool | None = None
    max_query_chars: int | None = Field(default=None, ge=50, le=50000)
    use_history: bool | None = None
    history_max_messages: int | None = Field(default=None, ge=0, le=50)
    history_ttl_seconds: int | None = Field(default=None, ge=60, le=604800)
    include_assets_in_prompt: bool | None = None
    source_url_mode: Literal["relative", "absolute"] | None = None


class TelegramBulkUpdate(BaseModel):
    llm_model: str | None = None
    decision_model: str | None = None
    guard_enabled: bool | None = None
    guard_offtopic_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    guard_noul_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    guard_timeout_seconds: float | None = Field(default=None, ge=0.5, le=30.0)
    guard_debug: bool | None = None


class TestGuardRequest(BaseModel):
    profile_id: ChannelName = "website"
    query: str = Field(min_length=1, max_length=4000)
    history: list[ChatMessageInput] | None = None
    page_context: PageContext | None = None
    decision_model_override: str | None = None
    decision_questions_override: dict[str, Any] | None = None
    offtopic_threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)
    noul_threshold_override: float | None = Field(default=None, ge=0.0, le=1.0)


class ReindexTriggerRequest(BaseModel):
    sync_github: bool = True
    force_reembed: bool = False
    background: bool = True


class EvalQuestionInput(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    query: str = Field(min_length=1, max_length=4000)
    relevant_chunk_ids: list[str] = Field(min_length=1)
    category: str = "unknown"
    query_type: str = "unknown"
    difficulty: str = "unknown"


class EvalDatasetUploadRequest(BaseModel):
    questions: list[EvalQuestionInput] = Field(min_length=1)


class EvalRunRequest(BaseModel):
    dataset_path: str | None = None
    questions: list[EvalQuestionInput] | None = None
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    search_mode: EvalSearchMode = "hybrid_scored"
    search_limit: int | None = Field(default=None, ge=1, le=200)
    use_query_expansion: bool = True
    include_context_metrics: bool = True
    min_score_override: float | None = Field(default=None, ge=0.0, le=1.0)
    rel_ratio_override: float | None = Field(default=None, ge=0.0, le=1.0)
    max_sources_override: int | None = Field(default=None, ge=1, le=20)
    categories: list[str] | None = None
    query_types: list[str] | None = None
    difficulties: list[str] | None = None
    limit_questions: int | None = Field(default=None, ge=1, le=5000)
    skip_missing_chunks: bool = False
    force_reembed: bool = False
    background: bool = False
    include_details: bool = False


class EvalChunkLookupRequest(BaseModel):
    chunk_ids: list[str] = Field(min_length=1, max_length=200)


class EvalMetricsBlock(BaseModel):
    count: int
    mrr: float
    mrr_at_k: dict[str, float]
    recall_at_k: dict[str, float]
    hit_rate_at_k: dict[str, float]
    context_recall: float | None = None
    context_hit_rate: float | None = None
    no_docs_rate: float | None = None
    avg_selected_sources: float | None = None
    avg_latency_ms: float