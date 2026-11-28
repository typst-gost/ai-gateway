from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from loguru import logger
from openai import AsyncOpenAI

from ai_gateway.config import Settings

MODEL_TYPE_ALIASES: dict[str, str | None] = {
    "all": None,
    "llm": "chat",
    "decision": "classification",
    "chat": "chat",
    "classification": "classification",
    "image": "image",
    "embedding": "embedding",
    "audio": "audio",
    "video": "video",
    "tts": "tts",
    "stt": "stt",
}


class PolzaClient:
    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.polza_base_url
        self._api_root = settings.polza_api_root
        self._api_key = settings.polza_api_key
        self._openai = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        self._http = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        self._models_cache: dict[tuple[str | None, bool, bool], tuple[float, list[dict[str, Any]]]] = {}
        self._models_cache_ttl = 60.0

    @property
    def openai(self) -> AsyncOpenAI:
        return self._openai

    async def close(self) -> None:
        await self._http.aclose()
        await self._openai.close()

    async def create_embedding(self, model: str, text: str) -> list[float]:
        resp = await self._openai.embeddings.create(model=model, input=[text])
        return resp.data[0].embedding

    async def create_embeddings(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._openai.embeddings.create(model=model, input=texts)
        vectors: list[list[float] | None] = [None] * len(texts)
        for item in resp.data:
            vectors[item.index] = item.embedding
        if any(v is None for v in vectors):
            raise RuntimeError("Incomplete embeddings response from Polza AI")
        return [v for v in vectors if v is not None]

    async def create_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        completion = await self._openai.chat.completions.create(**kwargs)
        content = (completion.choices[0].message.content or "").strip()
        usage = completion.usage.model_dump() if completion.usage else None
        return content, usage

    async def stream_chat_completion(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[tuple[str, Any]]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        stream = await self._openai.chat.completions.create(**kwargs)
        last_usage: dict[str, Any] | None = None
        try:
            async for chunk in stream:
                if getattr(chunk, "usage", None) is not None:
                    last_usage = chunk.usage.model_dump()
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield ("delta", delta)
        finally:
            await stream.close()

        if last_usage is not None:
            yield ("usage", last_usage)

    async def evaluate_decision(
        self,
        *,
        model: str,
        state: str,
        questions: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        payload = {
            "model": model,
            "state": state,
            "questions": questions,
        }
        resp = await self._http.post(
            f"{self._base_url}/systemone",
            json=payload,
            timeout=timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    async def get_balance(self) -> dict[str, Any]:
        resp = await self._http.get(f"{self._api_root}/v2/balance")
        resp.raise_for_status()
        data = resp.json()
        return {
            "amount": data.get("amount"),
            "available": data.get("available"),
            "currency": data.get("currency", "RUB"),
            "raw": data,
        }

    async def list_models(
        self,
        *,
        category: str = "all",
        include_providers: bool = False,
        include_capabilities: bool = False,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        normalized_cat = category.strip().lower()
        if normalized_cat not in MODEL_TYPE_ALIASES:
            raise ValueError(f"Unsupported model category: {category}")

        polza_type = MODEL_TYPE_ALIASES[normalized_cat]
        cache_key = (polza_type, include_providers, include_capabilities)
        now = time.monotonic()

        if not refresh and cache_key in self._models_cache:
            cached_at, cached_items = self._models_cache[cache_key]
            if now - cached_at <= self._models_cache_ttl:
                return cached_items

        params: dict[str, str] = {}
        if polza_type:
            params["type"] = polza_type
        if include_providers:
            params["include_providers"] = "true"
        if include_capabilities:
            params["include_capabilities"] = "true"

        resp = await self._http.get(f"{self._base_url}/models", params=params)
        resp.raise_for_status()
        payload = resp.json()
        raw_items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(raw_items, list):
            logger.warning(f"Unexpected models payload from Polza AI: {type(payload)}")
            raw_items = []

        self._models_cache[cache_key] = (now, raw_items)
        return raw_items