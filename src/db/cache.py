"""
Redis-backed caching helpers for parsed posts.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime
from typing import Any

from redis.asyncio import Redis

from ..aggregator.parsers.base import ParsedPost


class PostsCache:
    """Simple namespaced cache for storing serialized parsed posts in Redis."""

    def __init__(self, client: Redis, *, namespace: str = "posts") -> None:
        self._client = client
        self._namespace = namespace

    async def get_posts(self, source: str) -> list[ParsedPost] | None:
        """Return cached posts for a source or None if the key is absent."""

        key = self._build_key(source)
        data = await self._client.get(key)
        if data is None:
            return None
        records = json.loads(data)
        return [self._deserialize_post(item) for item in records]

    async def cache_posts(
        self,
        source: str,
        posts: Iterable[ParsedPost],
        *,
        ttl_seconds: int = 900,
    ) -> None:
        """Persist posts for a source with the provided TTL."""

        key = self._build_key(source)
        payload = json.dumps([self._serialize_post(post) for post in posts])
        await self._client.set(key, payload, ex=ttl_seconds)

    async def invalidate_source(self, source: str) -> None:
        """Delete cached posts for the provided source."""

        key = self._build_key(source)
        await self._client.delete(key)

    def _build_key(self, source: str) -> str:
        return f"{self._namespace}:{source}"

    @staticmethod
    def _serialize_post(post: ParsedPost) -> dict[str, Any]:
        data = asdict(post)
        data["published_at"] = post.published_at.isoformat()
        data["fetched_at"] = post.fetched_at.isoformat()
        data["raw_data"] = PostsCache._coerce_raw(data["raw_data"])
        return data

    @staticmethod
    def _deserialize_post(payload: dict[str, Any]) -> ParsedPost:
        payload = payload.copy()
        payload["published_at"] = datetime.fromisoformat(payload["published_at"])
        payload["fetched_at"] = datetime.fromisoformat(payload["fetched_at"])
        return ParsedPost(**payload)

    @staticmethod
    def _coerce_raw(payload: Any) -> Any:
        if isinstance(payload, datetime):
            return payload.isoformat()
        if isinstance(payload, dict):
            return {key: PostsCache._coerce_raw(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [PostsCache._coerce_raw(item) for item in payload]
        return payload


__all__ = ["PostsCache"]
