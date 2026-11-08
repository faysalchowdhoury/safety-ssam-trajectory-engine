"""Response cache: Redis when configured, otherwise an in-process TTL LRU."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

from ..core.logging import get_logger

log = get_logger(__name__)


class ResponseCache(ABC):
    backend: str = "abstract"

    @abstractmethod
    async def get(self, key: str) -> str | None: ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl_s: int) -> None: ...

    @abstractmethod
    async def delete_prefix(self, prefix: str) -> None: ...

    async def close(self) -> None:  # pragma: no cover - trivial
        return None


class InMemoryCache(ResponseCache):
    """Bounded LRU with per-entry expiry. Safe within one event loop."""

    backend = "memory"

    def __init__(self, max_entries: int = 10_000) -> None:
        self.max_entries = max_entries
        self._data: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            expires, value = item
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        async with self._lock:
            self._data[key] = (time.monotonic() + ttl_s, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    async def delete_prefix(self, prefix: str) -> None:
        async with self._lock:
            for k in [k for k in self._data if k.startswith(prefix)]:
                del self._data[k]


class RedisCache(ResponseCache):
    """``redis.asyncio`` client; failures degrade to cache misses rather than errors."""

    backend = "redis"

    def __init__(self, url: str) -> None:
        import redis.asyncio as redis

        self._client: Any = redis.from_url(url, decode_responses=True, socket_timeout=0.05)

    async def get(self, key: str) -> str | None:
        try:
            return await self._client.get(key)
        except Exception as exc:  # noqa: BLE001 - cache must never fail a request
            log.warning("cache.get_failed", extra={"error": repr(exc)})
            return None

    async def set(self, key: str, value: str, ttl_s: int) -> None:
        try:
            await self._client.set(key, value, ex=ttl_s)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache.set_failed", extra={"error": repr(exc)})

    async def delete_prefix(self, prefix: str) -> None:
        try:
            async for key in self._client.scan_iter(match=f"{prefix}*", count=500):
                await self._client.delete(key)
        except Exception as exc:  # noqa: BLE001
            log.warning("cache.delete_failed", extra={"error": repr(exc)})

    async def close(self) -> None:
        await self._client.aclose()


def build_cache(redis_url: str | None, max_entries: int) -> ResponseCache:
    if redis_url:
        return RedisCache(redis_url)
    return InMemoryCache(max_entries)
