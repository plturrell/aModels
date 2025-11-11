from __future__ import annotations

from functools import lru_cache
from typing import Optional

from ..config import get_settings

try:
    import redis.asyncio as redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]


class RedisNotAvailable(RuntimeError):
    """Raised when Redis is requested but the driver or configuration is missing."""


@lru_cache()
def get_redis_client() -> Optional["redis.Redis"]:
    """
    Return an asyncio Redis client when enabled, otherwise None.
    """
    settings = get_settings()
    if not settings.redis_enabled or redis is None:
        return None

    return redis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
    )
