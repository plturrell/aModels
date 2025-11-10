from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

from ..config import get_settings

try:
    import redis.asyncio as redis
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - optional dependency
    redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]


logger = logging.getLogger(__name__)


class RedisNotAvailable(RuntimeError):
    """Raised when Redis is requested but the driver or configuration is missing."""


@lru_cache()
def get_redis_client() -> Optional["redis.Redis"]:
    """
    Return an asyncio Redis client when enabled, otherwise None.
    """
    settings = get_settings()
    if not settings.redis_enabled:
        logger.info("Redis cache disabled via configuration")
        return None

    if redis is None:
        logger.warning("Redis driver not installed; falling back to in-process cache only")
        return None

    try:
        client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        logger.debug("Connected to Redis cache", extra={"redis_url": settings.redis_url})
        return client
    except RedisError as exc:  # pragma: no cover - connection failures are environment-specific
        logger.warning(
            "Failed to connect to Redis cache; continuing without distributed cache",
            extra={"redis_url": settings.redis_url},
            exc_info=exc,
        )
        return None
