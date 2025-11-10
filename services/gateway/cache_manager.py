"""
Redis-based caching layer with smart TTL strategies.
Implements cache warming, invalidation patterns, and statistics.
"""
import json
import hashlib
from typing import Any, Dict, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from logging_config import get_logger

logger = get_logger(__name__)


class CacheStrategy(str, Enum):
    """Cache TTL strategies for different data types."""
    NO_CACHE = "no_cache"  # Don't cache
    SHORT = "short"  # 60 seconds - frequently changing data
    MEDIUM = "medium"  # 5 minutes - semi-static data
    LONG = "long"  # 1 hour - relatively static data
    VERY_LONG = "very_long"  # 24 hours - rarely changing data


@dataclass
class CacheConfig:
    """Cache configuration for endpoint."""
    strategy: CacheStrategy
    ttl_seconds: int
    cache_key_prefix: str
    include_params: bool = True
    include_body: bool = False


class CacheManager:
    """
    Intelligent caching layer with TTL strategies and statistics.
    """
    
    # Default TTL values (seconds)
    TTL_MAP = {
        CacheStrategy.NO_CACHE: 0,
        CacheStrategy.SHORT: 60,  # 1 minute
        CacheStrategy.MEDIUM: 300,  # 5 minutes
        CacheStrategy.LONG: 3600,  # 1 hour
        CacheStrategy.VERY_LONG: 86400,  # 24 hours
    }
    
    # Endpoint-specific cache strategies
    ENDPOINT_STRATEGIES = {
        # Frequently changing - short cache
        "/api/*/status/*": CacheStrategy.SHORT,
        "/api/*/history": CacheStrategy.SHORT,
        
        # Semi-static data - medium cache
        "/api/*/results/*": CacheStrategy.MEDIUM,
        "/search/unified": CacheStrategy.MEDIUM,
        "/catalog/semantic-search": CacheStrategy.MEDIUM,
        
        # Static reference data - long cache
        "/catalog/data-elements": CacheStrategy.LONG,
        "/catalog/ontology": CacheStrategy.LONG,
        "/sap-bdc/data-products": CacheStrategy.LONG,
        
        # Health checks - very short cache
        "/healthz": CacheStrategy.SHORT,
        "/**/health*": CacheStrategy.SHORT,
        
        # Expensive operations - cache aggressively
        "/deep-research/research": CacheStrategy.LONG,
        "/catalog/data-products/build": CacheStrategy.LONG,
    }
    
    def __init__(self, redis_client: Optional[Any] = None):
        self.redis = redis_client
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
    def get_strategy_for_endpoint(self, path: str) -> CacheStrategy:
        """
        Determine cache strategy for endpoint path.
        
        Args:
            path: Request path
            
        Returns:
            CacheStrategy enum
        """
        # Check exact matches first
        if path in self.ENDPOINT_STRATEGIES:
            return self.ENDPOINT_STRATEGIES[path]
        
        # Check pattern matches
        for pattern, strategy in self.ENDPOINT_STRATEGIES.items():
            if self._matches_pattern(path, pattern):
                return strategy
        
        # Default to medium caching
        return CacheStrategy.MEDIUM
    
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Simple pattern matching with * wildcards."""
        if "*" not in pattern:
            return path == pattern
        
        parts = pattern.split("*")
        pos = 0
        for part in parts:
            if not part:
                continue
            idx = path.find(part, pos)
            if idx == -1:
                return False
            pos = idx + len(part)
        return True
    
    def get_config_for_endpoint(self, path: str, method: str = "GET") -> CacheConfig:
        """
        Get cache configuration for endpoint.
        
        Args:
            path: Request path
            method: HTTP method
            
        Returns:
            CacheConfig with strategy and TTL
        """
        # POST requests typically shouldn't be cached (mutations)
        # Exception: search/query operations
        if method == "POST" and "/search" not in path and "/query" not in path:
            return CacheConfig(
                strategy=CacheStrategy.NO_CACHE,
                ttl_seconds=0,
                cache_key_prefix=f"gateway:nocache",
                include_body=True
            )
        
        strategy = self.get_strategy_for_endpoint(path)
        ttl = self.TTL_MAP[strategy]
        
        return CacheConfig(
            strategy=strategy,
            ttl_seconds=ttl,
            cache_key_prefix=f"gateway:cache:{method.lower()}",
            include_params=True,
            include_body=(method == "POST")
        )
    
    def make_cache_key(
        self,
        path: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        config: Optional[CacheConfig] = None
    ) -> str:
        """
        Generate cache key from request parameters.
        
        Args:
            path: Request path
            method: HTTP method
            params: Query parameters
            body: Request body
            config: Cache configuration
            
        Returns:
            Cache key string
        """
        if config is None:
            config = self.get_config_for_endpoint(path, method)
        
        key_parts = [config.cache_key_prefix, path]
        
        # Add params if configured
        if config.include_params and params:
            params_str = json.dumps(params, sort_keys=True)
            params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key_parts.append(f"p:{params_hash}")
        
        # Add body if configured
        if config.include_body and body:
            body_str = json.dumps(body, sort_keys=True)
            body_hash = hashlib.md5(body_str.encode()).hexdigest()[:8]
            key_parts.append(f"b:{body_hash}")
        
        return ":".join(key_parts)
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        if not self.redis:
            return default
        
        try:
            cached_str = await self.redis.get(key)
            if cached_str:
                self.stats["hits"] += 1
                logger.debug("Cache hit", key=key)
                return json.loads(cached_str)
            else:
                self.stats["misses"] += 1
                return default
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache get error", key=key, error=str(e))
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        config: Optional[CacheConfig] = None
    ) -> bool:
        """
        Set value in cache with TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (overrides config)
            config: Cache configuration
            
        Returns:
            True if successful
        """
        if not self.redis:
            return False
        
        # Determine TTL
        if ttl is None and config:
            ttl = config.ttl_seconds
        if ttl is None or ttl <= 0:
            return False
        
        try:
            value_str = json.dumps(value)
            await self.redis.set(key, value_str, ex=ttl)
            self.stats["sets"] += 1
            logger.debug("Cache set", key=key, ttl=ttl)
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache set error", key=key, error=str(e))
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if not self.redis:
            return False
        
        try:
            deleted = await self.redis.delete(key)
            if deleted:
                self.stats["deletes"] += 1
                logger.debug("Cache delete", key=key)
            return deleted > 0
        except Exception as e:
            self.stats["errors"] += 1
            logger.warning("Cache delete error", key=key, error=str(e))
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "gateway:cache:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis:
            return 0
        
        try:
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.redis.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.redis.delete(*keys)
                    deleted += len(keys)
                    self.stats["deletes"] += len(keys)
                if cursor == 0:
                    break
            
            logger.info("Cache pattern invalidated", pattern=pattern, deleted=deleted)
            return deleted
        except Exception as e:
            self.stats["errors"] += 1
            logger.error("Cache invalidation error", pattern=pattern, error=str(e))
            return 0
    
    async def invalidate_endpoint(self, path: str):
        """Invalidate all cache entries for an endpoint."""
        pattern = f"gateway:cache:*:{path}*"
        return await self.invalidate_pattern(pattern)
    
    async def clear_all(self) -> bool:
        """Clear all gateway cache entries."""
        try:
            deleted = await self.invalidate_pattern("gateway:cache:*")
            logger.info("All cache cleared", deleted_keys=deleted)
            return True
        except Exception as e:
            logger.error("Cache clear error", error=str(e))
            return False
    
    async def warm_cache(
        self,
        endpoint_callable: Callable,
        cache_key: str,
        config: CacheConfig
    ):
        """
        Warm cache by pre-fetching data.
        
        Args:
            endpoint_callable: Async function to fetch data
            cache_key: Key to cache under
            config: Cache configuration
        """
        try:
            logger.info("Warming cache", key=cache_key)
            result = await endpoint_callable()
            await self.set(cache_key, result, config=config)
        except Exception as e:
            logger.error("Cache warming failed", key=cache_key, error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "sets": self.stats["sets"],
            "deletes": self.stats["deletes"],
            "errors": self.stats["errors"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "miss_rate": 1.0 - hit_rate if total_requests > 0 else 0.0
        }
    
    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
