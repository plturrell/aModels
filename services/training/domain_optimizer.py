"""Domain-specific optimizations (caching, batching, etc.).

This module provides domain-aware optimizations for improved performance.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class DomainOptimizer:
    """Optimize domain-specific operations."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600
    ):
        """Initialize domain optimizer.
        
        Args:
            redis_url: Redis connection string for caching
            cache_ttl: Cache TTL in seconds
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.cache_ttl = cache_ttl
        
        # In-memory cache (fallback if Redis not available)
        self.memory_cache: Dict[str, tuple] = {}
        
        # Batch queues per domain
        self.batch_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Domain-specific optimization configs
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
    
    def get_cached_response(
        self,
        domain_id: str,
        query: str,
        query_hash: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response for a query.
        
        Args:
            domain_id: Domain identifier
            query: Query text
            query_hash: Optional pre-computed hash
        
        Returns:
            Cached response if available, None otherwise
        """
        if query_hash is None:
            query_hash = self._hash_query(domain_id, query)
        
        cache_key = f"domain_cache:{domain_id}:{query_hash}"
        
        # Try Redis first
        if self.redis_url:
            try:
                import redis
                r = redis.from_url(self.redis_url)
                cached = r.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.debug(f"Redis cache miss: {e}")
        
        # Try memory cache
        if cache_key in self.memory_cache:
            cached_data, expiry = self.memory_cache[cache_key]
            if datetime.now() < expiry:
                return cached_data
            else:
                del self.memory_cache[cache_key]
        
        return None
    
    def cache_response(
        self,
        domain_id: str,
        query: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ):
        """Cache a response for a query.
        
        Args:
            domain_id: Domain identifier
            query: Query text
            response: Response to cache
            ttl: Optional TTL override
        """
        query_hash = self._hash_query(domain_id, query)
        cache_key = f"domain_cache:{domain_id}:{query_hash}"
        
        if ttl is None:
            ttl = self.cache_ttl
        
        expiry = datetime.now() + timedelta(seconds=ttl)
        
        # Cache in Redis
        if self.redis_url:
            try:
                import redis
                r = redis.from_url(self.redis_url)
                r.setex(
                    cache_key,
                    ttl,
                    json.dumps(response)
                )
            except Exception as e:
                logger.debug(f"Redis cache write failed: {e}")
        
        # Cache in memory (fallback)
        self.memory_cache[cache_key] = (response, expiry)
        
        # Cleanup old memory cache entries
        self._cleanup_memory_cache()
    
    def add_to_batch(
        self,
        domain_id: str,
        request: Dict[str, Any]
    ) -> bool:
        """Add a request to batch queue.
        
        Args:
            domain_id: Domain identifier
            request: Request to batch
        
        Returns:
            True if request added, False if batch should be processed
        """
        batch_config = self.domain_configs.get(domain_id, {})
        batch_size = batch_config.get("batch_size", 10)
        batch_timeout = batch_config.get("batch_timeout", 5)  # seconds
        
        queue = self.batch_queues[domain_id]
        queue.append({
            "request": request,
            "added_at": datetime.now(),
        })
        
        # Check if batch is ready
        if len(queue) >= batch_size:
            return False  # Batch ready, don't add more
        
        # Check if oldest request is too old
        if queue:
            oldest = queue[0]["added_at"]
            if (datetime.now() - oldest).total_seconds() >= batch_timeout:
                return False  # Batch timeout reached
        
        return True  # Request added to batch
    
    def get_batch(
        self,
        domain_id: str
    ) -> List[Dict[str, Any]]:
        """Get and clear batch queue for a domain.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            List of batched requests
        """
        batch = self.batch_queues[domain_id].copy()
        self.batch_queues[domain_id] = []
        return batch
    
    def configure_domain_optimizations(
        self,
        domain_id: str,
        config: Dict[str, Any]
    ):
        """Configure optimizations for a domain.
        
        Args:
            domain_id: Domain identifier
            config: Optimization configuration:
                - cache_enabled: Enable caching
                - cache_ttl: Cache TTL in seconds
                - batch_enabled: Enable batching
                - batch_size: Batch size
                - batch_timeout: Batch timeout in seconds
        """
        self.domain_configs[domain_id] = config
        logger.info(f"✅ Configured optimizations for domain {domain_id}")
    
    def get_optimization_stats(
        self,
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get optimization statistics.
        
        Args:
            domain_id: Optional domain ID (if None, returns all)
        
        Returns:
            Optimization statistics
        """
        stats = {
            "cache": {},
            "batching": {},
        }
        
        domains = [domain_id] if domain_id else list(self.domain_configs.keys())
        
        for did in domains:
            config = self.domain_configs.get(did, {})
            
            stats["cache"][did] = {
                "enabled": config.get("cache_enabled", True),
                "ttl": config.get("cache_ttl", self.cache_ttl),
            }
            
            stats["batching"][did] = {
                "enabled": config.get("batch_enabled", False),
                "batch_size": config.get("batch_size", 10),
                "current_queue_size": len(self.batch_queues.get(did, [])),
            }
        
        return stats
    
    def _hash_query(self, domain_id: str, query: str) -> str:
        """Generate hash for query caching."""
        combined = f"{domain_id}:{query}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _cleanup_memory_cache(self):
        """Cleanup expired memory cache entries."""
        now = datetime.now()
        expired_keys = [
            key for key, (_, expiry) in self.memory_cache.items()
            if now >= expiry
        ]
        for key in expired_keys:
            del self.memory_cache[key]

