"""GNN Cache Manager with TTL and Invalidation.

This module provides a comprehensive caching layer for GNN embeddings,
insights, and query results with TTL support and invalidation.
"""

import os
import json
import logging
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import Redis for distributed caching
try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None


class GNNCacheEntry:
    """Cache entry with metadata."""
    
    def __init__(
        self,
        data: Any,
        created_at: datetime,
        expires_at: Optional[datetime] = None,
        domain_id: Optional[str] = None,
        cache_type: str = "embedding",
        tags: Optional[List[str]] = None
    ):
        """Initialize cache entry.
        
        Args:
            data: Cached data
            created_at: Creation timestamp
            expires_at: Expiration timestamp (None for no expiration)
            domain_id: Optional domain identifier
            cache_type: Type of cache entry (embedding, insight, classification, etc.)
            tags: Optional tags for invalidation
        """
        self.data = data
        self.created_at = created_at
        self.expires_at = expires_at
        self.domain_id = domain_id
        self.cache_type = cache_type
        self.tags = tags or []
        self.access_count = 0
        self.last_accessed = created_at
    
    def is_expired(self) -> bool:
        """Check if entry is expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class GNNCacheManager:
    """Comprehensive cache manager for GNN operations.
    
    Supports TTL, invalidation, and both Redis and in-memory caching.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_ttl: int = 3600,  # 1 hour
        cache_dir: Optional[str] = None,
        max_memory_size: int = 1000,
        enable_persistent_cache: bool = True
    ):
        """Initialize cache manager.
        
        Args:
            redis_url: Redis connection URL (optional)
            default_ttl: Default TTL in seconds
            cache_dir: Directory for persistent cache files
            max_memory_size: Maximum number of in-memory entries
            enable_persistent_cache: Enable persistent file-based cache
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.default_ttl = default_ttl
        self.cache_dir = cache_dir or os.getenv("GNN_CACHE_DIR", "./gnn_cache")
        self.max_memory_size = max_memory_size
        self.enable_persistent_cache = enable_persistent_cache
        
        # Redis client (if available)
        self.redis_client = None
        if self.redis_url and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("✅ Redis cache enabled")
            except Exception as e:
                logger.warning(f"⚠️  Redis not available, using in-memory cache: {e}")
                self.redis_client = None
        
        # In-memory cache
        self.memory_cache: Dict[str, GNNCacheEntry] = {}
        self.lock = threading.RLock()
        
        # Persistent cache directory
        if self.enable_persistent_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.cache_index_file = os.path.join(self.cache_dir, "cache_index.json")
            self._load_cache_index()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "invalidations": 0,
            "expired": 0,
        }
    
    def _generate_cache_key(
        self,
        cache_type: str,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        domain_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key.
        
        Args:
            cache_type: Type of cache (embedding, insight, classification, etc.)
            nodes: Graph nodes (optional)
            edges: Graph edges (optional)
            query: Query string (optional)
            domain_id: Domain identifier (optional)
            config: Additional configuration (optional)
        
        Returns:
            Cache key string
        """
        key_parts = [cache_type]
        
        if domain_id:
            key_parts.append(f"domain:{domain_id}")
        
        if query:
            key_parts.append(f"query:{hashlib.sha256(query.encode()).hexdigest()[:16]}")
        
        if nodes and edges:
            # Create hashable representation
            graph_repr = {
                "nodes": sorted([(n.get("id", ""), n.get("type", "")) for n in nodes]),
                "edges": sorted([(e.get("source_id", ""), e.get("target_id", "")) for e in edges]),
            }
            graph_str = json.dumps(graph_repr, sort_keys=True)
            key_parts.append(f"graph:{hashlib.sha256(graph_str.encode()).hexdigest()[:16]}")
        
        if config:
            config_str = json.dumps(config, sort_keys=True)
            key_parts.append(f"config:{hashlib.sha256(config_str.encode()).hexdigest()[:16]}")
        
        return ":".join(key_parts)
    
    def get(
        self,
        cache_type: str,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        domain_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """Get cached value.
        
        Args:
            cache_type: Type of cache
            nodes: Graph nodes (optional)
            edges: Graph edges (optional)
            query: Query string (optional)
            domain_id: Domain identifier (optional)
            config: Additional configuration (optional)
        
        Returns:
            Cached data or None
        """
        cache_key = self._generate_cache_key(cache_type, nodes, edges, query, domain_id, config)
        
        with self.lock:
            # Try Redis first
            if self.redis_client:
                try:
                    cached_data = self.redis_client.get(cache_key)
                    if cached_data:
                        entry = pickle.loads(cached_data)
                        if not entry.is_expired():
                            entry.touch()
                            self.stats["hits"] += 1
                            return entry.data
                        else:
                            # Expired, remove from Redis
                            self.redis_client.delete(cache_key)
                            self.stats["expired"] += 1
                except Exception as e:
                    logger.debug(f"Redis get failed: {e}")
            
            # Try memory cache
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                if entry.is_expired():
                    del self.memory_cache[cache_key]
                    self.stats["expired"] += 1
                    self.stats["misses"] += 1
                    return None
                
                entry.touch()
                self.stats["hits"] += 1
                return entry.data
            
            # Try persistent cache
            if self.enable_persistent_cache:
                try:
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as f:
                            entry = pickle.load(f)
                        if not entry.is_expired():
                            entry.touch()
                            # Promote to memory cache
                            self._add_to_memory_cache(cache_key, entry)
                            self.stats["hits"] += 1
                            return entry.data
                        else:
                            os.remove(cache_file)
                            self.stats["expired"] += 1
                except Exception as e:
                    logger.debug(f"Persistent cache get failed: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def set(
        self,
        cache_type: str,
        data: Any,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
        domain_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None
    ):
        """Set cached value.
        
        Args:
            cache_type: Type of cache
            data: Data to cache
            nodes: Graph nodes (optional)
            edges: Graph edges (optional)
            query: Query string (optional)
            domain_id: Domain identifier (optional)
            config: Additional configuration (optional)
            ttl: TTL in seconds (uses default if None)
            tags: Optional tags for invalidation
        """
        cache_key = self._generate_cache_key(cache_type, nodes, edges, query, domain_id, config)
        
        if ttl is None:
            ttl = self.default_ttl
        
        expires_at = datetime.now() + timedelta(seconds=ttl)
        entry = GNNCacheEntry(
            data=data,
            created_at=datetime.now(),
            expires_at=expires_at,
            domain_id=domain_id,
            cache_type=cache_type,
            tags=tags or []
        )
        
        with self.lock:
            # Store in Redis
            if self.redis_client:
                try:
                    serialized = pickle.dumps(entry)
                    self.redis_client.setex(cache_key, ttl, serialized)
                except Exception as e:
                    logger.debug(f"Redis set failed: {e}")
            
            # Store in memory
            self._add_to_memory_cache(cache_key, entry)
            
            # Store persistently
            if self.enable_persistent_cache:
                try:
                    cache_file = os.path.join(self.cache_dir, f"{cache_key}.pkl")
                    with open(cache_file, "wb") as f:
                        pickle.dump(entry, f)
                except Exception as e:
                    logger.debug(f"Persistent cache set failed: {e}")
            
            self.stats["sets"] += 1
    
    def _add_to_memory_cache(self, cache_key: str, entry: GNNCacheEntry):
        """Add entry to memory cache with size management.
        
        Args:
            cache_key: Cache key
            entry: Cache entry
        """
        # Remove expired entries first
        expired_keys = [
            key for key, e in self.memory_cache.items()
            if e.is_expired()
        ]
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still at max size, remove least recently used
        if len(self.memory_cache) >= self.max_memory_size:
            lru_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k].last_accessed
            )
            del self.memory_cache[lru_key]
        
        self.memory_cache[cache_key] = entry
    
    def invalidate(
        self,
        cache_type: Optional[str] = None,
        domain_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> int:
        """Invalidate cache entries.
        
        Args:
            cache_type: Invalidate entries of specific type (None for all)
            domain_id: Invalidate entries for specific domain (None for all)
            tags: Invalidate entries with specific tags (None for all)
            pattern: Invalidate entries matching key pattern (None for all)
        
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        with self.lock:
            # Invalidate Redis entries
            if self.redis_client:
                try:
                    if pattern:
                        keys = list(self.redis_client.scan_iter(match=pattern))
                        if keys:
                            self.redis_client.delete(*keys)
                            invalidated += len(keys)
                    else:
                        # Scan all keys and match criteria
                        keys_to_delete = []
                        for key in self.redis_client.scan_iter(match="*"):
                            try:
                                cached_data = self.redis_client.get(key)
                                if cached_data:
                                    entry = pickle.loads(cached_data)
                                    if self._matches_invalidation_criteria(
                                        entry, cache_type, domain_id, tags
                                    ):
                                        keys_to_delete.append(key)
                            except Exception:
                                pass
                        
                        if keys_to_delete:
                            self.redis_client.delete(*keys_to_delete)
                            invalidated += len(keys_to_delete)
                except Exception as e:
                    logger.debug(f"Redis invalidation failed: {e}")
            
            # Invalidate memory cache
            keys_to_remove = []
            for key, entry in self.memory_cache.items():
                if self._matches_invalidation_criteria(entry, cache_type, domain_id, tags):
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
            
            # Invalidate persistent cache
            if self.enable_persistent_cache:
                try:
                    for cache_file in Path(self.cache_dir).glob("*.pkl"):
                        try:
                            with open(cache_file, "rb") as f:
                                entry = pickle.load(f)
                            if self._matches_invalidation_criteria(entry, cache_type, domain_id, tags):
                                os.remove(cache_file)
                                invalidated += 1
                        except Exception:
                            pass
                except Exception as e:
                    logger.debug(f"Persistent cache invalidation failed: {e}")
        
        self.stats["invalidations"] += invalidated
        logger.info(f"Invalidated {invalidated} cache entries")
        return invalidated
    
    def _matches_invalidation_criteria(
        self,
        entry: GNNCacheEntry,
        cache_type: Optional[str],
        domain_id: Optional[str],
        tags: Optional[List[str]]
    ) -> bool:
        """Check if entry matches invalidation criteria.
        
        Args:
            entry: Cache entry
            cache_type: Cache type filter
            domain_id: Domain filter
            tags: Tags filter
        
        Returns:
            True if matches, False otherwise
        """
        if cache_type and entry.cache_type != cache_type:
            return False
        
        if domain_id and entry.domain_id != domain_id:
            return False
        
        if tags and not any(tag in entry.tags for tag in tags):
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "sets": self.stats["sets"],
                "invalidations": self.stats["invalidations"],
                "expired": self.stats["expired"],
                "hit_rate": round(hit_rate, 2),
                "memory_cache_size": len(self.memory_cache),
                "max_memory_size": self.max_memory_size,
                "redis_enabled": self.redis_client is not None,
                "persistent_cache_enabled": self.enable_persistent_cache,
            }
    
    def clear(self):
        """Clear all caches."""
        with self.lock:
            # Clear Redis
            if self.redis_client:
                try:
                    self.redis_client.flushdb()
                except Exception as e:
                    logger.debug(f"Redis clear failed: {e}")
            
            # Clear memory
            self.memory_cache.clear()
            
            # Clear persistent
            if self.enable_persistent_cache:
                try:
                    for cache_file in Path(self.cache_dir).glob("*.pkl"):
                        os.remove(cache_file)
                except Exception as e:
                    logger.debug(f"Persistent cache clear failed: {e}")
        
        logger.info("Cache cleared")
    
    def _load_cache_index(self):
        """Load cache index from disk (for persistent cache management)."""
        # This is a placeholder for future index management
        pass

