"""Caching layer for TOON translation to improve performance."""

from __future__ import annotations

import os
import hashlib
import json
import time
import logging
from typing import Optional, Dict, Any, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


class TOONTranslationCache:
    """Simple in-memory cache for translations."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize cache.
        
        Args:
            max_size: Maximum number of cached entries
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
    
    def _make_key(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str,
        use_toon: bool,
        toon_config: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a cache key from request parameters."""
        # Normalize text (strip whitespace, lowercase for key)
        normalized_text = text.strip()
        
        # Create hash of all parameters
        key_data = {
            "text": normalized_text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "model": model,
            "use_toon": use_toon,
            "toon_config": toon_config or {}
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str,
        use_toon: bool,
        toon_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[str, Optional[Dict[str, Any]]]]:
        """Get cached translation.
        
        Returns:
            Tuple of (translated_text, toon_data) or None if not cached
        """
        key = self._make_key(text, source_lang, target_lang, model, use_toon, toon_config)
        
        if key not in self.cache:
            self.misses += 1
            return None
        
        entry = self.cache[key]
        
        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self.cache[key]
            self.misses += 1
            return None
        
        self.hits += 1
        logger.debug(f"Cache hit for key: {key[:16]}...")
        return entry["translated_text"], entry.get("toon_data")
    
    def set(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        model: str,
        use_toon: bool,
        translated_text: str,
        toon_data: Optional[Dict[str, Any]] = None,
        toon_config: Optional[Dict[str, Any]] = None
    ):
        """Cache a translation."""
        key = self._make_key(text, source_lang, target_lang, model, use_toon, toon_config)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(
                self.cache.keys(),
                key=lambda k: self.cache[k]["timestamp"]
            )
            del self.cache[oldest_key]
        
        self.cache[key] = {
            "translated_text": translated_text,
            "toon_data": toon_data,
            "timestamp": time.time()
        }
        logger.debug(f"Cached translation for key: {key[:16]}...")
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        logger.info("Translation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


# Global cache instance
_cache: Optional[TOONTranslationCache] = None


def get_cache() -> Optional[TOONTranslationCache]:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        # Initialize cache from environment
        max_size = int(os.getenv("TOON_CACHE_MAX_SIZE", "1000"))
        ttl = int(os.getenv("TOON_CACHE_TTL_SECONDS", "3600"))
        cache_enabled = os.getenv("TOON_CACHE_ENABLED", "true").lower() == "true"
        
        if cache_enabled:
            _cache = TOONTranslationCache(max_size=max_size, ttl_seconds=ttl)
            logger.info(f"Translation cache enabled: max_size={max_size}, ttl={ttl}s")
        else:
            logger.info("Translation cache disabled")
    
    return _cache

