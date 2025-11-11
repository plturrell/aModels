"""LLM client pool with connection reuse, rate limiting, and caching."""

import hashlib
import json
import logging
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False
    redis = None

logger = logging.getLogger(__name__)


@dataclass
class LLMPoolConfig:
    """Configuration for LLM client pool."""
    max_connections: int = 10
    rate_limit: int = 100  # requests per minute
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5 minutes
    redis_url: Optional[str] = None


class TokenBucket:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int, capacity: Optional[int] = None):
        """Initialize token bucket.
        
        Args:
            rate: Tokens per minute
            capacity: Maximum bucket capacity (defaults to rate)
        """
        self.rate = rate
        self.capacity = capacity or rate
        self.tokens = float(self.capacity)
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if tokens acquired, False if rate limited
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            # Add tokens based on elapsed time (rate per minute)
            self.tokens = min(
                self.capacity,
                self.tokens + (elapsed * self.rate / 60.0)
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_for_token(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait until tokens are available.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start = time.time()
        while not self.acquire(tokens):
            if timeout and (time.time() - start) > timeout:
                return False
            time.sleep(0.1)
        return True


class LLMClientPool:
    """Pool for LLM clients with rate limiting and caching."""
    
    def __init__(self, config: LLMPoolConfig, factory: Callable[[], Any]):
        """Initialize LLM client pool.
        
        Args:
            config: Pool configuration
            factory: Function to create new LLM clients
        """
        self.config = config
        self.factory = factory
        self.pool = queue.Queue(maxsize=config.max_connections)
        self.rate_limiter = TokenBucket(config.rate_limit)
        self.cache: Dict[str, Any] = {}
        self.cache_lock = threading.Lock()
        self.redis_client = None
        
        # Initialize Redis if available
        if config.cache_enabled and config.redis_url and HAS_REDIS:
            try:
                self.redis_client = redis.from_url(config.redis_url)
                self.redis_client.ping()
                logger.info("LLM pool: Redis cache enabled")
            except Exception as e:
                logger.warning(f"LLM pool: Redis unavailable, using memory cache: {e}")
                self.redis_client = None
        
        # Pre-populate pool
        for _ in range(min(3, config.max_connections)):
            try:
                client = factory()
                self.pool.put(client, block=False)
            except Exception as e:
                logger.warning(f"LLM pool: Failed to create initial client: {e}")
    
    def get_client(self, timeout: Optional[float] = None) -> Any:
        """Get a client from the pool.
        
        Args:
            timeout: Maximum time to wait for a client
            
        Returns:
            LLM client instance
        """
        try:
            client = self.pool.get(timeout=timeout)
            return client
        except queue.Empty:
            # Pool empty, create new client
            logger.debug("LLM pool: Creating new client (pool exhausted)")
            return self.factory()
    
    def return_client(self, client: Any) -> None:
        """Return a client to the pool.
        
        Args:
            client: LLM client to return
        """
        try:
            self.pool.put(client, block=False)
        except queue.Full:
            # Pool full, discard client
            logger.debug("LLM pool: Discarding client (pool full)")
            if hasattr(client, 'close'):
                try:
                    client.close()
                except Exception:
                    pass
    
    def _cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate cache key for prompt.
        
        Args:
            prompt: User prompt
            model: Model name
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        key_data = {
            'prompt': prompt,
            'model': model,
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def invoke_with_cache(
        self,
        client: Any,
        messages: List[Dict[str, str]],
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Invoke LLM with caching.
        
        Args:
            client: LLM client
            messages: List of messages
            cache_key: Optional cache key (auto-generated if not provided)
            **kwargs: Additional arguments for LLM invocation
            
        Returns:
            LLM response
        """
        if not self.config.cache_enabled:
            return self._invoke_llm(client, messages, **kwargs)
        
        # Generate cache key
        if cache_key is None:
            prompt = str(messages)
            model = kwargs.get('model', 'default')
            cache_key = self._cache_key(prompt, model, **kwargs)
        
        # Try cache first
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            logger.debug("LLM pool: Cache hit")
            return cached
        
        # Wait for rate limit
        if not self.rate_limiter.wait_for_token(timeout=30.0):
            raise RuntimeError("Rate limit timeout exceeded")
        
        # Invoke LLM
        response = self._invoke_llm(client, messages, **kwargs)
        
        # Cache response
        self._set_cache(cache_key, response)
        
        return response
    
    def _invoke_llm(self, client: Any, messages: List[Dict[str, str]], **kwargs) -> Any:
        """Invoke LLM client.
        
        Args:
            client: LLM client
            messages: List of messages
            **kwargs: Additional arguments
            
        Returns:
            LLM response
        """
        if hasattr(client, 'invoke'):
            return client.invoke(messages, **kwargs)
        elif hasattr(client, '__call__'):
            return client(messages, **kwargs)
        else:
            raise ValueError("LLM client does not support invoke or call")
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try Redis first
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"llm:{key}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"LLM pool: Redis cache get error: {e}")
        
        # Try memory cache
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry['expires']:
                    return entry['value']
                else:
                    del self.cache[key]
        
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Try Redis first
        if self.redis_client:
            try:
                cached = json.dumps(value)
                self.redis_client.setex(
                    f"llm:{key}",
                    self.config.cache_ttl,
                    cached
                )
                return
            except Exception as e:
                logger.warning(f"LLM pool: Redis cache set error: {e}")
        
        # Fall back to memory cache
        with self.cache_lock:
            # Limit memory cache size
            if len(self.cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(
                    self.cache.keys(),
                    key=lambda k: self.cache[k]['expires']
                )
                for old_key in sorted_keys[:100]:
                    del self.cache[old_key]
            
            self.cache[key] = {
                'value': value,
                'expires': time.time() + self.config.cache_ttl
            }
    
    def clear_cache(self) -> None:
        """Clear all cached responses."""
        with self.cache_lock:
            self.cache.clear()
        
        if self.redis_client:
            try:
                # Clear Redis keys with pattern
                for key in self.redis_client.scan_iter(match="llm:*"):
                    self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"LLM pool: Redis cache clear error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        return {
            'pool_size': self.pool.qsize(),
            'max_connections': self.config.max_connections,
            'cache_size': len(self.cache),
            'rate_limit': self.config.rate_limit,
            'cache_enabled': self.config.cache_enabled,
            'redis_enabled': self.redis_client is not None,
        }

