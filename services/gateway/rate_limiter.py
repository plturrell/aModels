"""
Rate limiting middleware for API protection.
"""
import time
import logging
from collections import defaultdict
from typing import Dict, Tuple
import asyncio

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.
    Allows burst traffic while enforcing average rate limit.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per client
            burst_size: Maximum burst size (tokens in bucket)
        """
        self.rate = requests_per_minute / 60.0  # requests per second
        self.burst_size = burst_size
        self.buckets: Dict[str, Tuple[float, float]] = {}  # {client_id: (tokens, last_update)}
        self._lock = asyncio.Lock()
        
    async def is_allowed(self, client_id: str) -> Tuple[bool, Dict[str, any]]:
        """
        Check if request is allowed for client.
        
        Args:
            client_id: Client identifier (IP address, API key, etc.)
            
        Returns:
            Tuple of (is_allowed, info_dict)
        """
        async with self._lock:
            now = time.time()
            
            if client_id not in self.buckets:
                # New client, initialize with full bucket
                self.buckets[client_id] = (self.burst_size - 1, now)
                return True, {
                    "allowed": True,
                    "tokens_remaining": self.burst_size - 1,
                    "retry_after": 0
                }
                
            tokens, last_update = self.buckets[client_id]
            
            # Add tokens based on time elapsed
            time_passed = now - last_update
            tokens = min(self.burst_size, tokens + time_passed * self.rate)
            
            if tokens >= 1.0:
                # Allow request and consume token
                self.buckets[client_id] = (tokens - 1, now)
                return True, {
                    "allowed": True,
                    "tokens_remaining": int(tokens - 1),
                    "retry_after": 0
                }
            else:
                # Rate limit exceeded
                retry_after = (1.0 - tokens) / self.rate
                self.buckets[client_id] = (tokens, now)
                return False, {
                    "allowed": False,
                    "tokens_remaining": 0,
                    "retry_after": int(retry_after) + 1
                }
                
    async def cleanup_old_entries(self, max_age_seconds: int = 3600):
        """Remove old entries to prevent memory leak."""
        async with self._lock:
            now = time.time()
            to_remove = [
                client_id
                for client_id, (tokens, last_update) in self.buckets.items()
                if now - last_update > max_age_seconds
            ]
            for client_id in to_remove:
                del self.buckets[client_id]
            if to_remove:
                logger.info(f"Cleaned up {len(to_remove)} rate limiter entries")
                
    def get_stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "total_clients": len(self.buckets),
            "rate_limit": f"{self.rate * 60:.0f} req/min",
            "burst_size": self.burst_size
        }


class MultiTierRateLimiter:
    """
    Multi-tier rate limiter with different limits for different endpoints.
    """
    
    def __init__(self):
        # Default: 100 req/min, burst 20
        self.default_limiter = RateLimiter(100, 20)
        
        # Strict: 20 req/min, burst 5 (for expensive operations)
        self.strict_limiter = RateLimiter(20, 5)
        
        # Lenient: 300 req/min, burst 50 (for health checks, etc.)
        self.lenient_limiter = RateLimiter(300, 50)
        
        self._cleanup_task = None
        
    def get_limiter(self, path: str) -> RateLimiter:
        """Get appropriate rate limiter for endpoint."""
        # Expensive operations get strict limits
        if any(x in path for x in [
            "/api/perplexity/process",
            "/api/dms/process",
            "/api/relational/process",
            "/api/murex/process",
            "/search/unified",
            "/search/narrative",
            "/deep-research/research",
            "/catalog/data-products/build"
        ]):
            return self.strict_limiter
            
        # Health checks get lenient limits
        if any(x in path for x in ["/healthz", "/health", "/ping"]):
            return self.lenient_limiter
            
        # Default for everything else
        return self.default_limiter
        
    async def is_allowed(self, client_id: str, path: str) -> Tuple[bool, Dict]:
        """Check if request is allowed."""
        limiter = self.get_limiter(path)
        return await limiter.is_allowed(client_id)
        
    async def start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while True:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.default_limiter.cleanup_old_entries()
                await self.strict_limiter.cleanup_old_entries()
                await self.lenient_limiter.cleanup_old_entries()
                
        self._cleanup_task = asyncio.create_task(cleanup_loop())
        
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
    def get_stats(self) -> Dict:
        """Get statistics for all limiters."""
        return {
            "default": self.default_limiter.get_stats(),
            "strict": self.strict_limiter.get_stats(),
            "lenient": self.lenient_limiter.get_stats()
        }
