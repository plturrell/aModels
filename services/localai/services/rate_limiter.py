"""Rate limiting and throttling for translation requests."""

from __future__ import annotations

import time
import logging
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 20  # Allow burst of requests


class RateLimiter:
    """Rate limiter using token bucket algorithm."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter."""
        self.config = config or RateLimitConfig()
        self.lock = Lock()
        
        # Track requests per client (by IP or API key)
        self.client_buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: {
            "tokens": float(self.config.burst_size),
            "last_update": time.time()
        })
        
        # Global rate limiting
        self.global_bucket = {
            "tokens": float(self.config.burst_size),
            "last_update": time.time()
        }
    
    def _refill_tokens(self, bucket: Dict[str, float], rate: float, max_tokens: float):
        """Refill tokens in a bucket based on elapsed time."""
        now = time.time()
        elapsed = now - bucket["last_update"]
        
        # Add tokens based on rate
        new_tokens = min(max_tokens, bucket["tokens"] + elapsed * rate)
        bucket["tokens"] = new_tokens
        bucket["last_update"] = now
    
    def _check_limit(
        self,
        bucket: Dict[str, float],
        rate: float,
        max_tokens: float,
        tokens_needed: float = 1.0
    ) -> bool:
        """Check if request is allowed and consume tokens."""
        with self.lock:
            self._refill_tokens(bucket, rate, max_tokens)
            
            if bucket["tokens"] >= tokens_needed:
                bucket["tokens"] -= tokens_needed
                return True
            return False
    
    def is_allowed(self, client_id: Optional[str] = None) -> tuple[bool, Optional[str]]:
        """Check if request is allowed.
        
        Args:
            client_id: Optional client identifier (IP, API key, etc.)
        
        Returns:
            Tuple of (is_allowed, error_message)
        """
        # Check global rate limit
        if not self._check_limit(
            self.global_bucket,
            self.config.requests_per_second,
            self.config.burst_size
        ):
            return False, "Global rate limit exceeded"
        
        # Check per-client rate limit if client_id provided
        if client_id:
            client_bucket = self.client_buckets[client_id]
            
            # Check per-second limit
            if not self._check_limit(
                client_bucket,
                self.config.requests_per_second,
                self.config.burst_size
            ):
                return False, "Rate limit exceeded: too many requests per second"
        
        return True, None
    
    def get_remaining_requests(self, client_id: Optional[str] = None) -> int:
        """Get remaining requests for a client."""
        if client_id:
            bucket = self.client_buckets[client_id]
            self._refill_tokens(bucket, self.config.requests_per_second, self.config.burst_size)
            return int(bucket["tokens"])
        else:
            self._refill_tokens(self.global_bucket, self.config.requests_per_second, self.config.burst_size)
            return int(self.global_bucket["tokens"])
    
    def reset(self, client_id: Optional[str] = None):
        """Reset rate limit for a client or globally."""
        with self.lock:
            if client_id:
                if client_id in self.client_buckets:
                    del self.client_buckets[client_id]
            else:
                self.global_bucket = {
                    "tokens": float(self.config.burst_size),
                    "last_update": time.time()
                }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> Optional[RateLimiter]:
    """Get the global rate limiter instance."""
    global _rate_limiter
    
    if _rate_limiter is None:
        import os
        
        # Initialize from environment variables
        requests_per_second = float(os.getenv("RATE_LIMIT_RPS", "10.0"))
        requests_per_minute = float(os.getenv("RATE_LIMIT_RPM", "100.0"))
        requests_per_hour = float(os.getenv("RATE_LIMIT_RPH", "1000.0"))
        burst_size = int(os.getenv("RATE_LIMIT_BURST", "20"))
        enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        
        if enabled:
            config = RateLimitConfig(
                requests_per_second=requests_per_second,
                requests_per_minute=requests_per_minute,
                requests_per_hour=requests_per_hour,
                burst_size=burst_size
            )
            _rate_limiter = RateLimiter(config)
            logger.info(
                f"Rate limiting enabled: {requests_per_second} req/s, "
                f"burst: {burst_size}"
            )
        else:
            logger.info("Rate limiting disabled")
    
    return _rate_limiter

