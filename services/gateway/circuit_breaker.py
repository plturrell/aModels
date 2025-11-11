"""
Circuit breaker implementation for backend service resilience.
Prevents cascading failures by stopping requests to failing services.
"""
import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional, TypeVar
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures detected, blocking requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Failures before opening circuit
    success_threshold: int = 2  # Successes to close from half-open
    timeout: float = 60.0       # Seconds before trying half-open
    
    
@dataclass
class CircuitBreakerStats:
    """Circuit breaker statistics."""
    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    successes: int = 0
    last_failure_time: Optional[float] = None
    opened_at: Optional[float] = None
    total_requests: int = 0
    total_failures: int = 0
    total_successes: int = 0


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests blocked immediately
    - HALF_OPEN: Testing recovery, limited requests pass through
    """
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self.stats.state
        
    async def call(self, func: Callable[[], T]) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Async function to execute
            
        Returns:
            Result of function call
            
        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Original exception from function
        """
        async with self._lock:
            self.stats.total_requests += 1
            
            # Check if circuit should transition to half-open
            if self.stats.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self.stats.state = CircuitState.HALF_OPEN
                    self.stats.successes = 0
                    logger.info(f"Circuit breaker '{self.name}' entering HALF_OPEN state")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN. "
                        f"Service unavailable. Will retry after {self.config.timeout}s"
                    )
        
        # Execute function (outside lock to allow concurrent requests in CLOSED state)
        try:
            result = await func()
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
            
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.stats.opened_at is None:
            return False
        return time.time() - self.stats.opened_at >= self.config.timeout
        
    async def _on_success(self):
        """Handle successful request."""
        async with self._lock:
            self.stats.total_successes += 1
            
            if self.stats.state == CircuitState.HALF_OPEN:
                self.stats.successes += 1
                if self.stats.successes >= self.config.success_threshold:
                    # Recovered! Close circuit
                    self.stats.state = CircuitState.CLOSED
                    self.stats.failures = 0
                    self.stats.successes = 0
                    logger.info(
                        f"Circuit breaker '{self.name}' recovered. "
                        f"State: CLOSED. Total requests: {self.stats.total_requests}, "
                        f"Success rate: {self._success_rate():.1%}"
                    )
            elif self.stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self.stats.failures = 0
                
    async def _on_failure(self, error: Exception):
        """Handle failed request."""
        async with self._lock:
            self.stats.total_failures += 1
            self.stats.failures += 1
            self.stats.last_failure_time = time.time()
            
            if self.stats.state == CircuitState.HALF_OPEN:
                # Failed during recovery, reopen circuit
                self.stats.state = CircuitState.OPEN
                self.stats.opened_at = time.time()
                logger.warning(
                    f"Circuit breaker '{self.name}' reopened after failure in HALF_OPEN. "
                    f"Error: {error}"
                )
            elif self.stats.state == CircuitState.CLOSED:
                if self.stats.failures >= self.config.failure_threshold:
                    # Too many failures, open circuit
                    self.stats.state = CircuitState.OPEN
                    self.stats.opened_at = time.time()
                    logger.error(
                        f"Circuit breaker '{self.name}' opened after {self.stats.failures} failures. "
                        f"Error: {error}. Will retry after {self.config.timeout}s"
                    )
                    
    def _success_rate(self) -> float:
        """Calculate success rate."""
        if self.stats.total_requests == 0:
            return 0.0
        return self.stats.total_successes / self.stats.total_requests
        
    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self.stats.state.value,
            "failures": self.stats.failures,
            "successes": self.stats.successes,
            "total_requests": self.stats.total_requests,
            "total_failures": self.stats.total_failures,
            "total_successes": self.stats.total_successes,
            "success_rate": self._success_rate(),
            "opened_at": self.stats.opened_at,
            "last_failure_time": self.stats.last_failure_time
        }
        
    async def reset(self):
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self.stats.state = CircuitState.CLOSED
            self.stats.failures = 0
            self.stats.successes = 0
            logger.info(f"Circuit breaker '{self.name}' manually reset to CLOSED")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerManager:
    """Manages multiple circuit breakers for different services."""
    
    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        
    def get_breaker(self, service_name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self._breakers:
            self._breakers[service_name] = CircuitBreaker(service_name, config)
        return self._breakers[service_name]
        
    def get_all_stats(self) -> dict:
        """Get statistics for all circuit breakers."""
        return {
            name: breaker.get_stats()
            for name, breaker in self._breakers.items()
        }
        
    async def reset_all(self):
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
