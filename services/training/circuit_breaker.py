"""Circuit Breaker Pattern for Service Resilience."""

import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerError(Exception):
    pass


class CircuitBreaker:
    """Circuit breaker for service calls."""
    
    def __init__(self, name: str, failure_threshold: int = 5, 
                 recovery_timeout: int = 60, success_threshold: int = 2):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.RLock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._check_recovery()
            return self._state
    
    def _check_recovery(self):
        if (self._state == CircuitState.OPEN and self._last_failure_time):
            elapsed = (datetime.now() - self._last_failure_time).total_seconds()
            if elapsed >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                logger.info(f"Circuit '{self.name}' -> HALF_OPEN")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        with self._lock:
            self._check_recovery()
            if self._state == CircuitState.OPEN:
                raise CircuitBreakerError(f"Circuit '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info(f"Circuit '{self.name}' -> CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _on_failure(self):
        with self._lock:
            self._last_failure_time = datetime.now()
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit '{self.name}' -> OPEN (half-open failure)")
            else:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(f"Circuit '{self.name}' -> OPEN (threshold reached)")
