"""HTTP client library with retry, circuit breaker, health checks, correlation IDs, and metrics."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, Callable, Optional
from urllib.parse import urljoin

import httpx

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class IntegrationError(Exception):
    """Base exception for integration errors."""
    pass


class ServiceUnavailableError(IntegrationError):
    """Service is unavailable."""
    pass


class TimeoutError(IntegrationError):
    """Request timeout."""
    pass


class ValidationError(IntegrationError):
    """Response validation error."""
    pass


class AuthenticationError(IntegrationError):
    """Authentication error."""
    pass


class CircuitBreaker:
    """Circuit breaker pattern implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
        on_state_change: Optional[Callable[[CircuitBreakerState], None]] = None,
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Seconds to wait before attempting half-open
            on_state_change: Callback for state changes
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.on_state_change = on_state_change
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
    
    async def call(self, func: Callable) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            # Check if we should attempt to reset
            if self.state == CircuitBreakerState.OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time) >= self.reset_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.failure_count = 0
                    self._notify_state_change()
                else:
                    raise ServiceUnavailableError("Circuit breaker is open")
        
        # Execute the function
        try:
            result = await func() if asyncio.iscoroutinefunction(func) else func()
            
            async with self._lock:
                # Success - reset if we were in half-open
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self._notify_state_change()
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0
                
                return result
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    if self.state != CircuitBreakerState.OPEN:
                        self.state = CircuitBreakerState.OPEN
                        self._notify_state_change()
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    self._notify_state_change()
            
            raise
    
    def _notify_state_change(self) -> None:
        """Notify about state change."""
        if self.on_state_change:
            try:
                self.on_state_change(self.state)
            except Exception as e:
                logger.warning("Error in circuit breaker state change callback: %s", e)


class ResilientHTTPClient:
    """HTTP client with retry, circuit breaker, health checks, and correlation IDs."""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        initial_backoff: float = 1.0,
        max_backoff: float = 10.0,
        circuit_breaker: Optional[CircuitBreaker] = None,
        health_check_path: str = "/healthz",
        health_check_cache_ttl: float = 30.0,
        correlation_id_header: str = "X-Request-ID",
        metrics_collector: Optional[Callable[[str, str, int, float, Optional[str]], None]] = None,
    ):
        """Initialize resilient HTTP client.
        
        Args:
            base_url: Base URL for the service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff delay in seconds
            max_backoff: Maximum backoff delay in seconds
            circuit_breaker: Circuit breaker instance (creates default if None)
            health_check_path: Path for health check endpoint
            health_check_cache_ttl: TTL for health check cache in seconds
            correlation_id_header: Header name for correlation ID
            metrics_collector: Callback for metrics collection (service, endpoint, status, latency, correlation_id)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.health_check_path = health_check_path
        self.health_check_cache_ttl = health_check_cache_ttl
        self.correlation_id_header = correlation_id_header
        self.metrics_collector = metrics_collector
        
        self._health_check_cache: Optional[tuple[bool, float]] = None
        self._client: Optional[httpx.AsyncClient] = None
    
    @asynccontextmanager
    async def _get_client(self):
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        try:
            yield self._client
        finally:
            pass  # Keep client alive for connection pooling
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _check_health(self) -> bool:
        """Check service health with caching."""
        now = time.time()
        
        # Check cache
        if self._health_check_cache:
            is_healthy, cached_time = self._health_check_cache
            if (now - cached_time) < self.health_check_cache_ttl:
                return is_healthy
        
        # Perform health check
        try:
            async with self._get_client() as client:
                health_url = urljoin(self.base_url, self.health_check_path)
                response = await client.get(health_url, timeout=5.0)
                is_healthy = response.status_code == 200
        except Exception as e:
            logger.warning("Health check failed for %s: %s", self.base_url, e)
            is_healthy = False
        
        # Update cache
        self._health_check_cache = (is_healthy, now)
        return is_healthy
    
    def _get_correlation_id(self, context: dict[str, Any]) -> str:
        """Extract or generate correlation ID from context."""
        # Try to get from context
        correlation_id = context.get("correlation_id")
        if correlation_id:
            return str(correlation_id)
        
        # Generate new one
        return str(uuid.uuid4())
    
    async def _retry_with_backoff(
        self,
        func: Callable,
        context: dict[str, Any],
        service_name: str,
        endpoint: str,
    ) -> Any:
        """Execute function with exponential backoff retry."""
        last_error: Optional[Exception] = None
        backoff = self.initial_backoff
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                logger.info(
                    "Retrying %s %s (attempt %d/%d, backoff %.2fs)",
                    service_name,
                    endpoint,
                    attempt + 1,
                    self.max_retries + 1,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
            
            try:
                start_time = time.time()
                result = await self.circuit_breaker.call(func)
                latency = time.time() - start_time
                
                correlation_id = self._get_correlation_id(context)
                if self.metrics_collector:
                    self.metrics_collector(service_name, endpoint, 200, latency, correlation_id)
                
                if attempt > 0:
                    logger.info(
                        "Request succeeded after %d attempts: %s %s",
                        attempt + 1,
                        service_name,
                        endpoint,
                    )
                
                return result
            except ServiceUnavailableError as e:
                # Circuit breaker is open
                correlation_id = self._get_correlation_id(context)
                if self.metrics_collector:
                    self.metrics_collector(service_name, endpoint, 503, 0.0, correlation_id)
                raise
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                last_error = e
                status_code = getattr(e, "response", None) and e.response.status_code or 500
                latency = time.time() - start_time if "start_time" in locals() else 0.0
                
                correlation_id = self._get_correlation_id(context)
                if self.metrics_collector:
                    self.metrics_collector(service_name, endpoint, status_code, latency, correlation_id)
                
                # Don't retry on 4xx errors (client errors)
                if 400 <= status_code < 500:
                    if status_code == 401 or status_code == 403:
                        raise AuthenticationError(f"Authentication failed: {e}") from e
                    raise ValidationError(f"Client error {status_code}: {e}") from e
                
                # Don't retry on timeout
                if isinstance(e, (httpx.TimeoutException, httpx.ConnectTimeout)):
                    raise TimeoutError(f"Request timeout: {e}") from e
                
                # Retry on 5xx errors or network errors
                if attempt == self.max_retries:
                    raise IntegrationError(f"Request failed after {self.max_retries + 1} attempts: {e}") from e
        
        raise IntegrationError(f"Request failed after {self.max_retries + 1} attempts") from last_error
    
    async def request(
        self,
        method: str,
        path: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make HTTP request with retry and circuit breaker.
        
        Args:
            method: HTTP method
            path: Request path (relative to base_url)
            context: Request context (for correlation ID, auth token, etc.)
            **kwargs: Additional arguments for httpx request
        
        Returns:
            HTTP response
        """
        if context is None:
            context = {}
        
        # Check health before request
        if not await self._check_health():
            raise ServiceUnavailableError(f"Service {self.base_url} is unhealthy")
        
        # Get correlation ID
        correlation_id = self._get_correlation_id(context)
        
        # Add correlation ID to headers
        headers = kwargs.get("headers", {})
        headers[self.correlation_id_header] = correlation_id
        kwargs["headers"] = headers
        
        # Add auth token if available
        auth_token = context.get("auth_token")
        if auth_token:
            if not auth_token.startswith("Bearer "):
                headers["Authorization"] = f"Bearer {auth_token}"
            else:
                headers["Authorization"] = auth_token
        
        # Build URL
        url = urljoin(self.base_url, path.lstrip("/"))
        
        # Extract service name and endpoint for metrics
        service_name = self.base_url.split("//")[-1].split("/")[0] if "//" in self.base_url else "unknown"
        endpoint = path
        
        async def _make_request() -> httpx.Response:
            async with self._get_client() as client:
                return await client.request(method, url, **kwargs)
        
        return await self._retry_with_backoff(_make_request, context, service_name, endpoint)
    
    async def get(
        self,
        path: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make GET request."""
        return await self.request("GET", path, context, **kwargs)
    
    async def post(
        self,
        path: str,
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make POST request."""
        return await self.request("POST", path, context, **kwargs)
    
    async def post_json(
        self,
        path: str,
        data: Any,
        context: Optional[dict[str, Any]] = None,
        validate_response: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Make POST request with JSON body and validation.
        
        Args:
            path: Request path
            data: Data to send as JSON
            context: Request context
            validate_response: Optional function to validate response structure
        
        Returns:
            Parsed JSON response
        
        Raises:
            ValidationError: If response validation fails
        """
        response = await self.post(path, context, json=data)
        response.raise_for_status()
        
        try:
            result = response.json()
        except Exception as e:
            raise ValidationError(f"Failed to parse JSON response: {e}") from e
        
        if validate_response:
            try:
                validate_response(result)
            except Exception as e:
                raise ValidationError(f"Response validation failed: {e}") from e
        
        return result
    
    async def get_json(
        self,
        path: str,
        context: Optional[dict[str, Any]] = None,
        validate_response: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        """Make GET request and parse JSON response.
        
        Args:
            path: Request path
            context: Request context
            validate_response: Optional function to validate response structure
        
        Returns:
            Parsed JSON response
        
        Raises:
            ValidationError: If response validation fails
        """
        response = await self.get(path, context)
        response.raise_for_status()
        
        try:
            result = response.json()
        except Exception as e:
            raise ValidationError(f"Failed to parse JSON response: {e}") from e
        
        if validate_response:
            try:
                validate_response(result)
            except Exception as e:
                raise ValidationError(f"Response validation failed: {e}") from e
        
        return result

