"""
Generic proxy utilities for DRY service proxying.
Eliminates code duplication across 91 endpoints.
"""
import json
from typing import Any, Dict, Optional, Callable
import httpx
from fastapi import HTTPException

from circuit_breaker import CircuitBreakerManager, CircuitBreakerOpenError, CircuitBreakerConfig
from logging_config import get_logger

logger = get_logger(__name__)


class ServiceProxy:
    """
    Generic service proxy with circuit breaker, caching, and retry logic.
    Handles common patterns: GET, POST, PUT, DELETE to backend services.
    """
    
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        circuit_breaker_manager: Optional[CircuitBreakerManager] = None,
        cache_client: Optional[Any] = None
    ):
        self.client = http_client
        self.cb_manager = circuit_breaker_manager
        self.cache = cache_client
        
    async def _call_with_circuit_breaker(
        self,
        service_name: str,
        func: Callable,
        timeout: float = 30.0
    ) -> httpx.Response:
        """Execute function with circuit breaker protection."""
        if self.cb_manager is None:
            return await func()
        
        breaker = self.cb_manager.get_breaker(
            service_name,
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                timeout=60.0
            )
        )
        
        try:
            result = await breaker.call(func)
            return result
        except CircuitBreakerOpenError as e:
            logger.error(
                f"Circuit breaker open for {service_name}",
                service=service_name,
                error=str(e)
            )
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "Service temporarily unavailable",
                    "service": service_name,
                    "message": str(e),
                    "retry_after": 60
                }
            )
    
    async def proxy_get(
        self,
        service_name: str,
        service_url: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        cache_ttl: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Generic GET proxy with caching support.
        
        Args:
            service_name: Name of the service (for circuit breaker and logging)
            service_url: Base URL of the service
            path: Path to append to service URL
            params: Query parameters
            timeout: Request timeout
            cache_ttl: Cache TTL in seconds (None = no caching)
            headers: Additional headers
            
        Returns:
            Response JSON
            
        Raises:
            HTTPException: On error
        """
        full_url = f"{service_url.rstrip('/')}/{path.lstrip('/')}"
        
        # Check cache if enabled
        if cache_ttl and self.cache:
            cache_key = self._make_cache_key("GET", full_url, params)
            cached = await self._get_from_cache(cache_key)
            if cached:
                logger.debug(f"Cache hit for {service_name}", url=full_url)
                return cached
        
        try:
            async def make_request():
                return await self.client.get(
                    full_url,
                    params=params,
                    timeout=timeout,
                    headers=headers
                )
            
            response = await self._call_with_circuit_breaker(
                service_name,
                make_request,
                timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Cache if enabled
            if cache_ttl and self.cache:
                await self._set_in_cache(cache_key, result, cache_ttl)
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"{service_name} HTTP error",
                service=service_name,
                status_code=e.response.status_code,
                url=full_url
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )
        except httpx.RequestError as e:
            logger.error(
                f"{service_name} request error",
                service=service_name,
                error=str(e),
                url=full_url
            )
            raise HTTPException(
                status_code=502,
                detail=f"{service_name} service error: {e}"
            )
    
    async def proxy_post(
        self,
        service_name: str,
        service_url: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Generic POST proxy.
        
        Args:
            service_name: Name of the service
            service_url: Base URL of the service
            path: Path to append to service URL
            payload: JSON payload
            params: Query parameters
            timeout: Request timeout
            headers: Additional headers
            
        Returns:
            Response JSON
            
        Raises:
            HTTPException: On error
        """
        full_url = f"{service_url.rstrip('/')}/{path.lstrip('/')}"
        
        try:
            async def make_request():
                return await self.client.post(
                    full_url,
                    json=payload,
                    params=params,
                    timeout=timeout,
                    headers=headers
                )
            
            response = await self._call_with_circuit_breaker(
                service_name,
                make_request,
                timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"{service_name} HTTP error",
                service=service_name,
                status_code=e.response.status_code,
                url=full_url
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )
        except httpx.RequestError as e:
            logger.error(
                f"{service_name} request error",
                service=service_name,
                error=str(e),
                url=full_url
            )
            raise HTTPException(
                status_code=502,
                detail=f"{service_name} service error: {e}"
            )
    
    async def proxy_delete(
        self,
        service_name: str,
        service_url: str,
        path: str,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Generic DELETE proxy.
        
        Args:
            service_name: Name of the service
            service_url: Base URL of the service
            path: Path to append to service URL
            timeout: Request timeout
            headers: Additional headers
            
        Returns:
            Response JSON
            
        Raises:
            HTTPException: On error
        """
        full_url = f"{service_url.rstrip('/')}/{path.lstrip('/')}"
        
        try:
            async def make_request():
                return await self.client.delete(
                    full_url,
                    timeout=timeout,
                    headers=headers
                )
            
            response = await self._call_with_circuit_breaker(
                service_name,
                make_request,
                timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"{service_name} HTTP error",
                service=service_name,
                status_code=e.response.status_code,
                url=full_url
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )
        except httpx.RequestError as e:
            logger.error(
                f"{service_name} request error",
                service=service_name,
                error=str(e),
                url=full_url
            )
            raise HTTPException(
                status_code=502,
                detail=f"{service_name} service error: {e}"
            )
    
    async def proxy_put(
        self,
        service_name: str,
        service_url: str,
        path: str,
        payload: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
        headers: Optional[Dict[str, str]] = None
    ) -> Any:
        """
        Generic PUT proxy.
        
        Args:
            service_name: Name of the service
            service_url: Base URL of the service
            path: Path to append to service URL
            payload: JSON payload
            timeout: Request timeout
            headers: Additional headers
            
        Returns:
            Response JSON
            
        Raises:
            HTTPException: On error
        """
        full_url = f"{service_url.rstrip('/')}/{path.lstrip('/')}"
        
        try:
            async def make_request():
                return await self.client.put(
                    full_url,
                    json=payload,
                    timeout=timeout,
                    headers=headers
                )
            
            response = await self._call_with_circuit_breaker(
                service_name,
                make_request,
                timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            logger.error(
                f"{service_name} HTTP error",
                service=service_name,
                status_code=e.response.status_code,
                url=full_url
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )
        except httpx.RequestError as e:
            logger.error(
                f"{service_name} request error",
                service=service_name,
                error=str(e),
                url=full_url
            )
            raise HTTPException(
                status_code=502,
                detail=f"{service_name} service error: {e}"
            )
    
    def _make_cache_key(self, method: str, url: str, params: Optional[Dict] = None) -> str:
        """Generate cache key from request parameters."""
        key_parts = [method, url]
        if params:
            # Sort params for consistent keys
            sorted_params = sorted(params.items())
            key_parts.append(str(sorted_params))
        return ":".join(key_parts)
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.cache:
            return None
        try:
            cached_str = await self.cache.get(key)
            if cached_str:
                return json.loads(cached_str)
        except Exception as e:
            logger.warning(f"Cache get error", key=key, error=str(e))
        return None
    
    async def _set_in_cache(self, key: str, value: Any, ttl: int):
        """Set value in cache with TTL."""
        if not self.cache:
            return
        try:
            value_str = json.dumps(value)
            await self.cache.set(key, value_str, ex=ttl)
            logger.debug(f"Cache set", key=key, ttl=ttl)
        except Exception as e:
            logger.warning(f"Cache set error", key=key, error=str(e))
    
    async def invalidate_cache(self, pattern: str):
        """Invalidate cache keys matching pattern."""
        if not self.cache:
            return
        try:
            # Scan for keys matching pattern
            cursor = 0
            deleted = 0
            while True:
                cursor, keys = await self.cache.scan(cursor, match=pattern, count=100)
                if keys:
                    await self.cache.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break
            logger.info(f"Cache invalidated", pattern=pattern, deleted_keys=deleted)
        except Exception as e:
            logger.error(f"Cache invalidation error", pattern=pattern, error=str(e))


def create_proxy_helper(
    client: httpx.AsyncClient,
    circuit_breaker_manager: Optional[CircuitBreakerManager],
    redis_client: Optional[Any]
) -> ServiceProxy:
    """Factory function to create ServiceProxy instance."""
    return ServiceProxy(client, circuit_breaker_manager, redis_client)
