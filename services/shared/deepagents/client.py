"""Shared Python client library for DeepAgents service."""

import os
import time
import logging
from typing import Optional, Dict, Any, List
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)


class DeepAgentsClient:
    """Standardized HTTP client for DeepAgents service with retry and circuit breaker."""
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        initial_backoff: float = 1.0,
        max_backoff: float = 5.0,
        enabled: bool = True,
    ):
        """Initialize DeepAgents client.
        
        Args:
            base_url: Base URL for DeepAgents service (defaults to env var or default)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff delay in seconds
            max_backoff: Maximum backoff delay in seconds
            enabled: Whether client is enabled (can be disabled for testing)
        """
        self.base_url = base_url or os.getenv("DEEPAGENTS_URL", "http://deepagents-service:9004")
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.enabled = enabled
        self.client = httpx.Client(timeout=timeout)
    
    def invoke(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        config: Optional[Dict[str, Any]] = None,
        response_format: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invoke the agent with a conversation.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream responses
            config: Optional configuration dict
            response_format: Optional response format (string or dict)
        
        Returns:
            Response dict with messages and result, or None if disabled/unavailable
        """
        if not self.enabled:
            return None
        
        # Quick health check
        if not self._check_health():
            logger.warning("DeepAgents service unavailable, skipping invocation")
            return None
        
        request = {
            "messages": messages,
            "stream": stream,
        }
        if config:
            request["config"] = config
        if response_format:
            request["response_format"] = response_format
        
        endpoint = f"{self.base_url}/invoke"
        return self._call_with_retry(endpoint, request)
    
    def invoke_structured(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
        stream: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Invoke the agent with structured output.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            response_format: Response format dict with 'type' and 'json_schema'
            stream: Whether to stream responses
            config: Optional configuration dict
        
        Returns:
            Response dict with structured_output, or None if disabled/unavailable
        """
        if not self.enabled:
            return None
        
        # Quick health check
        if not self._check_health():
            logger.warning("DeepAgents service unavailable, skipping structured invocation")
            return None
        
        request = {
            "messages": messages,
            "response_format": response_format,
            "stream": stream,
        }
        if config:
            request["config"] = config
        
        endpoint = f"{self.base_url}/invoke/structured"
        return self._call_with_retry(endpoint, request)
    
    def _call_with_retry(
        self,
        endpoint: str,
        request: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Perform HTTP request with exponential backoff retry."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                backoff = self._calculate_backoff(attempt)
                logger.info(f"Retrying DeepAgents request (attempt {attempt + 1}/{self.max_retries + 1}) after {backoff}s")
                time.sleep(backoff)
            
            try:
                resp = self.client.post(endpoint, json=request)
                resp.raise_for_status()
                return resp.json()
            
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code >= 500 and attempt < self.max_retries:
                    continue  # Retry on server errors
                logger.warning(f"DeepAgents returned status {e.response.status_code}: {e.response.text}")
                return None  # Non-fatal
            
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    continue  # Retry on network/timeout errors
                logger.warning(f"DeepAgents request failed after {attempt + 1} attempts: {e}")
                return None  # Non-fatal
            
            except Exception as e:
                last_error = str(e)
                logger.warning(f"DeepAgents request failed (non-fatal): {e}")
                return None
        
        logger.warning(f"DeepAgents request failed after all retries: {last_error}")
        return None
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff duration."""
        backoff = self.initial_backoff * (2 ** attempt)
        return min(backoff, self.max_backoff)
    
    def _check_health(self) -> bool:
        """Perform a quick health check."""
        try:
            health_client = httpx.Client(timeout=5.0)
            resp = health_client.get(f"{self.base_url}/healthz")
            return resp.status_code == 200
        except Exception:
            return False
    
    def close(self):
        """Close the HTTP client."""
        self.client.close()


def create_client(
    base_url: Optional[str] = None,
    timeout: float = 120.0,
    max_retries: int = 2,
    enabled: Optional[bool] = None,
) -> DeepAgentsClient:
    """Create a DeepAgents client with default configuration.
    
    Args:
        base_url: Optional base URL (defaults to DEEPAGENTS_URL env var or default)
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        enabled: Whether client is enabled (defaults to True)
    
    Returns:
        Configured DeepAgentsClient instance
    """
    if enabled is None:
        enabled = os.getenv("DEEPAGENTS_ENABLED", "true").lower() != "false"
    
    return DeepAgentsClient(
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        enabled=enabled,
    )

