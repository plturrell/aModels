"""
LocalAI client integration for AgentFlow service.

Provides direct LocalAI client access for LLM nodes in flows,
with connection pooling and retry logic for improved performance.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class LocalAIError(Exception):
    """Raised when LocalAI returns a non-success response."""


class LocalAIClient:
    """
    Async HTTP client for LocalAI with connection pooling and retry logic.
    """

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        timeout_seconds: int = 60,
        max_retries: int = 3,
        pool_size: int = 10,
        max_idle: int = 5,
    ):
        self.base_url = base_url or os.getenv("LOCALAI_URL", "http://localhost:8080")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        # Get pool configuration from environment
        pool_size = int(os.getenv("AGENTFLOW_LOCALAI_POOL_SIZE", str(pool_size)))
        max_idle = int(os.getenv("AGENTFLOW_LOCALAI_MAX_IDLE", str(max_idle)))

        # Create HTTP client with connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=pool_size,
            max_connections=pool_size * 2,
        )

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            limits=limits,
        )

        logger.info(
            f"LocalAI client initialized: {self.base_url} "
            f"(pool_size={pool_size}, max_idle={max_idle})"
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def health_check(self) -> bool:
        """Check if LocalAI service is available."""
        try:
            response = await self._client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"LocalAI health check failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from LocalAI."""
        payload = await self._request("GET", "/v1/models")
        if isinstance(payload, dict):
            models = payload.get("data", [])
            if isinstance(models, list):
                return models
        return []

    async def chat_completion(
        self,
        *,
        model: str,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        Perform a chat completion request to LocalAI.

        Args:
            model: Model name to use
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream the response

        Returns:
            Chat completion response
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if stop is not None:
            payload["stop"] = stop
        if stream:
            payload["stream"] = stream

        return await self._request("POST", "/v1/chat/completions", json=payload)

    async def get_content(self, response: Dict[str, Any]) -> str:
        """Extract content from a chat completion response."""
        choices = response.get("choices", [])
        if choices and len(choices) > 0:
            message = choices[0].get("message", {})
            return message.get("content", "")
        return ""

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Make an HTTP request with retry logic."""
        initial_backoff = 0.1
        max_backoff = 2.0
        backoff = initial_backoff

        last_error = None
        for attempt in range(self.max_retries):
            if attempt > 0:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                logger.info(
                    f"Retrying LocalAI request (attempt {attempt + 1}/{self.max_retries})"
                )

            try:
                response = await self._client.request(method, path, json=json)
                if response.status_code >= 400:
                    detail = response.text
                    try:
                        data = response.json()
                        detail = data.get("detail") or data.get("message") or detail
                    except ValueError:
                        pass

                    # Check if error is retryable
                    if self._is_retryable_error(response.status_code, detail):
                        last_error = LocalAIError(
                            f"LocalAI request failed ({response.status_code}): {detail}"
                        )
                        continue

                    raise LocalAIError(
                        f"LocalAI request failed ({response.status_code}): {detail}"
                    )

                if response.status_code == 204:
                    return None

                return response.json()
            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                raise LocalAIError(f"LocalAI request timeout: {e}") from e
            except httpx.NetworkError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    continue
                raise LocalAIError(f"LocalAI network error: {e}") from e
            except Exception as e:
                raise LocalAIError(f"LocalAI request error: {e}") from e

        if last_error:
            raise last_error
        raise LocalAIError("LocalAI request failed after retries")

    def _is_retryable_error(self, status_code: int, detail: str) -> bool:
        """Check if an error is retryable."""
        # Retry on 5xx errors and network issues
        if status_code >= 500:
            return True

        detail_lower = str(detail).lower()
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "unavailable",
            "service",
        ]
        return any(keyword in detail_lower for keyword in retryable_keywords)


# Global client instance
_localai_client: Optional[LocalAIClient] = None


def get_localai_client() -> LocalAIClient:
    """Get or create the global LocalAI client."""
    global _localai_client
    if _localai_client is None:
        max_retries = int(os.getenv("AGENTFLOW_LOCALAI_RETRY_MAX_ATTEMPTS", "3"))
        _localai_client = LocalAIClient(max_retries=max_retries)
    return _localai_client


async def close_localai_client() -> None:
    """Close the global LocalAI client."""
    global _localai_client
    if _localai_client is not None:
        await _localai_client.close()
        _localai_client = None

