"""
Retry utilities for handling transient connection errors.
"""
import asyncio
import logging
from typing import Callable, TypeVar, Optional
import httpx

logger = logging.getLogger(__name__)

T = TypeVar('T')


async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (httpx.ConnectError, httpx.TimeoutException),
) -> T:
    """
    Retry a function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
        retryable_exceptions: Tuple of exceptions that should trigger retry
    
    Returns:
        Result of the function call
    
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    delay = initial_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e
            if attempt < max_retries:
                logger.warning(
                    f"Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s: {e}"
                )
                await asyncio.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                logger.error(f"All {max_retries + 1} retry attempts failed")
        except Exception as e:
            # Non-retryable exception, re-raise immediately
            raise
    
    # All retries exhausted
    raise last_exception


async def retry_http_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    **kwargs
) -> httpx.Response:
    """
    Retry an HTTP request with exponential backoff.
    
    Args:
        client: httpx AsyncClient instance
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        **kwargs: Additional arguments to pass to client.request
    
    Returns:
        httpx.Response object
    
    Raises:
        httpx.HTTPError if all retries fail
    """
    async def make_request():
        response = await client.request(method, url, **kwargs)
        # Only retry on connection errors, not HTTP errors
        # HTTP errors (4xx, 5xx) should not be retried
        return response
    
    return await retry_with_backoff(
        make_request,
        max_retries=max_retries,
        initial_delay=initial_delay,
        retryable_exceptions=(httpx.ConnectError, httpx.TimeoutException),
    )

