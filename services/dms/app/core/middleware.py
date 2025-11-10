"""Middleware for correlation ID propagation and logging."""

from __future__ import annotations

import logging
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Header names for correlation IDs
REQUEST_ID_HEADER = "X-Request-ID"
TRACE_ID_HEADER = "X-Trace-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to extract, generate, and propagate correlation IDs."""
    
    def __init__(self, app: ASGIApp, header_name: str = REQUEST_ID_HEADER):
        """Initialize correlation ID middleware.
        
        Args:
            app: ASGI application
            header_name: Header name to use for correlation ID (default: X-Request-ID)
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add correlation ID."""
        # Extract correlation ID from headers (try multiple header names)
        correlation_id = (
            request.headers.get(self.header_name) or
            request.headers.get(TRACE_ID_HEADER) or
            request.headers.get(CORRELATION_ID_HEADER)
        )
        
        # Generate new correlation ID if not present
        if not correlation_id:
            correlation_id = str(uuid.uuid4())
        
        # Add to request state for use in handlers
        request.state.correlation_id = correlation_id
        
        # Add to request context for use in background tasks
        request.scope["correlation_id"] = correlation_id
        
        # Log request with correlation ID
        logger.info(
            "[%s] %s %s",
            correlation_id,
            request.method,
            request.url.path,
            extra={"correlation_id": correlation_id},
        )
        
        # Process request
        response = await call_next(request)
        
        # Add correlation ID to response headers
        response.headers[self.header_name] = correlation_id
        
        # Log response with correlation ID
        logger.info(
            "[%s] %s %s -> %d",
            correlation_id,
            request.method,
            request.url.path,
            response.status_code,
            extra={"correlation_id": correlation_id, "status_code": response.status_code},
        )
        
        return response


def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Correlation ID string
    """
    return getattr(request.state, "correlation_id", "")


def get_correlation_id_from_context(context: dict) -> str:
    """Get correlation ID from context dictionary.
    
    Args:
        context: Context dictionary (e.g., from request.state or custom context)
    
    Returns:
        Correlation ID string, or empty string if not found
    """
    return context.get("correlation_id", "")

