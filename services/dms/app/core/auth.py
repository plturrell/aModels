"""Authentication middleware for token extraction and forwarding."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and forward authentication tokens."""
    
    def __init__(self, app: ASGIApp, require_auth: bool = False):
        """Initialize auth middleware.
        
        Args:
            app: ASGI application
            require_auth: Whether authentication is required (default: False for optional auth)
        """
        super().__init__(app)
        self.require_auth = require_auth
    
    async def dispatch(self, request: Request, call_next):
        """Process request and extract auth token."""
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization", "")
        token = None
        
        if auth_header:
            # Support both "Bearer <token>" and direct token
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
            else:
                token = auth_header
        
        # Add to request state for use in handlers
        request.state.auth_token = token
        
        # Add to request context for use in background tasks
        if token:
            request.scope["auth_token"] = token
        
        # If auth is required and token is missing, return 401
        if self.require_auth and not token:
            from fastapi import Response
            return Response(
                content='{"error": "Authorization header required"}',
                status_code=401,
                media_type="application/json",
            )
        
        # Process request
        response = await call_next(request)
        return response


def get_auth_token(request: Request) -> Optional[str]:
    """Get auth token from request.
    
    Args:
        request: FastAPI request object
    
    Returns:
        Auth token string, or None if not present
    """
    return getattr(request.state, "auth_token", None)


def get_auth_token_from_context(context: dict) -> Optional[str]:
    """Get auth token from context dictionary.
    
    Args:
        context: Context dictionary
    
    Returns:
        Auth token string, or None if not found
    """
    return context.get("auth_token")

