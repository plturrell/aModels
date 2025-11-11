"""Authentication middleware for token extraction and forwarding."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Security configuration from environment
JWT_SECRET_KEY = os.getenv("DMS_JWT_SECRET", "dev-secret-change-in-production")
JWT_ALGORITHM = os.getenv("DMS_JWT_ALGORITHM", "HS256")
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set(os.getenv("DMS_API_KEYS", "").split(",")) if os.getenv("DMS_API_KEYS") else set()

# HTTPBearer security scheme for OpenAPI docs
security = HTTPBearer(auto_error=False)


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


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time (default: 7 days)
    
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(days=7))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_jwt_token(token: str) -> Optional[dict]:
    """Verify and decode JWT token.
    
    Args:
        token: JWT token string
    
    Returns:
        Decoded token payload or None if invalid
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("Invalid JWT token: %s", e)
        return None


def verify_api_key(api_key: str) -> bool:
    """Verify API key.
    
    Args:
        api_key: API key string
    
    Returns:
        True if valid, False otherwise
    """
    if not VALID_API_KEYS:
        # If no API keys configured, allow all (dev mode)
        logger.warning("No API keys configured - allowing all requests")
        return True
    return api_key in VALID_API_KEYS


async def verify_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Dependency to verify JWT token or API key.
    
    Args:
        request: FastAPI request
        credentials: HTTP authorization credentials
    
    Returns:
        Token payload with user information
    
    Raises:
        HTTPException: If authentication fails
    """
    # Check API key first
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key and verify_api_key(api_key):
        return {"auth_type": "api_key", "api_key": api_key}
    
    # Check JWT token
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"auth_type": "jwt", **payload}


async def optional_verify_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Optional dependency to verify token but don't fail if missing.
    
    Args:
        request: FastAPI request
        credentials: HTTP authorization credentials
    
    Returns:
        Token payload or None if not authenticated
    """
    # Check API key
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key and verify_api_key(api_key):
        return {"auth_type": "api_key", "api_key": api_key}
    
    # Check JWT token
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    if payload is None:
        return None
    
    return {"auth_type": "jwt", **payload}

