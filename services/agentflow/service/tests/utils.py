from __future__ import annotations

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient


def create_async_client(app: FastAPI) -> AsyncClient:
    """Return an AsyncClient bound to the FastAPI app with consistent settings."""
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    return AsyncClient(transport=transport, base_url="http://testserver")
