from __future__ import annotations

from secrets import compare_digest

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from .config import Settings
from .db import get_session
from .repositories import FlowRegistryRepository
from .services import FlowCatalog, FlowRegistryService
from .services.langflow import LangflowClient
from .services.localai import LocalAIClient


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_service_api_key(
    request: Request,
    api_key: str | None = Depends(_api_key_header),
) -> None:
    """Validate incoming requests when a service API key is configured."""

    settings = request.app.state.settings
    expected_key = settings.service_api_key

    if not expected_key:
        return

    provided_key = api_key
    if not provided_key:
        auth_header = request.headers.get("Authorization") or ""
        if auth_header.startswith("Bearer "):
            provided_key = auth_header.removeprefix("Bearer ")

    if not provided_key or not compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key",
        )


def get_catalog(request: Request) -> FlowCatalog:
    return request.app.state.catalog


def get_langflow_client(request: Request) -> LangflowClient:
    return request.app.state.langflow_client


def get_localai_client(request: Request) -> LocalAIClient:
    return request.app.state.localai_client


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_registry_service(
    request: Request,
    session=Depends(get_session),
) -> FlowRegistryService:
    repository = FlowRegistryRepository(session)
    redis_client = getattr(request.app.state, "redis", None)
    return FlowRegistryService(repository, redis_client)
