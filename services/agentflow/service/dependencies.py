from __future__ import annotations

from fastapi import Depends, Request

from .config import Settings
from .db import get_session
from .repositories import FlowRegistryRepository
from .services import FlowCatalog, FlowRegistryService
from .services.langflow import LangflowClient


def get_catalog(request: Request) -> FlowCatalog:
    return request.app.state.catalog


def get_langflow_client(request: Request) -> LangflowClient:
    return request.app.state.langflow_client


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


async def get_registry_service(
    request: Request,
    session=Depends(get_session),
) -> FlowRegistryService:
    repository = FlowRegistryRepository(session)
    redis_client = getattr(request.app.state, "redis", None)
    return FlowRegistryService(repository, redis_client)
