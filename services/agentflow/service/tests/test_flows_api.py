"""Tests for AgentFlow flow router authentication and GPU handling."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
import httpx

sys.path.append(str(Path(__file__).resolve().parents[4]))

from services.agentflow.service.dependencies import (  # noqa: E402
    get_catalog,
    get_langflow_client,
    get_registry_service,
)
from services.agentflow.service.routers import flows as flows_module  # noqa: E402
from services.agentflow.service.tests.utils import create_async_client  # noqa: E402


router = flows_module.router


pytestmark = pytest.mark.anyio("asyncio")


class DummyCatalog:
    """Minimal catalog stub returning in-memory specs."""

    def __init__(self) -> None:
        self.specs: list[SimpleNamespace] = []

    def list(self) -> list[SimpleNamespace]:
        return list(self.specs)

    def get(self, flow_id: str) -> SimpleNamespace:
        for spec in self.specs:
            if spec.id == flow_id:
                return spec
        raise KeyError(flow_id)


class DummyRegistry:
    """Registry stub backed by a single mapping object."""

    def __init__(self) -> None:
        self.mapping: SimpleNamespace | None = None

    async def get(self, local_id: str) -> SimpleNamespace | None:
        return self.mapping


class DummyLangflow:
    """Langflow client stub that can optionally raise during execution."""

    def __init__(self) -> None:
        self.raise_exc: Exception | None = None
        self.response: dict[str, object] = {"result": "ok"}

    async def run_flow(self, flow_id: str, request) -> dict[str, object]:  # type: ignore[override]
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.response


@pytest.fixture
def app_with_stubs():
    """Return a FastAPI app wired with dependency overrides and test doubles."""

    app = FastAPI()
    app.include_router(router)

    catalog = DummyCatalog()
    registry = DummyRegistry()
    langflow_client = DummyLangflow()

    app.state.settings = SimpleNamespace(service_api_key="secret-key")

    def override_get_catalog(_: Request) -> DummyCatalog:
        return catalog

    async def override_get_registry(_: Request) -> DummyRegistry:
        return registry

    async def override_get_langflow(_: Request) -> DummyLangflow:
        return langflow_client

    app.dependency_overrides[get_catalog] = override_get_catalog
    app.dependency_overrides[get_registry_service] = override_get_registry
    app.dependency_overrides[get_langflow_client] = override_get_langflow

    return {
        "app": app,
        "catalog": catalog,
        "registry": registry,
        "langflow": langflow_client,
    }


async def test_list_flows_requires_api_key(app_with_stubs):
    app = app_with_stubs["app"]
    async with create_async_client(app) as client:
        response = await client.get("/flows")
    assert response.status_code == 401


async def test_list_flows_rejects_incorrect_api_key(app_with_stubs):
    app = app_with_stubs["app"]
    async with create_async_client(app) as client:
        response = await client.get("/flows", headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


async def test_list_flows_accepts_valid_api_key(app_with_stubs):
    app = app_with_stubs["app"]
    async with create_async_client(app) as client:
        response = await client.get("/flows", headers={"X-API-Key": "secret-key"})
    assert response.status_code == 200
    assert response.json() == []


async def test_run_flow_releases_gpu_allocation_on_error(app_with_stubs, monkeypatch):
    catalog: DummyCatalog = app_with_stubs["catalog"]
    registry: DummyRegistry = app_with_stubs["registry"]
    langflow: DummyLangflow = app_with_stubs["langflow"]
    app: FastAPI = app_with_stubs["app"]

    spec = SimpleNamespace(
        id="flow-1",
        raw={"nodes": [{"type": "LLM"}]},
        name="Test Flow",
        description="",
    )
    catalog.specs = [spec]
    registry.mapping = SimpleNamespace(
        local_id="flow-1",
        remote_id="remote-1",
        name="Test Flow",
        description="",
        project_id=None,
        folder_path=None,
        updated_at=None,
        synced_at=None,
    )

    release_calls: list[str] = []

    async def fake_request_gpu_allocation(*_, **__):
        return "alloc-123"

    async def fake_release_gpu_allocation(allocation_id: str) -> None:
        release_calls.append(allocation_id)

    monkeypatch.setattr(
        "services.agentflow.service.routers.flows._request_gpu_allocation",
        fake_request_gpu_allocation,
    )
    monkeypatch.setattr(
        "services.agentflow.service.routers.flows._release_gpu_allocation",
        fake_release_gpu_allocation,
    )

    langflow.raise_exc = RuntimeError("boom")

    async with create_async_client(app) as client:
        response = await client.post(
            "/flows/flow-1/run",
            json={},
            headers={"X-API-Key": "secret-key"},
        )

    assert response.status_code == 502
    assert release_calls == ["alloc-123"]


async def test_request_gpu_allocation_logs_failure(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="services.agentflow.service.routers.flows")

    class FailingClient:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *_, **__):
            raise httpx.ConnectTimeout("timeout", request=httpx.Request("POST", "http://test"))

    monkeypatch.setattr(
        "services.agentflow.service.routers.flows.httpx.AsyncClient",
        FailingClient,
    )

    result = await flows_module._request_gpu_allocation("flow-1")

    assert result is None
    assert any(
        "GPU allocation request failed" in message for message in caplog.messages
    )


async def test_request_gpu_allocation_logs_missing_url(caplog, monkeypatch):
    caplog.set_level(logging.DEBUG, logger="services.agentflow.service.routers.flows")
    monkeypatch.setenv("GPU_ORCHESTRATOR_URL", "")

    result = await flows_module._request_gpu_allocation("flow-2")

    assert result is None
    assert any(
        "GPU orchestrator URL missing; skipping allocation" in message
        for message in caplog.messages
    )
