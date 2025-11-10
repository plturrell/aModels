"""Tests for DeepAgents health check caching logic."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from services.agentflow.service import deepagents


class StubResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class StubClient:
    def __init__(self, responses: Iterable[object]):
        self._responses = list(responses)
        self.calls = 0

    async def get(self, path: str, timeout: float):  # noqa: ARG002 - signature matches httpx.AsyncClient
        self.calls += 1
        if not self._responses:
            raise AssertionError("Unexpected health check call")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


@pytest.mark.anyio
async def test_health_check_success_sets_cache(monkeypatch):
    client = StubClient([StubResponse(200)])

    monkeypatch.setattr(deepagents, "_deepagents_health_status", {"ok": False, "checked_at": 0.0})
    monkeypatch.setattr(deepagents, "_health_check_lock", asyncio.Lock())
    monkeypatch.setattr(deepagents, "_health_check_ttl_seconds", 60.0)

    times = iter([100.0, 101.0])
    monkeypatch.setattr(deepagents.time, "monotonic", lambda: next(times))

    result = await deepagents._deepagents_is_healthy(client)

    assert result is True
    assert client.calls == 1
    assert deepagents._deepagents_health_status["ok"] is True
    assert deepagents._deepagents_health_status["checked_at"] == 101.0


@pytest.mark.anyio
async def test_health_check_uses_cache_within_ttl(monkeypatch):
    client = StubClient([StubResponse(200)])

    monkeypatch.setattr(deepagents, "_deepagents_health_status", {"ok": False, "checked_at": 0.0})
    monkeypatch.setattr(deepagents, "_health_check_lock", asyncio.Lock())
    monkeypatch.setattr(deepagents, "_health_check_ttl_seconds", 30.0)

    times = iter([100.0, 101.0, 110.0])
    monkeypatch.setattr(deepagents.time, "monotonic", lambda: next(times))

    first = await deepagents._deepagents_is_healthy(client)
    second = await deepagents._deepagents_is_healthy(client)

    assert first is True
    assert second is True
    assert client.calls == 1
    assert deepagents._deepagents_health_status["checked_at"] == 101.0
