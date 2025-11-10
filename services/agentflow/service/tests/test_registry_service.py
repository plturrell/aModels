from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytest


sys.path.append(str(Path(__file__).resolve().parents[4]))

from services.agentflow.service.models.flow import FlowMapping  # noqa: E402
from services.agentflow.service.services.registry_service import (  # noqa: E402
    FlowRegistryService,
    RedisError,
)


class DummyRepository:
    def __init__(self) -> None:
        self.list_calls = 0
        self.get_calls = 0
        self.upsert_calls = []
        self.records: dict[str, FlowMapping] = {}

    def list(self):
        self.list_calls += 1
        return list(self.records.values())

    def get(self, local_id: str) -> Optional[FlowMapping]:
        self.get_calls += 1
        return self.records.get(local_id)

    def upsert(self, **kwargs):  # type: ignore[override]
        self.upsert_calls.append(kwargs)
        mapping = FlowMapping(
            local_id=kwargs["local_id"],
            remote_id=kwargs.get("remote_id"),
            name=kwargs.get("name"),
            description=kwargs.get("description"),
            project_id=kwargs.get("project_id"),
            folder_path=kwargs.get("folder_path"),
            updated_at=kwargs.get("updated_at"),
        )
        mapping.synced_at = datetime.utcnow()
        self.records[mapping.local_id] = mapping
        return mapping


class DummyRedis:
    def __init__(self) -> None:
        self.storage: dict[str, str] = {}
        self.fail_read = False
        self.fail_write = False

    async def set(self, key: str, value: str, ex: int = 0):
        if self.fail_write:
            raise RedisError("write failure")
        self.storage[key] = value

    async def get(self, key: str) -> Optional[str]:
        if self.fail_read:
            raise RedisError("read failure")
        return self.storage.get(key)


@pytest.mark.anyio
async def test_registry_logs_cache_hit(caplog):
    caplog.set_level(logging.DEBUG)
    repo = DummyRepository()
    redis = DummyRedis()
    test_logger = logging.getLogger("registry-tests-cache-hit")
    test_logger.setLevel(logging.DEBUG)
    service = FlowRegistryService(repo, redis_client=redis, logger=test_logger)

    mapping = FlowMapping(local_id="flow-1", remote_id="remote-1")
    mapping.synced_at = datetime.utcnow()
    await service._cache_mapping(mapping)

    result = await service.get("flow-1")
    assert isinstance(result, FlowMapping)
    assert result.local_id == "flow-1"
    assert any("flow mapping served from cache" in message for message in caplog.messages)


@pytest.mark.anyio
async def test_registry_logs_redis_errors(caplog):
    caplog.set_level(logging.DEBUG)
    repo = DummyRepository()
    redis = DummyRedis()
    redis.fail_read = True
    redis.fail_write = True
    test_logger = logging.getLogger("registry-tests-redis-errors")
    test_logger.setLevel(logging.DEBUG)
    service = FlowRegistryService(repo, redis_client=redis, logger=test_logger)

    mapping = FlowMapping(local_id="flow-2", remote_id="remote-2")
    mapping.synced_at = datetime.utcnow()

    await service._cache_mapping(mapping)
    await service.get("flow-2")

    assert any("failed to write flow mapping to redis" in message for message in caplog.messages)
    assert any("failed to read flow mapping from redis" in message for message in caplog.messages)
