from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi.concurrency import run_in_threadpool

from ..config import get_settings
from ..db.hana import upsert_hana_record
from ..models.flow import FlowMapping
from ..repositories import FlowRegistryRepository
from .catalog import FlowSpec
from .langflow import FlowRecord

try:
    from redis.asyncio import Redis  # type: ignore
    from redis.exceptions import RedisError  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[assignment]
    RedisError = Exception  # type: ignore[assignment]


class FlowRegistryService:
    """
    Coordinates persistence across SQLite, HANA, and optional Redis caches.
    """

    def __init__(self, repository: FlowRegistryRepository, redis_client: Optional["Redis"] = None):
        self._repo = repository
        self._redis = redis_client
        self._settings = get_settings()

    async def list(self) -> list[FlowMapping]:
        return await run_in_threadpool(lambda: list(self._repo.list()))

    async def get(self, local_id: str) -> Optional[FlowMapping]:
        cached = await self._get_cached(local_id)
        if cached:
            return cached
        record = await run_in_threadpool(lambda: self._repo.get(local_id))
        if record:
            await self._cache_mapping(record)
        return record

    async def upsert_from_flow(
        self,
        *,
        spec: FlowSpec,
        remote: FlowRecord,
    ) -> FlowMapping:
        def persist() -> FlowMapping:
            try:
                relative_parent = spec.path.relative_to(self._settings.flows_dir).parent
                folder_path = str(relative_parent) if str(relative_parent) != "." else None
            except ValueError:
                folder_path = None
            return self._repo.upsert(
                local_id=spec.id,
                remote_id=remote.id or spec.id,
                name=remote.name or spec.name,
                description=remote.description or spec.description,
                project_id=remote.project_id,
                folder_path=folder_path,
                updated_at=remote.updated_at,
                metadata=spec.metadata,
                raw_definition=spec.raw,
            )

        mapping = await run_in_threadpool(persist)

        # Mirror to HANA when configured.
        await run_in_threadpool(
            upsert_hana_record,
            mapping.local_id,
            mapping.remote_id,
            mapping.name,
            mapping.description,
            mapping.project_id,
            mapping.updated_at,
            mapping.synced_at,
        )

        await self._cache_mapping(mapping)

        return mapping

    async def _cache_mapping(self, mapping: FlowMapping) -> None:
        if self._redis is None:
            return
        payload = json.dumps(
            {
                "local_id": mapping.local_id,
                "remote_id": mapping.remote_id,
                "name": mapping.name,
                "description": mapping.description,
                "project_id": mapping.project_id,
                "folder_path": mapping.folder_path,
                "updated_at": mapping.updated_at.isoformat() if mapping.updated_at else None,
                "synced_at": mapping.synced_at.isoformat() if mapping.synced_at else None,
            }
        )
        try:
            await self._redis.set(self._redis_key(mapping.local_id), payload, ex=3600)
        except RedisError:
            # Redis unavailable – treat as cache miss and continue.
            return

    async def _get_cached(self, local_id: str) -> Optional[FlowMapping]:
        if self._redis is None:
            return None
        try:
            payload = await self._redis.get(self._redis_key(local_id))
        except RedisError:
            return None
        if not payload:
            return None
        data = json.loads(payload)
        return FlowMapping(
            local_id=data.get("local_id", local_id),
            remote_id=data.get("remote_id"),
            name=data.get("name"),
            description=data.get("description"),
            project_id=data.get("project_id"),
            folder_path=data.get("folder_path"),
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None,
            synced_at=datetime.fromisoformat(data["synced_at"]) if data.get("synced_at") else None,
        )

    def _redis_key(self, local_id: str) -> str:
        namespace = self._settings.redis_namespace or "agentflow"
        return f"{namespace}:flow:{local_id}"
