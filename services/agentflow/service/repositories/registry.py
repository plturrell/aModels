from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable, Optional

from sqlmodel import Session, select

from ..models.flow import FlowMapping


class FlowRegistryRepository:
    """
    CRUD helpers for the flow registry persistence model.
    """

    def __init__(self, session: Session):
        self._session = session

    def list(self) -> Iterable[FlowMapping]:
        statement = select(FlowMapping).order_by(FlowMapping.local_id)
        return self._session.exec(statement).all()

    def get(self, local_id: str) -> Optional[FlowMapping]:
        if not local_id:
            return None
        return self._session.get(FlowMapping, local_id)

    def upsert(
        self,
        *,
        local_id: str,
        remote_id: Optional[str],
        name: Optional[str],
        description: Optional[str],
        project_id: Optional[str],
        folder_path: Optional[str],
        updated_at: Optional[datetime],
        metadata: Optional[dict],
        raw_definition: Optional[dict],
    ) -> FlowMapping:
        record = self._session.get(FlowMapping, local_id)
        if record is None:
            record = FlowMapping(local_id=local_id)
            self._session.add(record)

        if remote_id is not None:
            record.remote_id = remote_id
        if name is not None:
            record.name = name
        if description is not None:
            record.description = description
        if project_id is not None:
            record.project_id = project_id
        if folder_path is not None:
            record.folder_path = folder_path
        if updated_at is not None:
            record.updated_at = updated_at
        record.synced_at = datetime.utcnow()

        if metadata is not None:
            record.metadata_json = json.dumps(metadata)
        if raw_definition is not None:
            record.raw_definition = json.dumps(raw_definition)

        self._session.commit()
        self._session.refresh(record)
        return record

    def delete(self, local_id: str) -> None:
        record = self._session.get(FlowMapping, local_id)
        if record is None:
            return
        self._session.delete(record)
        self._session.commit()
