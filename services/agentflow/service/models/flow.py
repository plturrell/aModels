from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


class FlowMapping(SQLModel, table=True):
    """
    Persistence model mapping local catalog identifiers to Langflow remote IDs.
    """

    __tablename__ = "flow_registry"

    local_id: str = Field(primary_key=True, index=True, max_length=255)
    remote_id: Optional[str] = Field(default=None, index=True, max_length=255)
    name: Optional[str] = Field(default=None, max_length=255)
    description: Optional[str] = Field(default=None, max_length=1024)
    project_id: Optional[str] = Field(default=None, max_length=255)
    folder_path: Optional[str] = Field(default=None, max_length=255)
    updated_at: Optional[datetime] = Field(default=None)
    synced_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text, nullable=True))
    raw_definition: Optional[str] = Field(
        default=None,
        sa_column=Column(Text, nullable=True),
    )

    class Config:
        arbitrary_types_allowed = True
