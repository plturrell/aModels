from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import Form
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    description: Optional[str] = Field(default=None)
    content_type: Optional[str] = Field(default=None)


class DocumentCreate(BaseModel):
    name: str
    description: Optional[str] = Field(default=None)
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def as_form(
        cls,
        name: str = Form(...),
        description: Optional[str] = Form(None),
        tags: str = Form("")
    ) -> "DocumentCreate":
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        return cls(name=name, description=description, tags=tag_list)


class DocumentRead(BaseModel):
    id: str
    name: str
    description: Optional[str]
    storage_path: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
