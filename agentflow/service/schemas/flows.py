from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FlowInfo(BaseModel):
    local_id: str = Field(..., description="Local catalog identifier")
    remote_id: Optional[str] = Field(None, description="Langflow identifier")
    name: Optional[str] = None
    description: Optional[str] = None
    project_id: Optional[str] = None
    folder_path: Optional[str] = None
    updated_at: Optional[datetime] = None
    synced_at: Optional[datetime] = None


class FlowSyncRequest(BaseModel):
    force: bool = Field(default=True, description="Force overwrite when importing the flow")
    project_id: Optional[str] = Field(default=None, description="Langflow project identifier override")
    folder_path: Optional[str] = Field(default=None, description="Target folder path in Langflow")


class FlowSyncResponse(FlowInfo):
    status: str = Field("synced", description="Outcome of the synchronisation request")


class FlowRunRequestSchema(BaseModel):
    input_value: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    tweaks: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    ensure: bool = Field(default=False, description="Sync the flow before running")
