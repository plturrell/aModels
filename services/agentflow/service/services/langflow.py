from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx


class LangflowError(Exception):
    """Raised when Langflow returns a non-success response."""


@dataclass
class FlowRecord:
    id: str
    name: Optional[str]
    description: Optional[str]
    project_id: Optional[str]
    updated_at: Optional[datetime]
    raw: Dict[str, Any]


@dataclass
class FlowImportRequest:
    flow: Dict[str, Any]
    force: bool = True
    project_id: Optional[str] = None
    folder_path: Optional[str] = None
    remote_id: Optional[str] = None

    def to_payload(self) -> Dict[str, Any]:
        flow_payload: Dict[str, Any] = dict(self.flow)
        if self.project_id:
            flow_payload.setdefault("project_id", self.project_id)
        if self.folder_path:
            flow_payload.setdefault("folder_path", self.folder_path)
        return {"flows": [flow_payload]}


@dataclass
class RunFlowRequest:
    input_value: Optional[str] = None
    inputs: Optional[Dict[str, Any]] = None
    chat_history: Optional[List[Dict[str, Any]]] = None
    session_id: Optional[str] = None
    tweaks: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None

    def to_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.input_value is not None:
            payload["input_value"] = self.input_value
        if self.inputs:
            payload["inputs"] = self.inputs
        if self.chat_history:
            payload["chat_history"] = self.chat_history
        if self.session_id:
            payload["session_id"] = self.session_id
        if self.tweaks:
            payload["tweaks"] = self.tweaks
        if self.stream is not None:
            payload["stream"] = self.stream
        return payload


class LangflowClient:
    """
    Minimal async HTTP client mirroring the Go AgentFlow functionality.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        auth_token: Optional[str] = None,
        timeout_seconds: int = 120,
    ):
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "agenticAiETH-agentflow-service/0.1",
        }
        if api_key:
            headers["X-API-Key"] = api_key
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout_seconds,
            headers=headers,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def version(self) -> str:
        response = await self._request("GET", "/api/v1/version")
        version = response.get("version")
        if isinstance(version, str):
            return version
        raise LangflowError(f"unexpected version payload: {response}")

    async def list_flows(self) -> List[FlowRecord]:
        payload = await self._request("GET", "/api/v1/flows/")
        return self._decode_flow_list(payload)

    async def import_flow(self, request: FlowImportRequest) -> FlowRecord:
        target_name = str(request.flow.get("name", "")).strip()

        if request.force and request.remote_id:
            response = await self._client.request("DELETE", f"/api/v1/flows/{request.remote_id}")
            if response.status_code not in (200, 202, 204, 404):
                detail = await response.aread()
                raise LangflowError(
                    f"Langflow delete failed ({response.status_code}): {detail.decode('utf-8', errors='ignore')}"
                )

        if request.force and target_name:
            existing = await self.list_flows()
            for candidate in existing:
                if candidate.name and candidate.name.strip() == target_name:
                    response = await self._client.request("DELETE", f"/api/v1/flows/{candidate.id}")
                    if response.status_code not in (200, 202, 204, 404):
                        detail = await response.aread()
                        raise LangflowError(
                            f"Langflow delete failed ({response.status_code}): {detail.decode('utf-8', errors='ignore')}"
                        )

        payload = await self._request("POST", "/api/v1/flows/batch/", json=request.to_payload())
        records = self._decode_flow_list(payload)
        if not records:
            raise LangflowError("empty response from Langflow import endpoint")
        return records[0]

    async def run_flow(self, flow_id: str, request: RunFlowRequest) -> Dict[str, Any]:
        payload = await self._request("POST", f"/api/v1/run/{flow_id}", json=request.to_payload())
        if not isinstance(payload, dict):
            raise LangflowError(f"unexpected run response payload: {payload}")
        return payload

    async def get_flow(self, flow_id: str) -> FlowRecord:
        payload = await self._request("GET", f"/api/v1/flows/{flow_id}")
        record = self._decode_flow(payload)
        if record is None:
            raise LangflowError(f"flow {flow_id} not found")
        return record

    async def _request(self, method: str, path: str, *, json: Optional[Dict[str, Any]] = None) -> Any:
        response = await self._client.request(method, path, json=json)
        if response.status_code >= 400:
            detail = response.text
            try:
                data = response.json()
                detail = data.get("detail") or data.get("message") or detail
            except ValueError:
                pass
            raise LangflowError(f"Langflow request failed ({response.status_code}): {detail}")
        if response.status_code == 204:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise LangflowError(f"invalid JSON response: {exc}") from exc

    def _decode_flow_list(self, payload: Any) -> List[FlowRecord]:
        if payload is None:
            return []
        if isinstance(payload, list):
            return [rec for rec in (self._decode_flow(item) for item in payload) if rec]
        if isinstance(payload, dict):
            nested = payload.get("flows") or payload.get("data") or payload.get("items")
            if nested is not None:
                return self._decode_flow_list(nested)
            record = self._decode_flow(payload)
            return [record] if record else []
        raise LangflowError(f"unexpected flows payload: {type(payload)}")

    def _decode_flow(self, payload: Any) -> Optional[FlowRecord]:
        if not isinstance(payload, dict):
            return None
        updated_at = payload.get("updated_at")
        parsed_updated: Optional[datetime] = None
        if isinstance(updated_at, str):
            with suppress(ValueError):
                parsed_updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        return FlowRecord(
            id=payload.get("id") or "",
            name=payload.get("name"),
            description=payload.get("description"),
            project_id=payload.get("project_id"),
            updated_at=parsed_updated,
            raw=payload,
        )
