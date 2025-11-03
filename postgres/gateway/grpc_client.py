from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any, Iterable

import grpc
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct
from google.protobuf.timestamp_pb2 import Timestamp

from gateway.generated.v1 import postgres_lang_service_pb2 as pb2
from gateway.generated.v1 import postgres_lang_service_pb2_grpc as pb2_grpc
from gateway.settings import Settings


_STATUS_BY_NAME: dict[str, pb2.OperationStatus.ValueType] = {
    "unspecified": pb2.OPERATION_STATUS_UNSPECIFIED,
    "running": pb2.OPERATION_STATUS_RUNNING,
    "success": pb2.OPERATION_STATUS_SUCCESS,
    "error": pb2.OPERATION_STATUS_ERROR,
}

_STATUS_BY_ENUM: dict[int, str] = {value: name for name, value in _STATUS_BY_NAME.items()}


def _struct_from_dict(payload: dict[str, Any] | None) -> Struct:
    struct = Struct()
    if payload:
        struct.update(payload)
    return struct


def _timestamp_from_iso(value: str | None) -> Timestamp | None:
    if not value:
        return None
    # Support both date-only and full ISO-8601 timestamps.
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    ts = Timestamp()
    ts.FromDatetime(dt)
    return ts


def _message_to_dict(message: Any) -> dict[str, Any]:
    return MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=True,
        including_default_value_fields=False,
    )


class PostgresTelemetryClient:
    """Thin wrapper around the generated gRPC client for PostgresLangService."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._channel: grpc.Channel | None = None
        self._stub: pb2_grpc.PostgresLangServiceStub | None = None

    def connect(self) -> None:
        if self._channel is not None:
            return

        self._channel = grpc.insecure_channel(self._settings.grpc_target)
        self._stub = pb2_grpc.PostgresLangServiceStub(self._channel)

    def close(self) -> None:
        if self._channel is None:
            return
        with contextlib.suppress(Exception):
            self._channel.close()  # type: ignore[attr-defined]
        self._channel = None
        self._stub = None

    @property
    def timeout(self) -> float:
        return max(self._settings.grpc_timeout_seconds, 0.1)

    def _require_stub(self) -> pb2_grpc.PostgresLangServiceStub:
        if self._stub is None:
            raise RuntimeError("gRPC stub not initialised; call connect() before issuing requests")
        return self._stub

    def health(self) -> dict[str, Any]:
        response = self._require_stub().HealthCheck(pb2.HealthCheckRequest(), timeout=self.timeout)
        return _message_to_dict(response)

    def list_operations(
        self,
        *,
        library_type: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> dict[str, Any]:
        request = pb2.ListLangOperationsRequest()
        if library_type:
            request.library_type = library_type
        if session_id:
            request.session_id = session_id
        if status:
            enum_value = _STATUS_BY_NAME.get(status.lower())
            if enum_value is None:
                raise ValueError(f"unknown status '{status}'")
            request.status = enum_value
        if page_size:
            request.page_size = page_size
        if page_token:
            request.page_token = page_token
        after_ts = _timestamp_from_iso(created_after)
        if after_ts is not None:
            request.created_after.CopyFrom(after_ts)
        before_ts = _timestamp_from_iso(created_before)
        if before_ts is not None:
            request.created_before.CopyFrom(before_ts)

        response = self._require_stub().ListLangOperations(request, timeout=self.timeout)
        return _message_to_dict(response)

    def get_operation(self, operation_id: str) -> dict[str, Any]:
        if not operation_id:
            raise ValueError("operation_id is required")
        request = pb2.GetLangOperationRequest(id=operation_id)
        response = self._require_stub().GetLangOperation(request, timeout=self.timeout)
        return _message_to_dict(response)

    def log_operation(self, payload: dict[str, Any]) -> dict[str, Any]:
        operation = pb2.LangOperation()
        if op_id := payload.get("id"):
            operation.id = str(op_id)
        if lib := payload.get("library_type"):
            operation.library_type = lib
        if operation_name := payload.get("operation"):
            operation.operation = operation_name
        operation.input.CopyFrom(_struct_from_dict(payload.get("input")))
        operation.output.CopyFrom(_struct_from_dict(payload.get("output")))
        if status := payload.get("status"):
            enum_value = _STATUS_BY_NAME.get(str(status).lower())
            if enum_value is None:
                raise ValueError(f"unknown status '{status}'")
            operation.status = enum_value
        if error := payload.get("error"):
            operation.error = str(error)
        if latency := payload.get("latency_ms"):
            operation.latency_ms = int(latency)
        if session_id := payload.get("session_id"):
            operation.session_id = str(session_id)
        if user_id := payload.get("user_id_hash"):
            operation.user_id_hash = str(user_id)
        if privacy := payload.get("privacy_level"):
            operation.privacy_level = str(privacy)
        created_at = _timestamp_from_iso(payload.get("created_at"))
        if created_at is not None:
            operation.created_at.CopyFrom(created_at)
        completed_at = _timestamp_from_iso(payload.get("completed_at"))
        if completed_at is not None:
            operation.completed_at.CopyFrom(completed_at)

        request = pb2.LogLangOperationRequest(operation=operation)
        response = self._require_stub().LogLangOperation(request, timeout=self.timeout)
        return _message_to_dict(response)

    def analytics(
        self,
        *,
        start_time: str | None = None,
        end_time: str | None = None,
        library_type: str | None = None,
    ) -> dict[str, Any]:
        request = pb2.AnalyticsRequest()
        start_ts = _timestamp_from_iso(start_time)
        if start_ts is not None:
            request.start_time.CopyFrom(start_ts)
        end_ts = _timestamp_from_iso(end_time)
        if end_ts is not None:
            request.end_time.CopyFrom(end_ts)
        if library_type:
            request.library_type = library_type

        response = self._require_stub().GetAnalytics(request, timeout=self.timeout)
        return _message_to_dict(response)

    def cleanup(self, *, older_than: str) -> dict[str, Any]:
        older_ts = _timestamp_from_iso(older_than)
        if older_ts is None:
            raise ValueError("older_than must be a valid ISO-8601 timestamp")
        request = pb2.CleanupRequest()
        request.older_than.CopyFrom(older_ts)
        response = self._require_stub().CleanupOperations(request, timeout=self.timeout)
        return _message_to_dict(response)

    @staticmethod
    def list_statuses() -> Iterable[dict[str, Any]]:
        for name, enum_value in _STATUS_BY_NAME.items():
            yield {"name": name, "value": int(enum_value)}

