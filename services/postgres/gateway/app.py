from __future__ import annotations

from typing import Any, Optional

import grpc
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from .grpc_client import PostgresTelemetryClient
from .db_admin import DatabaseAdmin
from .settings import Settings, load_settings


class OperationLogPayload(BaseModel):
    id: Optional[str] = Field(default=None, description="Operation identifier (UUID).")
    library_type: Optional[str] = Field(default=None, description="Source library (e.g. langflow, langgraph).")
    operation: Optional[str] = Field(default=None, description="Operation name.")
    input: Optional[dict[str, Any]] = Field(default=None, description="Input payload stored as JSON.")
    output: Optional[dict[str, Any]] = Field(default=None, description="Output payload stored as JSON.")
    status: Optional[str] = Field(default=None, description="Operation status: running, success, error.")
    error: Optional[str] = Field(default=None, description="Error message, if any.")
    latency_ms: Optional[int] = Field(default=None, description="Latency in milliseconds.")
    session_id: Optional[str] = Field(default=None, description="Workflow session identifier.")
    user_id_hash: Optional[str] = Field(default=None, description="Hashed user identifier.")
    privacy_level: Optional[str] = Field(default=None, description="Privacy level string.")
    created_at: Optional[str] = Field(
        default=None,
        description="Creation timestamp in ISO-8601 format (defaults to now on the server if omitted).",
    )
    completed_at: Optional[str] = Field(default=None, description="Completion timestamp in ISO-8601 format.")


class CleanupRequest(BaseModel):
    older_than: str = Field(description="Delete operations created strictly before this ISO-8601 timestamp.")


settings = load_settings()
client = PostgresTelemetryClient(settings)
db_admin = DatabaseAdmin.from_settings(settings)


def get_settings() -> Settings:
    return settings


def get_client(_: Settings = Depends(get_settings)) -> PostgresTelemetryClient:
    return client


def get_db_admin(_: Settings = Depends(get_settings)) -> DatabaseAdmin:
    if db_admin is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database administration is not enabled on this gateway.",
        )
    return db_admin


app = FastAPI(title="Postgres Lang Telemetry Gateway", version=settings.service_version)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.on_event("startup")
def on_startup() -> None:
    client.connect()
    if db_admin:
        # Warm the connection pool to surface DSN issues early.
        try:
            db_admin.list_tables()
        except Exception as exc:  # pragma: no cover - defensive logging
            raise RuntimeError(f"Failed to initialise database admin connection: {exc}") from exc


@app.on_event("shutdown")
def on_shutdown() -> None:
    client.close()
    if db_admin:
        db_admin.close()


def _grpc_exc_to_http(err: grpc.RpcError) -> HTTPException:
    detail = getattr(err, "details", None)
    status_code = status.HTTP_502_BAD_GATEWAY
    if hasattr(err, "code"):
        code = err.code()
        # Map a few common gRPC status codes to HTTP responses.
        if code == grpc.StatusCode.NOT_FOUND:
            status_code = status.HTTP_404_NOT_FOUND
        elif code == grpc.StatusCode.INVALID_ARGUMENT:
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        elif code in (grpc.StatusCode.DEADLINE_EXCEEDED, grpc.StatusCode.UNAVAILABLE):
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
        detail = {"grpc_code": code.name, "message": err.details()}
    return HTTPException(status_code=status_code, detail=detail or str(err))


@app.get("/health")
def health(client: PostgresTelemetryClient = Depends(get_client)) -> dict[str, Any]:
    try:
        return client.health()
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.get("/operations")
def list_operations(
    library_type: Optional[str] = Query(default=None),
    session_id: Optional[str] = Query(default=None),
    status_name: Optional[str] = Query(default=None, alias="status"),
    page_size: Optional[int] = Query(default=50, ge=1, le=500),
    page_token: Optional[str] = Query(default=None),
    created_after: Optional[str] = Query(default=None),
    created_before: Optional[str] = Query(default=None),
    client: PostgresTelemetryClient = Depends(get_client),
) -> dict[str, Any]:
    try:
        return client.list_operations(
            library_type=library_type,
            session_id=session_id,
            status=status_name,
            page_size=page_size,
            page_token=page_token,
            created_after=created_after,
            created_before=created_before,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.get("/operations/{operation_id}")
def get_operation(operation_id: str, client: PostgresTelemetryClient = Depends(get_client)) -> dict[str, Any]:
    try:
        return client.get_operation(operation_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.post("/operations", status_code=status.HTTP_201_CREATED)
def log_operation(payload: OperationLogPayload, client: PostgresTelemetryClient = Depends(get_client)) -> dict[str, Any]:
    try:
        return client.log_operation(payload.dict(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.get("/analytics")
def analytics(
    start_time: Optional[str] = Query(default=None),
    end_time: Optional[str] = Query(default=None),
    library_type: Optional[str] = Query(default=None),
    client: PostgresTelemetryClient = Depends(get_client),
) -> dict[str, Any]:
    try:
        return client.analytics(start_time=start_time, end_time=end_time, library_type=library_type)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.post("/cleanup")
def cleanup(request: CleanupRequest, client: PostgresTelemetryClient = Depends(get_client)) -> dict[str, Any]:
    try:
        return client.cleanup(older_than=request.older_than)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except grpc.RpcError as exc:
        raise _grpc_exc_to_http(exc)


@app.get("/statuses")
def statuses() -> list[dict[str, Any]]:
    return list(PostgresTelemetryClient.list_statuses())


class DBStatusResponse(BaseModel):
    enabled: bool
    allow_mutations: bool
    default_limit: int


class TableInfo(BaseModel):
    table_schema: str
    table_name: str


class TableListResponse(BaseModel):
    tables: list[TableInfo]


class ColumnInfo(BaseModel):
    column_name: str
    data_type: str
    is_nullable: str
    column_default: Optional[Any] = None


class SQLQueryRequest(BaseModel):
    sql: str = Field(description="SQL statement to execute.")
    limit: Optional[int] = Field(default=None, description="Maximum number of rows to return.")


class SQLQueryResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


@app.get("/db/status", response_model=DBStatusResponse)
def db_status() -> DBStatusResponse:
    return DBStatusResponse(
        enabled=db_admin is not None,
        allow_mutations=db_admin.allow_mutations if db_admin else False,
        default_limit=db_admin.default_limit if db_admin else 0,
    )


@app.get("/db/tables", response_model=TableListResponse)
def db_tables(admin: DatabaseAdmin = Depends(get_db_admin)) -> TableListResponse:
    tables = [TableInfo(**row) for row in admin.list_tables()]
    return TableListResponse(tables=tables)


@app.get("/db/table/{schema}/{table}", response_model=list[ColumnInfo])
def db_table_columns(schema: str, table: str, admin: DatabaseAdmin = Depends(get_db_admin)) -> list[ColumnInfo]:
    return [ColumnInfo(**row) for row in admin.get_columns(schema, table)]


@app.post("/db/query", response_model=SQLQueryResponse)
def db_query(payload: SQLQueryRequest, admin: DatabaseAdmin = Depends(get_db_admin)) -> SQLQueryResponse:
    try:
        result = admin.execute_query(payload.sql, payload.limit)
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    encoded_rows = jsonable_encoder(result.rows)
    return SQLQueryResponse(
        columns=result.columns,
        rows=encoded_rows,
        row_count=result.row_count,
        truncated=result.truncated,
    )
