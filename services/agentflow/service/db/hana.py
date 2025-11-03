from __future__ import annotations

from contextlib import suppress
from datetime import datetime
from typing import Optional

from ..config import get_settings

try:
    from hdbcli import dbapi
except ImportError:  # pragma: no cover - optional dependency
    dbapi = None  # type: ignore[assignment]


def _schema_and_table() -> tuple[str, str]:
    settings = get_settings()
    schema = settings.hana_schema or settings.hana_user
    table = settings.registry_table
    return schema.upper(), table.upper()


def get_hana_connection():
    """
    Establish a DBAPI connection to SAP HANA if the driver is available.
    """
    settings = get_settings()
    if not settings.hana_enabled:
        raise RuntimeError("HANA integration disabled via configuration")
    if dbapi is None:
        raise RuntimeError("hdbcli package not available")

    connect_kwargs = {
        "address": settings.hana_host,
        "port": settings.hana_port,
        "user": settings.hana_user,
        "password": settings.hana_password,
    }
    if settings.hana_ssl:
        connect_kwargs["encrypt"] = "true"
        connect_kwargs["sslValidateCertificate"] = "false"

    return dbapi.connect(**connect_kwargs)


def ensure_hana_registry() -> bool:
    """
    Ensure the registry table exists in HANA, returning True when available.
    """
    settings = get_settings()
    if not settings.hana_enabled or dbapi is None:
        return False

    schema, table = _schema_and_table()
    conn = get_hana_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT TABLE_NAME
            FROM SYS.TABLES
            WHERE SCHEMA_NAME = ? AND TABLE_NAME = ?
            """,
            (schema, table),
        )
        exists = cursor.fetchone() is not None
        if exists:
            return True

        cursor.execute(
            f"""
            CREATE TABLE "{schema}"."{table}" (
                "LOCAL_ID" NVARCHAR(255) PRIMARY KEY,
                "REMOTE_ID" NVARCHAR(255),
                "NAME" NVARCHAR(255),
                "DESCRIPTION" NVARCHAR(1024),
                "PROJECT_ID" NVARCHAR(255),
                "UPDATED_AT" TIMESTAMP,
                "SYNCED_AT" TIMESTAMP
            )
            """
        )
        conn.commit()
        return True
    finally:
        with suppress(Exception):
            cursor.close()
        with suppress(Exception):
            conn.close()


def upsert_hana_record(
    local_id: str,
    remote_id: Optional[str],
    name: Optional[str],
    description: Optional[str],
    project_id: Optional[str],
    updated_at: Optional[datetime],
    synced_at: datetime,
) -> None:
    """
    Upsert a flow registry row into HANA when available.
    """
    settings = get_settings()
    if not settings.hana_enabled or dbapi is None:
        return

    schema, table = _schema_and_table()
    conn = get_hana_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            f"""
            UPSERT "{schema}"."{table}"
            ("LOCAL_ID", "REMOTE_ID", "NAME", "DESCRIPTION", "PROJECT_ID", "UPDATED_AT", "SYNCED_AT")
            VALUES (?, ?, ?, ?, ?, ?, ?)
            WITH PRIMARY KEY
            """,
            (
                local_id,
                remote_id,
                name,
                description,
                project_id,
                updated_at,
                synced_at,
            ),
        )
        conn.commit()
    finally:
        with suppress(Exception):
            cursor.close()
        with suppress(Exception):
            conn.close()
