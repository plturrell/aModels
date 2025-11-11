
from __future__ import annotations

from contextlib import suppress

import psycopg2
from psycopg2 import sql

from ..config import get_settings

def _get_postgres_connection():
    """
    Establish a connection to PostgreSQL.
    """
    settings = get_settings()
    if not settings.postgres_enabled:
        raise RuntimeError("PostgreSQL integration disabled via configuration")

    return psycopg2.connect(
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        host=settings.postgres_host,
        port=settings.postgres_port,
    )

def ensure_postgres_registry() -> bool:
    """
    Ensure the registry table exists in PostgreSQL, returning True when available.
    """
    settings = get_settings()
    if not settings.postgres_enabled:
        return False

    schema = "public"
    table = settings.registry_table.lower()

    conn = _get_postgres_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            sql.SQL(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = %s
                    AND table_name = %s
                )
                """
            ),
            (schema, table),
        )
        exists = cursor.fetchone()[0]
        if exists:
            return True

        cursor.execute(
            sql.SQL(
                """
                CREATE TABLE {schema}.{table} (
                    local_id VARCHAR(255) PRIMARY KEY,
                    remote_id VARCHAR(255),
                    name VARCHAR(255),
                    description TEXT,
                    project_id VARCHAR(255),
                    updated_at TIMESTAMP,
                    synced_at TIMESTAMP
                )
                """
            ).format(schema=sql.Identifier(schema), table=sql.Identifier(table))
        )
        conn.commit()
        return True
    finally:
        with suppress(Exception):
            cursor.close()
        with suppress(Exception):
            conn.close()
