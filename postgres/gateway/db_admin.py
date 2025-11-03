from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Iterable

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from gateway.settings import Settings


_READONLY_BLOCKLIST = {
    "insert",
    "update",
    "delete",
    "alter",
    "drop",
    "create",
    "grant",
    "revoke",
    "truncate",
    "comment",
    "merge",
    "refresh",
    "vacuum",
    "analyze",
    "cluster",
    "checkpoint",
}


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool


class DatabaseAdmin:
    """Helper for exposing limited Postgres administration features."""

    def __init__(self, dsn: str, allow_mutations: bool, default_limit: int) -> None:
        self._pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=4,
            kwargs={"autocommit": True},
        )
        self._allow_mutations = allow_mutations
        self._default_limit = default_limit

    @classmethod
    def from_settings(cls, settings: Settings) -> DatabaseAdmin | None:
        if not settings.db_dsn:
            return None
        return cls(
            dsn=settings.db_dsn,
            allow_mutations=settings.db_allow_mutations,
            default_limit=settings.db_default_limit,
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._pool.close()

    @property
    def allow_mutations(self) -> bool:
        return self._allow_mutations

    @property
    def default_limit(self) -> int:
        return self._default_limit

    def list_tables(self) -> list[dict[str, Any]]:
        query = """
            SELECT table_schema, table_name
              FROM information_schema.tables
             WHERE table_type = 'BASE TABLE'
               AND table_schema NOT IN ('pg_catalog', 'information_schema')
          ORDER BY table_schema, table_name
        """
        return self._fetch_all(query)

    def get_columns(self, schema: str, table: str) -> list[dict[str, Any]]:
        query = """
            SELECT column_name, data_type, is_nullable, column_default
              FROM information_schema.columns
             WHERE table_schema = %(schema)s
               AND table_name = %(table)s
          ORDER BY ordinal_position
        """
        return self._fetch_all(query, {"schema": schema, "table": table})

    def execute_query(self, sql_text: str, limit: int | None = None) -> QueryResult:
        if not sql_text or not sql_text.strip():
            raise ValueError("SQL query must not be empty")

        statement = sql_text.strip().rstrip(";")
        keyword = statement.split(None, 1)[0].lower()
        if not self._allow_mutations and keyword in _READONLY_BLOCKLIST:
            raise PermissionError(f"{keyword.upper()} statements are disabled in read-only mode")

        effective_limit = limit if limit and limit > 0 else self._default_limit
        text_for_execution = statement
        if effective_limit and keyword in {"select", "with"} and " limit " not in statement.lower():
            text_for_execution = f"{statement} LIMIT {effective_limit + 1}"

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(text_for_execution)
            description = cursor.description
            if description is None:
                affected = cursor.rowcount if cursor.rowcount != -1 else 0
                return QueryResult(columns=[], rows=[], row_count=affected, truncated=False)

            rows = cursor.fetchall()
            column_names = [col.name for col in description]

        truncated = False
        if effective_limit and len(rows) > effective_limit:
            rows = rows[:effective_limit]
            truncated = True
        return QueryResult(
            columns=list(rows[0].keys()) if rows else column_names,
            rows=rows,
            row_count=len(rows),
            truncated=truncated,
        )

    def _fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cursor:
            cursor.execute(query, params or {})
            results = cursor.fetchall()
        return results
