
from __future__ import annotations

import os
import threading
import time
from contextlib import suppress
from functools import wraps
from typing import Optional

import psycopg2
from psycopg2 import pool, sql
from psycopg2.pool import ThreadedConnectionPool

from ..config import get_settings

# Global connection pool
_postgres_pool: Optional[ThreadedConnectionPool] = None
_pool_lock = threading.Lock()


def _get_postgres_pool() -> Optional[ThreadedConnectionPool]:
    """
    Get or create the PostgreSQL connection pool.
    """
    global _postgres_pool
    
    if _postgres_pool is not None:
        return _postgres_pool
    
    with _pool_lock:
        # Double-check after acquiring lock
        if _postgres_pool is not None:
            return _postgres_pool
        
        settings = get_settings()
        if not settings.postgres_enabled:
            return None

        # Get pool configuration from environment variables
        pool_size = 5
        if val := os.getenv("AGENTFLOW_POSTGRES_POOL_SIZE"):
            try:
                pool_size = int(val)
                if pool_size < 1:
                    pool_size = 5
            except ValueError:
                pass

        max_overflow = 5
        if val := os.getenv("AGENTFLOW_POSTGRES_POOL_MAX_OVERFLOW"):
            try:
                max_overflow = int(val)
                if max_overflow < 0:
                    max_overflow = 5
            except ValueError:
                pass

        minconn = 1
        maxconn = pool_size + max_overflow

        try:
            _postgres_pool = ThreadedConnectionPool(
                minconn=minconn,
                maxconn=maxconn,
                dbname=settings.postgres_db,
                user=settings.postgres_user,
                password=settings.postgres_password,
                host=settings.postgres_host,
                port=settings.postgres_port,
            )
            return _postgres_pool
        except Exception:
            # If pool creation fails, return None to fallback to direct connections
            return None


def _get_postgres_connection():
    """
    Get a connection from the pool or create a direct connection.
    Falls back to direct connection if pool is unavailable.
    """
    pool = _get_postgres_pool()
    if pool is not None:
        try:
            return pool.getconn()
        except Exception:
            # Fallback to direct connection if pool fails
            pass

    # Fallback to direct connection
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


def _return_postgres_connection(conn):
    """
    Return a connection to the pool or close it if direct connection.
    """
    pool = _get_postgres_pool()
    if pool is not None:
        try:
            pool.putconn(conn)
            return
        except Exception:
            pass
    
    # Fallback: close direct connection
    with suppress(Exception):
        conn.close()


def retry_postgres_operation(max_attempts: int = 3, initial_backoff: float = 0.1, max_backoff: float = 0.5):
    """
    Decorator to retry Postgres operations with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            backoff = initial_backoff
            last_exception = None

            for attempt in range(max_attempts):
                if attempt > 0:
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)

                try:
                    return func(*args, **kwargs)
                except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                    last_exception = e
                    error_str = str(e).lower()
                    # Check if error is retryable
                    is_retryable = (
                        "connection" in error_str or
                        "network" in error_str or
                        "timeout" in error_str or
                        "broken pipe" in error_str or
                        "connection reset" in error_str
                    )
                    if not is_retryable:
                        # Non-retryable error, re-raise immediately
                        raise
                    # Continue to retry
                except Exception:
                    # Non-retryable exceptions, re-raise immediately
                    raise

            # All retries exhausted
            raise Exception(f"Operation failed after {max_attempts} attempts") from last_exception
        return wrapper
    return decorator

@retry_postgres_operation(
    max_attempts=int(os.getenv("AGENTFLOW_POSTGRES_RETRY_MAX_ATTEMPTS", "3")),
    initial_backoff=0.1,
    max_backoff=0.5,
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
            _return_postgres_connection(conn)


def close_postgres_pool():
    """
    Close the PostgreSQL connection pool.
    Should be called on application shutdown.
    """
    global _postgres_pool
    if _postgres_pool is not None:
        try:
            _postgres_pool.closeall()
        except Exception:
            pass
        finally:
            _postgres_pool = None
