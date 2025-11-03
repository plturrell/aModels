from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    parts = [chunk.strip() for chunk in value.split(",")]
    return [item for item in parts if item]


@dataclass
class Settings:
    """Runtime configuration for the FastAPI gateway."""

    grpc_target: str = field(default_factory=lambda: _get_env("POSTGRES_LANG_SERVICE_ADDR", "localhost:50055"))
    grpc_timeout_seconds: float = field(
        default_factory=lambda: float(_get_env("POSTGRES_LANG_SERVICE_TIMEOUT_SECONDS", "5"))
    )
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(_get_env("POSTGRES_LANG_GATEWAY_CORS", ""))
    )
    service_version: str = field(default_factory=lambda: _get_env("SERVICE_VERSION", "0.1.0"))
    db_dsn: Optional[str] = field(
        default_factory=lambda: (
            _get_env("POSTGRES_LANG_DB_DSN", _get_env("POSTGRES_DSN", "")) or None
        )
    )
    db_allow_mutations: bool = field(
        default_factory=lambda: _get_bool("POSTGRES_DB_ALLOW_MUTATIONS", False)
    )
    db_default_limit: int = field(
        default_factory=lambda: _get_int("POSTGRES_DB_DEFAULT_LIMIT", 200)
    )


def load_settings() -> Settings:
    return Settings()
