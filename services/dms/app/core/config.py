from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="aModels DMS", env="DMS_APP_NAME")
    debug: bool = Field(default=False, env="DMS_DEBUG")

    postgres_dsn: str = Field(
        ...,
        env="DMS_POSTGRES_DSN",
        description="PostgreSQL connection string. Must be provided via environment variable.",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="DMS_REDIS_URL")
    neo4j_uri: str = Field(
        ...,
        env="DMS_NEO4J_URI",
        description="Neo4j connection URI. Must be provided via environment variable.",
    )
    neo4j_user: str = Field(
        ...,
        env="DMS_NEO4J_USER",
        description="Neo4j username. Must be provided via environment variable.",
    )
    neo4j_password: str = Field(
        ...,
        env="DMS_NEO4J_PASSWORD",
        description="Neo4j password. Must be provided via environment variable.",
    )

    @field_validator("postgres_dsn")
    @classmethod
    def validate_postgres_dsn(cls, v: str) -> str:
        """Validate PostgreSQL DSN does not contain default credentials."""
        if not v or v.strip() == "":
            raise ValueError("DMS_POSTGRES_DSN must be set and cannot be empty")
        # Check for common default credentials in DSN
        if "postgres:postgres@" in v or ":postgres@" in v:
            raise ValueError(
                "DMS_POSTGRES_DSN contains default credentials. "
                "Please use a secure password and set via environment variable."
            )
        return v

    @field_validator("neo4j_password")
    @classmethod
    def validate_neo4j_password(cls, v: str) -> str:
        """Validate Neo4j password is not a default value."""
        if not v or v.strip() == "":
            raise ValueError("DMS_NEO4J_PASSWORD must be set and cannot be empty")
        if v.lower() in ("neo4j", "password", "admin", "root"):
            raise ValueError(
                "DMS_NEO4J_PASSWORD cannot be a default/common password. "
                "Please use a strong, unique password."
            )
        return v

    extract_url: Optional[str] = Field(default=None, env="DMS_EXTRACT_URL")
    catalog_url: Optional[str] = Field(default=None, env="DMS_CATALOG_URL")

    storage_root: Path = Field(
        default=Path("./data/documents").resolve(),
        env="DMS_STORAGE_ROOT",
    )
    storage_prefix: Optional[str] = Field(default=None, env="DMS_STORAGE_PREFIX")

    class Config:
        env_file = ".env.dms"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    settings.storage_root.mkdir(parents=True, exist_ok=True)
    return settings
