from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_name: str = Field(default="aModels DMS", env="DMS_APP_NAME")
    debug: bool = Field(default=False, env="DMS_DEBUG")

    postgres_dsn: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/dms",
        env="DMS_POSTGRES_DSN",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="DMS_REDIS_URL")
    neo4j_uri: str = Field(default="neo4j://localhost:7687", env="DMS_NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="DMS_NEO4J_USER")
    neo4j_password: str = Field(default="neo4j", env="DMS_NEO4J_PASSWORD")
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
