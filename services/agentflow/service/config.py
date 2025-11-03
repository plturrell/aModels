from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


logger = logging.getLogger(__name__)


def _load_env_files() -> None:
    candidates: list[Path] = []

    custom = os.environ.get("AGENTFLOW_ENV_FILE", "").strip()
    if custom:
        candidates.append(Path(custom))

    # Local defaults relative to the working directory and this service package
    base_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path(__file__).resolve().parent.parent]
    filenames = [".env.agentflow", ".env"]

    for base in base_dirs:
        for name in filenames:
            candidates.append((base / name).resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen or not candidate.exists() or candidate.is_dir():
            continue
        if load_dotenv(candidate, override=False):
            logger.info("Loaded environment from %s", candidate)
            seen.add(candidate)


_load_env_files()


class Settings(BaseSettings):
    """
    Central configuration for the AgentFlow FastAPI service.

    Environment variables are prefixed with AGENTFLOW_*, mirroring the Go CLI
    flags where possible.
    """

    project_root: Path = Field(
        default=Path("/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_AgentFlow"),
        validation_alias="AGENTFLOW_PROJECT_ROOT",
    )
    flows_dir: Path = Field(
        default=Path("/Users/user/Library/CloudStorage/Dropbox/agenticAiETH/agenticAiETH_layer4_AgentFlow/flows"),
        validation_alias="AGENTFLOW_FLOWS_DIR",
    )
    sgmi_view_lineage_path: Path = Field(
        default=Path("store/sgmi_view_lineage.json"),
        validation_alias="AGENTFLOW_SGMI_VIEW_LINEAGE_PATH",
    )
    sgmi_view_summary_path: Path = Field(
        default=Path("store/sgmi_view_summary.json"),
        validation_alias="AGENTFLOW_SGMI_VIEW_SUMMARY_PATH",
    )
    registry_table: str = Field(
        default="AGENTFLOW_REGISTRY",
        validation_alias="AGENTFLOW_HANA_TABLE",
    )

    langflow_base_url: str = Field(
        default="http://localhost:7860",
        validation_alias="AGENTFLOW_LANGFLOW_URL",
    )
    langflow_api_key: Optional[str] = Field(
        default=None,
        validation_alias="AGENTFLOW_LANGFLOW_API_KEY",
    )
    langflow_auth_token: Optional[str] = Field(
        default=None,
        validation_alias="AGENTFLOW_LANGFLOW_AUTH_TOKEN",
    )
    langflow_timeout_seconds: int = Field(
        default=120,
        validation_alias="AGENTFLOW_LANGFLOW_TIMEOUT_SECONDS",
    )

    database_url: str = Field(
        default="sqlite:///./agentflow.db",
        validation_alias="AGENTFLOW_DATABASE_URL",
    )
    database_echo: bool = Field(
        default=False,
        validation_alias="AGENTFLOW_DATABASE_ECHO",
    )

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        validation_alias="AGENTFLOW_REDIS_URL",
    )
    redis_enabled: bool = Field(
        default=True,
        validation_alias="AGENTFLOW_REDIS_ENABLED",
    )
    redis_namespace: str = Field(
        default="agentflow",
        validation_alias="AGENTFLOW_REDIS_NAMESPACE",
    )

    hana_enabled: bool = Field(
        default=True,
        validation_alias="AGENTFLOW_HANA_ENABLED",
    )
    hana_host: str = Field(
        default="d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com",
        validation_alias="AGENTFLOW_HANA_HOST",
    )
    hana_port: int = Field(
        default=443,
        validation_alias="AGENTFLOW_HANA_PORT",
    )
    hana_user: str = Field(
        default="DBADMIN",
        validation_alias="AGENTFLOW_HANA_USER",
    )
    hana_password: str = Field(
        default="Initial@1",
        validation_alias="AGENTFLOW_HANA_PASSWORD",
    )
    hana_schema: Optional[str] = Field(
        default=None,
        validation_alias="AGENTFLOW_HANA_SCHEMA",
    )
    hana_ssl: bool = Field(
        default=True,
        validation_alias="AGENTFLOW_HANA_SSL",
    )

    postgres_enabled: bool = Field(
        default=True,
        validation_alias="AGENTFLOW_POSTGRES_ENABLED",
    )
    postgres_host: str = Field(
        default="localhost",
        validation_alias="AGENTFLOW_POSTGRES_HOST",
    )
    postgres_port: int = Field(
        default=5432,
        validation_alias="AGENTFLOW_POSTGRES_PORT",
    )
    postgres_user: str = Field(
        default="postgres",
        validation_alias="AGENTFLOW_POSTGRES_USER",
    )
    postgres_password: str = Field(
        default="postgres",
        validation_alias="AGENTFLOW_POSTGRES_PASSWORD",
    )
    postgres_db: str = Field(
        default="agentflow",
        validation_alias="AGENTFLOW_POSTGRES_DB",
    )

    service_api_key: Optional[str] = Field(
        default=None,
        validation_alias="AGENTFLOW_SERVICE_API_KEY",
    )
    allow_origins: str = Field(
        default="*",
        validation_alias="AGENTFLOW_ALLOW_ORIGINS",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    @field_validator("project_root", "flows_dir")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        path_value = Path(value)
        expanded = Path(os.path.expanduser(os.path.expandvars(str(path_value)))).resolve()
        if expanded.is_dir():
            return expanded
        if expanded.suffix:
            # File path – ensure parent directory exists
            expanded.parent.mkdir(parents=True, exist_ok=True)
            return expanded
        expanded.mkdir(parents=True, exist_ok=True)
        return expanded

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    def resolve_project_path(self, value: Path) -> Path:
        path_value = Path(os.path.expanduser(os.path.expandvars(str(value))))
        if path_value.is_absolute():
            return path_value.resolve()
        return (self.project_root / path_value).resolve()

    @property
    def sgmi_view_lineage_resolved(self) -> Path:
        return self.resolve_project_path(self.sgmi_view_lineage_path)

    @property
    def sgmi_view_summary_resolved(self) -> Path:
        return self.resolve_project_path(self.sgmi_view_summary_path)

    @field_validator(
        "redis_enabled",
        "hana_enabled",
        "hana_ssl",
        "database_echo",
        mode="before",
    )
    @classmethod
    def _parse_bool(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return False
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
        return value


@lru_cache()
def get_settings() -> Settings:
    return Settings()
