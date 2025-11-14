#!/usr/bin/env python3
"""
Generalized code view builder for knowledge graph payloads.

This module builds payloads for code-to-knowledge graph conversion,
supporting multiple code types (DDL, SQL, JSON, etc.) and projects.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

# Import shared utilities from sgmi_view_builder
# For now, we'll keep the core parsing logic but make it configurable
try:
    from sgmi_view_builder import (
        build_view_lineage,
        _collect_env_paths,
        VIEW_REGEX,
        _split_statements,
    )
except ImportError:
    # Fallback if sgmi_view_builder is not available
    def build_view_lineage(*args, **kwargs):
        return []
    
    def _collect_env_paths(var_name: str) -> list[Path]:
        raw = os.environ.get(var_name, "")
        if not raw:
            return []
        return [Path(part) for part in raw.split(":") if part]
    
    VIEW_REGEX = None
    _split_statements = None


def build_payload(
    json_paths: list[Path] = None,
    ddl_paths: list[Path] = None,
    sql_paths: list[Path] = None,
    xml_paths: list[Path] = None,
    project_id: str = "",
    system_id: str = "",
    information_system_id: str = "",
    view_tmp_dir: Optional[Path] = None,
    view_registry_out: Optional[Path] = None,
    view_summary_out: Optional[Path] = None,
) -> dict:
    """Build a knowledge graph payload from code files."""
    json_paths = json_paths or []
    ddl_paths = ddl_paths or []
    sql_paths = sql_paths or []
    xml_paths = xml_paths or []

    # Collect JSON files
    json_files = [str(path) for path in json_paths if path.exists()]

    # Collect DDL files
    ddl_files = []
    for path in ddl_paths:
        if path.exists():
            ddl_files.append(str(path))

    # Collect SQL query files
    sql_queries = []
    for path in sql_paths:
        if path.exists():
            content = path.read_text(encoding="utf-8")
            sql_queries.append(content)

    # Collect Control-M/XML files
    xml_files = [str(path) for path in xml_paths if path.exists()]

    # Build view lineage if DDL files provided
    view_lineage = []
    if ddl_paths and view_tmp_dir:
        try:
            view_lineage = build_view_lineage(
                ddl_paths=ddl_paths,
                tmp_dir=view_tmp_dir,
                registry_out=view_registry_out,
                summary_out=view_summary_out,
            )
        except Exception as e:
            print(f"Warning: Failed to build view lineage: {e}", file=sys.stderr)

    payload = {
        "json_tables": json_files,
        "hive_ddls": ddl_files,
        "sql_queries": sql_queries,
        "control_m_files": xml_files,
        "project_id": project_id or os.environ.get("PROJECT_ID", ""),
        "system_id": system_id or os.environ.get("SYSTEM_ID", ""),
        "information_system_id": information_system_id or os.environ.get("INFORMATION_SYSTEM_ID", ""),
    }

    if view_lineage:
        payload["view_lineage"] = view_lineage

    # Add Gitea storage configuration if available
    gitea_url = os.environ.get("GITEA_URL", "")
    gitea_token = os.environ.get("GITEA_TOKEN", "")
    if gitea_url or gitea_token:
        payload["gitea_storage"] = {
            "enabled": True,
            "gitea_url": gitea_url,
            "gitea_token": gitea_token,
            "owner": os.environ.get("GITEA_OWNER", "extract-service"),
            "repo_name": os.environ.get("GITEA_REPO_NAME", f"{payload.get('project_id', 'sgmi')}-extracted-code"),
            "branch": os.environ.get("GITEA_BRANCH", "main"),
            "base_path": os.environ.get("GITEA_BASE_PATH", ""),
            "auto_create": os.environ.get("GITEA_AUTO_CREATE", "true").lower() == "true",
            "description": os.environ.get("GITEA_DESCRIPTION", f"Extracted code for project {payload.get('project_id', 'sgmi')}"),
        }

    return payload


def load_config(config_path: Path) -> dict:
    """Load configuration from JSON or YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    content = config_path.read_text(encoding="utf-8")
    
    # Try JSON first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try YAML
    try:
        import yaml
        return yaml.safe_load(content)
    except ImportError:
        raise ValueError("YAML support requires PyYAML. Install with: pip install pyyaml")
    except yaml.YAMLError:
        raise ValueError(f"Invalid YAML in config file: {config_path}")


def main(output_path: Path, config_path: Optional[Path] = None) -> None:
    """Main function to build payload."""
    if config_path and config_path.exists():
        # Load from config file
        config = load_config(config_path)
        project_config = config.get("project", {})
        sources = config.get("sources", {})
        gitea_config = config.get("gitea_storage", {})
        
        json_paths = [Path(p) for p in sources.get("files", [])]
        ddl_paths = []
        sql_paths = []
        xml_paths = []
        
        # Extract file paths by extension
        for file_path in json_paths:
            ext = file_path.suffix.lower()
            if ext in [".hql", ".ddl"]:
                ddl_paths.append(file_path)
            elif ext == ".sql":
                sql_paths.append(file_path)
            elif ext == ".xml":
                xml_paths.append(file_path)
        
        payload = build_payload(
            json_paths=json_paths,
            ddl_paths=ddl_paths,
            sql_paths=sql_paths,
            xml_paths=xml_paths,
            project_id=project_config.get("id", ""),
            system_id=project_config.get("system_id", ""),
        )
        
        # Add Gitea storage configuration from config file
        if gitea_config.get("enabled", False):
            # Expand environment variables in Gitea config
            gitea_url = os.path.expandvars(gitea_config.get("gitea_url", os.environ.get("GITEA_URL", "")))
            gitea_token = os.path.expandvars(gitea_config.get("gitea_token", os.environ.get("GITEA_TOKEN", "")))
            
            if gitea_url or gitea_token:
                payload["gitea_storage"] = {
                    "enabled": True,
                    "gitea_url": gitea_url,
                    "gitea_token": gitea_token,
                    "owner": gitea_config.get("owner", "extract-service"),
                    "repo_name": gitea_config.get("repo_name", f"{payload.get('project_id', 'sgmi')}-extracted-code"),
                    "branch": gitea_config.get("branch", "main"),
                    "base_path": gitea_config.get("base_path", ""),
                    "auto_create": gitea_config.get("auto_create", True),
                    "description": gitea_config.get("description", f"Extracted code for project {payload.get('project_id', 'sgmi')}"),
                }
    else:
        # Fallback to environment variables (backward compatibility)
        json_paths = _collect_env_paths("JSON_FILES")
        ddl_paths = _collect_env_paths("DDL_FILES")
        sql_paths = _collect_env_paths("SQL_FILES")
        xml_paths = _collect_env_paths("CONTROLM_FILES")
        
        # Also support SGMI_ prefixed vars for backward compatibility
        if not json_paths:
            json_paths = _collect_env_paths("SGMI_JSON_FILES")
        if not ddl_paths:
            ddl_paths = _collect_env_paths("SGMI_DDL_FILES")
        if not xml_paths:
            xml_paths = _collect_env_paths("SGMI_CONTROLM_FILES")

        view_tmp_dir_raw = os.environ.get("VIEW_TMP_DIR") or os.environ.get("SGMI_VIEW_TMP_DIR", "")
        view_tmp_dir = Path(view_tmp_dir_raw) if view_tmp_dir_raw else None
        view_registry_raw = os.environ.get("VIEW_REGISTRY_OUT") or os.environ.get("SGMI_VIEW_REGISTRY_OUT", "")
        view_registry_out = Path(view_registry_raw) if view_registry_raw else None
        view_summary_raw = os.environ.get("VIEW_SUMMARY_OUT") or os.environ.get("SGMI_VIEW_SUMMARY_OUT", "")
        view_summary_out = Path(view_summary_raw) if view_summary_raw else None

        payload = build_payload(
            json_paths=json_paths,
            ddl_paths=ddl_paths,
            sql_paths=sql_paths,
            xml_paths=xml_paths,
            view_tmp_dir=view_tmp_dir,
            view_registry_out=view_registry_out,
            view_summary_out=view_summary_out,
        )

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Payload written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: code_view_builder.py <output_json_path> [config_path]")
    
    output_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    main(output_path, config_path)

