from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class FlowSpec:
    """
    Representation of a local Langflow JSON definition.
    """

    id: str
    name: Optional[str]
    description: Optional[str]
    category: Optional[str]
    tags: List[str]
    path: Path
    raw: Dict[str, Any]

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata block from the raw Langflow definition."""
        return self.raw.get("metadata", {})


class FlowCatalog:
    """
    Loads flow definitions from the repository-managed `flows/` directory.
    """

    def __init__(self, root: Path, *, logger: Optional[logging.Logger] = None):
        self.root = root
        self._specs_by_id: Dict[str, FlowSpec] = {}
        self._spec_list: List[FlowSpec] = []
        self._snapshot: Dict[str, float] = {}
        self._logger = logger or logging.getLogger(__name__)

    def _iter_json_files(self) -> Iterable[Path]:
        if not self.root.exists():
            return []
        return sorted(path for path in self.root.rglob("*.json") if path.is_file())

    def _snapshot_paths(self, paths: Iterable[Path]) -> Dict[str, float]:
        snapshot: Dict[str, float] = {}
        for path in paths:
            try:
                snapshot[str(path)] = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
        return snapshot

    def _build_spec(self, json_path: Path, data: Dict[str, Any]) -> FlowSpec:
        flow_id = data.get("id") or data.get("name")
        if not flow_id:
            relative = json_path.relative_to(self.root) if json_path.is_relative_to(self.root) else json_path
            self._logger.error(
                "Flow definition missing id",
                extra={"path": str(json_path)},
            )
            raise ValueError(f"Flow {relative} missing 'id' field")

        tags = [tag for tag in data.get("tags", []) if isinstance(tag, str)]

        return FlowSpec(
            id=flow_id,
            name=data.get("name"),
            description=data.get("description"),
            category=data.get("category"),
            tags=tags,
            path=json_path,
            raw=data,
        )

    def _refresh_cache(self) -> None:
        paths = list(self._iter_json_files())
        snapshot = self._snapshot_paths(paths)
        if snapshot == self._snapshot:
            self._logger.debug(
                "flow catalog snapshot unchanged",
                extra={"root": str(self.root)},
            )
            return

        specs_by_id: Dict[str, FlowSpec] = {}
        spec_list: List[FlowSpec] = []

        for json_path in paths:
            try:
                with json_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except FileNotFoundError:
                # File was removed between discovery and load – skip; snapshot already reflects removal.
                continue
            except json.JSONDecodeError as exc:
                self._logger.error(
                    "Invalid JSON payload for flow",
                    extra={"path": str(json_path)},
                    exc_info=exc,
                )
                raise ValueError(f"Flow JSON at {json_path} is not valid: {exc.msg}") from exc

            spec = self._build_spec(json_path, data)
            if spec.id in specs_by_id:
                existing_path = specs_by_id[spec.id].path
                self._logger.error(
                    "Duplicate flow id detected",
                    extra={
                        "flow_id": spec.id,
                        "current_path": str(json_path),
                        "existing_path": str(existing_path),
                    },
                )
                raise ValueError(
                    f"Duplicate flow id '{spec.id}' in {json_path} (already defined in {existing_path})"
                )
            specs_by_id[spec.id] = spec
            spec_list.append(spec)

        self._specs_by_id = specs_by_id
        self._spec_list = spec_list
        self._snapshot = snapshot
        self._logger.info(
            "flow catalog refreshed",
            extra={
                "root": str(self.root),
                "flow_count": len(self._spec_list),
            },
        )

    def list(self) -> List[FlowSpec]:
        self._refresh_cache()
        return list(self._spec_list)

    def get(self, flow_id: str) -> FlowSpec:
        self._refresh_cache()
        spec = self._specs_by_id.get(flow_id)
        if spec is None:
            raise KeyError(f"Flow {flow_id} not found in catalog {self.root}")
        return spec
