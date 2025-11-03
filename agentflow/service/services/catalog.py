from __future__ import annotations

import json
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
        return self.raw.get("metadata", {})


class FlowCatalog:
    """
    Loads flow definitions from the repository-managed `flows/` directory.
    """

    def __init__(self, root: Path):
        self.root = root

    def _iter_json_files(self) -> Iterable[Path]:
        if not self.root.exists():
            return []
        return sorted(path for path in self.root.rglob("*.json") if path.is_file())

    def list(self) -> List[FlowSpec]:
        specs: List[FlowSpec] = []
        for json_path in self._iter_json_files():
            with json_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            flow_id = data.get("id") or data.get("name")
            if not flow_id:
                raise ValueError(f"Flow {json_path} missing 'id' field")
            specs.append(
                FlowSpec(
                    id=flow_id,
                    name=data.get("name"),
                    description=data.get("description"),
                    category=data.get("category"),
                    tags=[tag for tag in data.get("tags", []) if isinstance(tag, str)],
                    path=json_path,
                    raw=data,
                )
            )
        return specs

    def get(self, flow_id: str) -> FlowSpec:
        for spec in self.list():
            if spec.id == flow_id:
                return spec
        raise KeyError(f"Flow {flow_id} not found in catalog {self.root}")
