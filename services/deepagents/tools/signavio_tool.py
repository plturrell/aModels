"""Signavio stub tools used by DeepAgents for upload and OData retrieval."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Optional

from langchain_core.tools import tool

# Default directory where manual fixtures live
_MANUAL_DIR = Path(os.getenv("SIGNAVIO_STUB_DIR", "testing/manual/signavio"))
_UPLOAD_PREFIX = "stub://signavio/uploads"
_ODATA_PREFIX = "stub://signavio/odata"


def _ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Stub artifact not found: {path}")


def _slug(text: str) -> str:
    return text.strip().replace(" ", "-") or "default"


@tool
def signavio_stub_upload(
    dataset: str,
    file_path: str,
    schema_path: Optional[str] = None,
    primary_keys: Optional[Iterable[str]] = None,
) -> str:
    """Simulate uploading a telemetry or process file to Signavio.

    Args:
        dataset: Logical dataset / subject slug in Signavio.
        file_path: CSV/XES artifact to send. Relative paths are resolved against the manual stub directory.
        schema_path: Optional Avro schema describing the payload.
        primary_keys: Optional iterable of primary key column names.

    Returns:
        Stub URI representing the uploaded artifact.
    """

    base_dataset = _slug(dataset)
    payload = Path(file_path)
    if not payload.is_absolute():
        payload = _MANUAL_DIR / payload
    _ensure_file(payload)

    if schema_path:
        schema = Path(schema_path)
        if not schema.is_absolute():
            schema = _MANUAL_DIR / schema
        _ensure_file(schema)

    # Primary keys are not used for stub response, but we validate iterable for developer ergonomics.
    if primary_keys is not None:
        list(primary_keys)  # ensures it is materialised / raises if invalid

    return f"{_UPLOAD_PREFIX}/{base_dataset}/{payload.name}"


@tool
def signavio_stub_fetch_view(view_name: str = "ProcessLibrary") -> str:
    """Simulate fetching aggregated analytics or process assets from Signavio.

    Args:
        view_name: Name of the analytical view to fetch.

    Returns:
        JSON string containing stubbed results.
    """

    slugged_name = _slug(view_name)
    library_path = _MANUAL_DIR / "process_library.json"

    if not library_path.exists():
        return json.dumps({
            "view": slugged_name,
            "status": "missing_library",
            "message": f"No process_library.json found under {_MANUAL_DIR}",
        })

    with library_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    return json.dumps({
        "view": slugged_name,
        "source": str(library_path),
        "process_count": len(data.get("processes", [])),
        "processes": data.get("processes", []),
        "stub_uri": f"{_ODATA_PREFIX}/{slugged_name}",
    })
