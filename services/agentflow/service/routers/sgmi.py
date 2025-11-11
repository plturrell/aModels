from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status

from ..config import Settings
from ..dependencies import get_settings

router = APIRouter(prefix="/sgmi", tags=["sgmi"])


def _load_json(path: Path) -> object:
    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{path} not found",
        ) from exc
    if not payload.strip():
        return []
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JSON in {path.name}: {exc}",
        ) from exc


def _view_entries(data: object) -> list[dict[str, object]]:
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    if isinstance(data, dict):
        views = data.get("views")
        if isinstance(views, list):
            return [entry for entry in views if isinstance(entry, dict)]
    return []


@router.get("/views")
def list_views(settings: Settings = Depends(get_settings)) -> object:
    """
    Return the full SGMI view lineage registry.
    """
    return _load_json(settings.sgmi_view_lineage_resolved)


@router.get("/views/{view_name}")
def get_view(
    view_name: str,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """
    Return a specific view entry by name (case insensitive).
    """
    data = _load_json(settings.sgmi_view_lineage_resolved)
    candidates = _view_entries(data)
    for entry in candidates:
        if entry.get("view", "").lower() == view_name.lower():
            return entry
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"View '{view_name}' not found in SGMI registry.",
    )


@router.get("/summary")
def get_summary(settings: Settings = Depends(get_settings)) -> object:
    """
    Return aggregated SGMI lineage metrics (information loss, type coverage, etc.).
    """
    return _load_json(settings.sgmi_view_summary_resolved)
