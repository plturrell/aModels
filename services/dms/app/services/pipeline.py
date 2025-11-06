from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.models.document import Document

logger = logging.getLogger(__name__)


async def orchestrate_document(document_id: str) -> None:
    """Run extraction and catalog registration for a stored document."""
    settings = get_settings()
    session_factory = get_session_factory()

    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if not document:
            logger.warning("document %s not found for orchestration", document_id)
            return

        summary_text: Optional[str] = None
        catalog_identifier: Optional[str] = None

        if settings.extract_url:
            try:
                summary_text = await _run_extraction(document, settings.extract_url)
            except Exception as exc:
                logger.error("extraction failed for %s: %s", document_id, exc)

        if settings.catalog_url:
            try:
                catalog_identifier = await _register_catalog(document, summary_text, settings.catalog_url)
            except Exception as exc:
                logger.error("catalog registration failed for %s: %s", document_id, exc)

        if summary_text or catalog_identifier:
            document.extraction_summary = summary_text
            document.catalog_identifier = catalog_identifier
            document.updated_at = datetime.utcnow()
            await session.commit()


async def _run_extraction(document: Document, base_url: str) -> Optional[str]:
    path = Path(document.storage_path)
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except UnicodeDecodeError:
        content = path.read_bytes().decode("latin-1", errors="ignore")

    if not content.strip():
        logger.warning("document %s is empty, skipping extraction", document.id)
        return None

    payload = {
        "document": content,
        "prompt_description": f"Summarise key entities for {document.name}",
        "model_id": ""
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(_join(base_url, "/extract"), json=payload)
        response.raise_for_status()
        data = response.json()

    return _summarise_extraction(data)


def _summarise_extraction(data: dict[str, Any]) -> Optional[str]:
    lines = []
    entities = data.get("entities")
    if isinstance(entities, dict):
        for label, values in list(entities.items())[:4]:
            if not values:
                continue
            excerpt = ", ".join(values[:3])
            lines.append(f"{label}: {excerpt}")

    extractions = data.get("extractions")
    if not lines and isinstance(extractions, list):
        snippets = [entry.get("extraction_text") for entry in extractions[:3] if entry.get("extraction_text")]
        if snippets:
            lines.append("; ".join(snippets))

    if not lines:
        compact = json.dumps(data)[:400]
        return compact
    return "\n".join(lines)


async def _register_catalog(document: Document, summary: Optional[str], base_url: str) -> Optional[str]:
    payload = {
        "topic": document.name,
        "customer_need": summary or document.description or "Document ingested via DMS"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(_join(base_url, "/catalog/data-products/build"), json=payload)
        response.raise_for_status()
        data = response.json()

    product = data.get("data_product") if isinstance(data, dict) else None
    identifier = product.get("identifier") if isinstance(product, dict) else None
    return identifier


def _join(base: str, path: str) -> str:
    return f"{base.rstrip('/')}{path}"
