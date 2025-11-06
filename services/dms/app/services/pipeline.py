from __future__ import annotations

import json
import logging
import base64
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

        ocr_text: Optional[str] = None
        summary_text: Optional[str] = None
        catalog_identifier: Optional[str] = None

        if settings.extract_url:
            try:
                ocr_text = await _run_ocr(document, settings.extract_url)
            except Exception as exc:
                logger.error("ocr failed for %s: %s", document_id, exc)
            try:
                summary_text = await _run_extraction(document, settings.extract_url, ocr_text)
            except Exception as exc:
                logger.error("extraction failed for %s: %s", document_id, exc)

        if settings.catalog_url:
            try:
                payload_summary = summary_text or ocr_text
                catalog_identifier = await _register_catalog(document, payload_summary, settings.catalog_url)
            except Exception as exc:
                logger.error("catalog registration failed for %s: %s", document_id, exc)

        if summary_text or catalog_identifier or ocr_text:
            document.extraction_summary = summary_text or ocr_text
            document.catalog_identifier = catalog_identifier
            document.updated_at = datetime.utcnow()
            await session.commit()


async def _run_extraction(document: Document, base_url: str, initial_text: Optional[str] = None) -> Optional[str]:
    path = Path(document.storage_path)
    if initial_text is not None:
        content = initial_text
    else:
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


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}


async def _run_ocr(document: Document, base_url: str) -> Optional[str]:
    path = Path(document.storage_path)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return None

    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("failed to read document %s for ocr: %s", document.id, exc)
        return None
    if not raw:
        return None

    payload = {
        "image_base64": base64.b64encode(raw).decode("utf-8"),
        "prompt": f"Convert the document {document.name} to markdown."
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(_join(base_url, "/ocr"), json=payload)
        response.raise_for_status()
        payload = response.json()

    text = payload.get("text")
    if isinstance(text, str) and text.strip():
        cleaned = text.strip()
        tables = payload.get("tables")
        if isinstance(tables, list) and tables:
            table_summaries = []
            for table in tables[:3]:
                headers = table.get("headers")
                if isinstance(headers, list) and headers:
                    table_summaries.append(", ".join(headers))
            if table_summaries:
                cleaned += "\n\nTables: " + " | ".join(table_summaries)
        return cleaned
    return None


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
