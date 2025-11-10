from __future__ import annotations

import json
import logging
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.database import get_session_factory
from app.core.http_client import ResilientHTTPClient
from app.core.metrics import collect_integration_metric
from app.core.middleware import get_correlation_id
from app.models.document import Document

logger = logging.getLogger(__name__)

# Global HTTP clients (initialized on first use)
_extract_client: Optional[ResilientHTTPClient] = None
_catalog_client: Optional[ResilientHTTPClient] = None


async def orchestrate_document(document_id: str) -> None:
    """Run extraction and catalog registration for a stored document.
    
    Optionally calls orchestration service for full pipeline processing.
    """
    import os
    
    settings = get_settings()
    session_factory = get_session_factory()

    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if not document:
            logger.warning("document %s not found for orchestration", document_id)
            return

        # Check if orchestration service is available and should be used
        orchestration_url = os.getenv("ORCHESTRATION_URL", "http://localhost:8080")
        use_orchestration = os.getenv("DMS_USE_ORCHESTRATION", "false").lower() == "true"
        
        if use_orchestration:
            # Call orchestration service for full pipeline processing
            try:
                import httpx
                async with httpx.AsyncClient(timeout=300.0) as client:
                    payload = {
                        "document_id": document_id,
                        "async": True,  # Process asynchronously
                    }
                    response = await client.post(
                        f"{orchestration_url}/api/dms/process",
                        json=payload
                    )
                    if response.status_code == 202:
                        # Job submitted successfully
                        data = response.json()
                        request_id = data.get("request_id")
                        logger.info("Document %s submitted to orchestration service with request_id %s", document_id, request_id)
                        # Store request_id for tracking (could add to document metadata)
                        return
                    else:
                        logger.warning("Orchestration service returned status %d, falling back to local processing", response.status_code)
            except Exception as exc:
                logger.warning("Failed to call orchestration service (non-fatal): %s", exc)
                # Fall through to local processing

        # Local processing (original logic)
        ocr_text: Optional[str] = None
        summary_text: Optional[str] = None
        catalog_identifier: Optional[str] = None

        # Build context for integration calls
        context = {
            "correlation_id": f"dms-{document_id}",
        }
        
        # Try to get correlation ID from request if available
        # (This would be set by middleware in a real request context)
        
        if settings.extract_url:
            try:
                ocr_text = await _run_ocr(document, settings.extract_url, context)
            except Exception as exc:
                logger.error("ocr failed for %s: %s", document_id, exc)
            try:
                summary_text = await _run_extraction(document, settings.extract_url, ocr_text, context)
            except Exception as exc:
                logger.error("extraction failed for %s: %s", document_id, exc)

        if settings.catalog_url:
            try:
                payload_summary = summary_text or ocr_text
                catalog_identifier = await _register_catalog(document, payload_summary, settings.catalog_url, context)
            except Exception as exc:
                logger.error("catalog registration failed for %s: %s", document_id, exc)

        if summary_text or catalog_identifier or ocr_text:
            document.extraction_summary = summary_text or ocr_text
            document.catalog_identifier = catalog_identifier
            document.updated_at = datetime.utcnow()
            await session.commit()


async def _run_extraction(
    document: Document,
    base_url: str,
    initial_text: Optional[str] = None,
    context: Optional[dict] = None,
) -> Optional[str]:
    """Run text extraction using resilient HTTP client."""
    global _extract_client
    
    if context is None:
        context = {}
    
    # Initialize client if needed
    if _extract_client is None or _extract_client.base_url != base_url:
        from app.core.metrics import collect_integration_metric
        _extract_client = ResilientHTTPClient(
            base_url=base_url,
            timeout=30.0,
            max_retries=3,
            metrics_collector=collect_integration_metric,
        )
    
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
    
    # Validate response structure
    def validate_response(data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Response is not a dictionary")
        # Response should have entities or extractions
        if "entities" not in data and "extractions" not in data:
            logger.warning("Response missing expected fields: %s", list(data.keys()))

    try:
        data = await _extract_client.post_json("/extract", payload, context, validate_response)
        return _summarise_extraction(data)
    except Exception as e:
        logger.error("Extraction failed: %s", e)
        raise


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}


async def _run_ocr(
    document: Document,
    base_url: str,
    context: Optional[dict] = None,
) -> Optional[str]:
    """Run OCR using resilient HTTP client."""
    global _extract_client
    
    if context is None:
        context = {}
    
    # Initialize client if needed
    if _extract_client is None or _extract_client.base_url != base_url:
        from app.core.metrics import collect_integration_metric
        _extract_client = ResilientHTTPClient(
            base_url=base_url,
            timeout=60.0,
            max_retries=3,
            metrics_collector=collect_integration_metric,
        )
    
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
    
    # Validate response structure
    def validate_response(data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Response is not a dictionary")
        if "text" not in data:
            raise ValueError("Response missing 'text' field")

    try:
        response_data = await _extract_client.post_json("/ocr", payload, context, validate_response)
        
        text = response_data.get("text")
        if isinstance(text, str) and text.strip():
            cleaned = text.strip()
            tables = response_data.get("tables")
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
    except Exception as e:
        logger.error("OCR failed: %s", e)
        raise


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


async def _register_catalog(
    document: Document,
    summary: Optional[str],
    base_url: str,
    context: Optional[dict] = None,
) -> Optional[str]:
    """Register document in catalog using resilient HTTP client."""
    global _catalog_client
    
    if context is None:
        context = {}
    
    # Initialize client if needed
    if _catalog_client is None or _catalog_client.base_url != base_url:
        from app.core.metrics import collect_integration_metric
        _catalog_client = ResilientHTTPClient(
            base_url=base_url,
            timeout=30.0,
            max_retries=3,
            metrics_collector=collect_integration_metric,
        )
    
    payload = {
        "topic": document.name,
        "customer_need": summary or document.description or "Document ingested via DMS"
    }
    
    # Validate response structure
    def validate_response(data: dict) -> None:
        if not isinstance(data, dict):
            raise ValueError("Response is not a dictionary")
        # Response should have data_product with identifier
        data_product = data.get("data_product")
        if not isinstance(data_product, dict):
            raise ValueError("Response missing 'data_product' field")
        if "identifier" not in data_product:
            raise ValueError("Response missing 'identifier' in data_product")

    try:
        data = await _catalog_client.post_json("/catalog/data-products/build", payload, context, validate_response)
        product = data.get("data_product") if isinstance(data, dict) else None
        identifier = product.get("identifier") if isinstance(product, dict) else None
        return identifier
    except Exception as e:
        logger.error("Catalog registration failed: %s", e)
        raise


# Removed _join function - using urljoin in ResilientHTTPClient instead
