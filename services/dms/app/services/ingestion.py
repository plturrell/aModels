from __future__ import annotations

import logging

from app.core.redis import get_redis_client

logger = logging.getLogger(__name__)

INGESTION_QUEUE = "dms:ingestion"


async def enqueue_ingestion(document_id: str, tags: list[str]) -> None:
    """Push document identifiers for downstream processing."""
    client = get_redis_client()
    payload = {"document_id": document_id, "tags": tags}
    await client.lpush(INGESTION_QUEUE, str(payload))
    logger.info("Enqueued document %s for ingestion", document_id)
