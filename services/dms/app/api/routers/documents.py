from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.api.dependencies import get_db_session
from app.models.document import Document
from app.schemas.document import DocumentCreate, DocumentRead
from app.services.ingestion import enqueue_ingestion
from app.services.pipeline import orchestrate_document
from app.services.storage import persist_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    payload: DocumentCreate = Depends(DocumentCreate.as_form),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
) -> DocumentRead:
    document = await persist_document(db, payload, file)
    background_tasks.add_task(enqueue_ingestion, document.id, payload.tags)
    background_tasks.add_task(orchestrate_document, document.id)
    return DocumentRead.model_validate(document)


@router.get("/", response_model=List[DocumentRead])
async def list_documents(
    db: AsyncSession = Depends(get_db_session),
) -> List[DocumentRead]:
    """
    List all documents, ordered by creation date (newest first).
    Returns up to 50 most recent documents.
    """
    result = await db.execute(
        select(Document).order_by(Document.created_at.desc()).limit(50)
    )
    documents = result.scalars().all()
    return [DocumentRead.model_validate(doc) for doc in documents]


@router.get("/{document_id}", response_model=DocumentRead)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentRead:
    """
    Get a specific document by ID.
    """
    document = await db.get(Document, document_id)
    if not document:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentRead.model_validate(document)


@router.get("/{document_id}/status")
async def get_document_status(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Get processing status for a document.
    This endpoint calls the orchestration service to get status.
    """
    import httpx
    import os
    
    orchestration_url = os.getenv("ORCHESTRATION_URL", "http://localhost:8080")
    
    # Try to find request_id for this document
    # For now, return basic status - in production, this would query orchestration service
    document = await db.get(Document, document_id)
    if not document:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Check if document has been processed
    status = "pending"
    if document.catalog_identifier:
        status = "processed"
    if document.extraction_summary:
        status = "processed"
    
    return {
        "document_id": document_id,
        "status": status,
        "catalog_identifier": document.catalog_identifier,
        "has_extraction": document.extraction_summary is not None,
        "created_at": document.created_at.isoformat(),
        "updated_at": document.updated_at.isoformat(),
    }


@router.get("/{document_id}/results")
async def get_document_results(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Get processing results for a document.
    This endpoint calls the orchestration service to get results.
    """
    import httpx
    import os
    
    orchestration_url = os.getenv("ORCHESTRATION_URL", "http://localhost:8080")
    
    document = await db.get(Document, document_id)
    if not document:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "document_id": document_id,
        "name": document.name,
        "description": document.description,
        "catalog_identifier": document.catalog_identifier,
        "extraction_summary": document.extraction_summary,
        "storage_path": document.storage_path,
        "created_at": document.created_at.isoformat(),
        "updated_at": document.updated_at.isoformat(),
    }


@router.get("/{document_id}/intelligence")
async def get_document_intelligence(
    document_id: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """
    Get intelligence data for a document.
    This endpoint calls the orchestration service to get intelligence.
    """
    import httpx
    import os
    
    orchestration_url = os.getenv("ORCHESTRATION_URL", "http://localhost:8080")
    
    document = await db.get(Document, document_id)
    if not document:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Try to get intelligence from orchestration service
    # For now, return basic intelligence
    intelligence = {
        "document_id": document_id,
        "catalog_identifier": document.catalog_identifier,
        "extraction_summary": document.extraction_summary,
    }
    
    # Try to query orchestration service for full intelligence
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Find request_id for this document (would need to track this)
            # For now, return basic intelligence
            pass
    except Exception:
        pass
    
    return intelligence
