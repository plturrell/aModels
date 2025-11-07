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
