from __future__ import annotations

from typing import List

from fastapi import APIRouter, BackgroundTasks, Depends, File, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db_session
from app.schemas.document import DocumentCreate, DocumentRead
from app.services.ingestion import enqueue_ingestion
from app.services.storage import persist_document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/", response_model=DocumentRead, status_code=status.HTTP_201_CREATED)
async def upload_document(
    payload: DocumentCreate = Depends(DocumentCreate.as_form),
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db_session),
) -> DocumentRead:
    document = await persist_document(db, payload, file)
    background_tasks.add_task(enqueue_ingestion, document.id, payload.tags)
    return DocumentRead.model_validate(document)


@router.get("/", response_model=List[DocumentRead])
async def list_documents(
    db: AsyncSession = Depends(get_db_session),
) -> List[DocumentRead]:
    result = await db.execute("SELECT * FROM documents ORDER BY created_at DESC LIMIT 50")
    rows = result.fetchall()
    return [
        DocumentRead(
            id=row.id,
            name=row.name,
            description=row.description,
            storage_path=row.storage_path,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row in rows
    ]
