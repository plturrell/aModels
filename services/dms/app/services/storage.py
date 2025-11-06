from __future__ import annotations

import shutil
import uuid
from pathlib import Path
from typing import Iterable

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document, DocumentVersion
from app.schemas.document import DocumentCreate


async def persist_document(
    db: AsyncSession,
    payload: DocumentCreate,
    file: UploadFile,
) -> Document:
    """Persist the uploaded document and initial version metadata."""
    settings = get_settings()
    document_id = str(uuid.uuid4())
    version_id = str(uuid.uuid4())

    storage_dir = settings.storage_root / document_id
    storage_dir.mkdir(parents=True, exist_ok=True)

    extension = Path(file.filename or "").suffix
    stored_filename = f"{version_id}{extension}"
    stored_path = storage_dir / stored_filename

    with stored_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    doc = Document(
        id=document_id,
        name=payload.name,
        description=payload.description,
        storage_path=str(stored_path),
    )
    doc.versions.append(
        DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_index=1,
            storage_path=str(stored_path),
        )
    )

    db.add(doc)
    await db.commit()
    await db.refresh(doc)
    return doc
