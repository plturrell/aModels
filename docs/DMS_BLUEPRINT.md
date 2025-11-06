# Document Management System Blueprint (Go & FastAPI Paths)

The objective is to deliver a document management service that integrates PostgreSQL, Redis, and Neo4j, aligns with aModels’ architecture, and supports AI-led enrichment. Two implementation paths are outlined below so we can choose based on velocity and team familiarity.

---

## Option A — Go Implementation

### Core Technology Choices

| Concern | Recommendation | Notes |
| --- | --- | --- |
| HTTP / routing | `gin-gonic/gin`, `labstack/echo`, or `gofiber/fiber/v2` | Echo already used elsewhere; pick what fits conventions. |
| PostgreSQL | `pgx` driver + `sqlc` | Type-safe queries; metadata, version history, ACLs. |
| Redis | `go-redis/redis/v9` | Session cache, presigned URLs, ingestion queue. |
| Neo4j | `neo4j/neo4j-go-driver/v5` | Document relationships, taxonomy graph. |
| Storage | Local FS + object store (S3/MinIO) | Keep binaries outside the DB; metadata only. |
| Auth | Reuse Layer4 auth middleware | Stay consistent with existing IAM. |

```go
type Document struct {
    ID        string    `json:"id"`
    Name      string    `json:"name"`
    Path      string    `json:"path"`
    Metadata  Metadata  `json:"metadata"`
    CreatedAt time.Time `json:"created_at"`
}

// PostgreSQL -> document metadata, version history
// Neo4j -> relationships, taxonomy
// Redis -> hot cache, job queues
```

### Open-Source References

- **GoDrive** – basic file manager; extend for metadata/graph features.
- **FileStash** – polished web UI; backend can be customized.
- **SFTPGo** – mature auth/storage patterns.
- **Paperless-ngx (Python)** – feature inspiration (OCR, tagging) if reimplemented in Go.

### Architecture (Go Path)

```text
                ┌────────────────────┐
                │    API Gateway     │
                └────────┬───────────┘
                         │ REST/JSON
                ┌────────▼───────────┐
                │   Go DMS Service   │
                └────────┬───────────┘
        ┌───────────┬────────────┴────────────┬───────────┐
        │           │                         │           │
   ┌────▼────┐ ┌────▼────┐               ┌────▼────┐ ┌────▼────┐
   │Postgres│ │Redis    │               │ Neo4j   │ │ Object  │
   │Metadata│ │Cache    │               │Graph    │ │Storage  │
   └────────┘ └─────────┘               └─────────┘ └─────────┘
```

### Next Steps (Go)

1. Scaffold `services/dms` with preferred HTTP framework.
2. Model tables (documents, versions, tags, ACLs) using `sqlc`.
3. Define Neo4j schema (Document, Tag nodes; `RELATED_TO`, `GENERATED_FROM` edges).
4. Implement ingestion pipeline (upload → store → metadata → queue processors).
5. Expose API through Layer4 gateway and integrate RAG hooks (pgvector, embeddings).

---

## Option B — FastAPI Implementation

FastAPI offers faster iteration and lines up with the existing Python-heavy services (LocalAI, Extract, AgentFlow).

### Core Dependencies

| Concern | Recommendation | Notes |
| --- | --- | --- |
| API layer | `fastapi` + `uvicorn[standard]` | Async-first; reuse existing FastAPI patterns. |
| PostgreSQL | `sqlalchemy>=2` (`sqlmodel` optional) | Alembic for migrations; optional `pgvector`. |
| Redis | `redis>=5` (`redis.asyncio`) | Cache, queue, rate limiting. |
| Neo4j | `neo4j>=5` driver | Async driver keeps event loop clean. |
| Storage | Local FS + S3/MinIO (via `boto3` or `minio`) | Store binaries separate from metadata. |
| Background jobs | Celery + Redis, or FastAPI `BackgroundTasks` initially | Celery for OCR, embeddings, notifications. |
| Auth | Layer4 JWT middleware integration | Keep IAM consistent. |

### Suggested Project Layout

```text
services/dms/
├── app/
│   ├── api/
│   │   ├── routers/
│   │   │   ├── documents.py
│   │   │   ├── versions.py
│   │   │   └── search.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py      # SQLAlchemy engine/session
│   │   ├── redis.py         # Redis connection pool
│   │   └── neo4j.py         # AsyncGraphDatabase driver
│   ├── models/              # SQLAlchemy models
│   ├── schemas/             # Pydantic models
│   ├── services/            # storage, graph, search, ingestion
│   ├── tasks/
│   │   ├── celery_app.py
│   │   └── processors.py
│   └── main.py
├── migrations/              # Alembic versions
├── tests/
└── pyproject.toml
```

### SQLModel Example

```python
from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship

class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    description: Optional[str] = None
    storage_path: str
    checksum: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    versions: List["DocumentVersion"] = Relationship(back_populates="document")

class DocumentVersion(SQLModel, table=True):
    id: str = Field(primary_key=True)
    document_id: str = Field(foreign_key="document.id")
    version_index: int
    storage_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    document: Document = Relationship(back_populates="versions")
```

### FastAPI Router Skeleton

```python
from fastapi import APIRouter, Depends, UploadFile, BackgroundTasks
from sqlmodel import Session

from app.api.dependencies import get_db, get_storage_service
from app.schemas.document import DocumentCreate, DocumentRead
from app.services.ingestion import enqueue_ingestion

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/", response_model=DocumentRead, status_code=201)
async def upload_document(
    payload: DocumentCreate,
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    storage = Depends(get_storage_service),
):
    doc = await storage.save(db, payload, file)
    background_tasks.add_task(enqueue_ingestion, doc.id, payload.tags)
    return doc
```

### Neo4j Integration (Async)

```python
from neo4j import AsyncGraphDatabase

async def relate_documents(driver: AsyncGraphDatabase, source_id: str, target_id: str):
    async with driver.session() as session:
        await session.run(
            """
            MERGE (s:Document {id: $source})
            MERGE (t:Document {id: $target})
            MERGE (s)-[:RELATES_TO]->(t)
            """,
            source=source_id,
            target=target_id,
        )
```

### Celery Task Example

```python
from celery import Celery
from app.services import storage, embeddings

celery_app = Celery(__name__)
celery_app.config_from_object("app.tasks.celery_config")

@celery_app.task
def process_document(document_id: str):
    file_bytes = storage.get_file(document_id)
    metadata = embeddings.extract_metadata(file_bytes)
    embeddings.store(document_id, metadata["vector"])
    # Update Postgres + Neo4j as needed
```

### Implementation Phases (FastAPI)

1. **MVP** — CRUD, storage, metadata persistence, REST endpoints.
2. **Integration** — Invoke Extract service for enrichment and register documents with Catalog.
3. **Graph Layer** — Sync relationships/tag taxonomy to Neo4j, expose traversal APIs.
4. **Search & AI** — Add full-text indices, pgvector embeddings, RAG/classification endpoints.
5. **Telemetry & UX** — Integrate with existing observability stack, build admin monitoring.

---

## Shared Feature Checklist

- Document upload/download with versioning.
- Metadata extraction and tagging.
- Search (full-text, metadata filters, optional embeddings).
- Access control (users/groups, integration with existing IAM).
- Document relationships (Neo4j) and taxonomy management.
- Redis-backed caching and background job orchestration.
- AI integration hooks (classification, embeddings, RAG pipelines).

## Integration Hooks

- **Layer4 Gateway** — expose REST/GraphQL endpoints for ingestion and querying.
- **Extract Service** — reuse OCR/entity extraction pipelines for enrichment.
- **LocalAI** — enable contextual Q/A via embeddings.
- **Telemetry** — ship metrics and traces into current dashboards.

---

## Decision Notes

- Go keeps the service homogeneous with the core aModels backend, but requires more upfront scaffolding.
- FastAPI aligns with Python-based services, accelerates AI integration, and leverages existing tooling (Celery, Pydantic, LangChain connectors).
- Keep API contracts and DB schemas language-agnostic so we can pivot if needed without breaking clients.

Whichever route we choose, break the work into phased milestones (ingestion MVP → search → graph enrichment → AI augmentation) and track against the main roadmap.
