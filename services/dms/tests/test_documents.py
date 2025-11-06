from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.dependencies import get_db_session
from app.main import create_app
from app.models.document import Base


@pytest_asyncio.fixture
async def test_client(tmp_path: Path):
    settings = SimpleNamespace(
        app_name="Test DMS",
        debug=False,
        storage_root=tmp_path,
        storage_prefix=None,
        postgres_dsn="sqlite+aiosqlite:///:memory:",
        redis_url="redis://localhost:6379/0",
        neo4j_uri="neo4j://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="neo4j"
    )
    settings.storage_root.mkdir(parents=True, exist_ok=True)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", future=True)
    async_session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async def override_session():
        async with async_session() as session:
            yield session

    fake_redis = SimpleNamespace(lpush=AsyncMock(return_value=None))
    fake_driver = AsyncMock()
    fake_driver.close = AsyncMock(return_value=None)

    with patch("app.core.config.get_settings", return_value=settings), \
         patch("app.services.storage.get_settings", return_value=settings), \
         patch("app.services.ingestion.get_redis_client", return_value=fake_redis), \
         patch("app.api.routers.documents.orchestrate_document", new=AsyncMock()), \
         patch("app.core.neo4j.get_neo4j_driver", return_value=fake_driver):
        app = create_app()
        app.dependency_overrides[get_db_session] = override_session
        async with AsyncClient(app=app, base_url="http://test") as client:
            yield client, tmp_path


@pytest.mark.asyncio
async def test_upload_and_list_documents(test_client):
    client, storage_root = test_client

    data = {
        "name": "Integration Spec",
        "description": "Pilot kickoff document",
        "tags": "pilot,sgmi"
    }
    files = {
        "file": ("spec.txt", b"hello world", "text/plain")
    }

    response = await client.post("/documents/", data=data, files=files)
    assert response.status_code == 201
    payload = response.json()
    assert payload["name"] == "Integration Spec"
    stored_path = Path(payload["storage_path"])
    assert stored_path.exists()
    assert stored_path.is_file()
    assert stored_path.resolve().parent == (storage_root / payload["id"]).resolve()

    listing = await client.get("/documents/")
    assert listing.status_code == 200
    documents = listing.json()
    assert len(documents) == 1
    assert documents[0]["id"] == payload["id"]
    assert documents[0]["name"] == payload["name"]
    assert "catalog_identifier" in documents[0]
    assert "extraction_summary" in documents[0]
