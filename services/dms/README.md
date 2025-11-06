# Document Management Service (FastAPI)

This service provides document ingestion, metadata management, and relationship modelling. It exposes a FastAPI application backed by PostgreSQL, Redis, and Neo4j.

## Local setup

### Local Venv

```bash
cd services/dms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

### Docker

```bash
docker build -t amodels-dms:dev services/dms
docker run --rm \
  --network amodels_default \  # adjust to match existing compose network
  -e DMS_POSTGRES_DSN=postgresql+psycopg://postgres:postgres@postgres:5432/dms \
-e DMS_REDIS_URL=redis://redis:6379/0 \
-e DMS_NEO4J_URI=neo4j://neo4j:7687 \
-e DMS_NEO4J_USER=neo4j \
-e DMS_NEO4J_PASSWORD=neo4j \
-e DMS_STORAGE_ROOT=/data/documents \
-e DMS_EXTRACT_URL=http://extract:8081 \
-e DMS_CATALOG_URL=http://catalog:8084 \
  -p 8080:8080 \
  amodels-dms:dev
```

### Docker Compose (local stack)

`docker-compose.yml` in this folder includes Postgres, Redis, and Neo4j for local development:

```bash
cd services/dms
docker compose up --build
```

Environment variables (see `app/core/config.py`) control database connections and storage paths. Ensure the service runs on the same Docker network as Postgres/Redis/Neo4j containers so hostnames resolve.

Key environment variables:

- `DMS_POSTGRES_DSN`
- `DMS_REDIS_URL`
- `DMS_NEO4J_URI`, `DMS_NEO4J_USER`, `DMS_NEO4J_PASSWORD`
- `DMS_STORAGE_ROOT` (local filesystem path)
- `DMS_EXTRACT_URL` (HTTP base for the Extract service, e.g., `http://extract:8081`)
- `DMS_CATALOG_URL` (HTTP base for the Catalog service, e.g., `http://catalog:8084`)
- When `DMS_EXTRACT_URL` is provided the service will first attempt DeepSeek OCR for image uploads before running text extraction and catalog registration.

### Tests

```bash
cd services/dms
PYTHONPATH=$(pwd) pytest
```
