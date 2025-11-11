# Document Management Service (FastAPI)

This service provides document ingestion, metadata management, and relationship modelling. It exposes a FastAPI application backed by PostgreSQL, Redis, and Neo4j.

## Features

- **Document Upload & Storage**: Multipart file upload with versioning support
- **Multi-database Architecture**: PostgreSQL (metadata), Redis (queuing), Neo4j (relationships)
- **Authentication**: JWT and API key authentication support
- **Health Checks**: `/healthz`, `/healthz/detailed`, `/readyz`, `/livez` endpoints
- **Resilient Integrations**: Circuit breaker, retry logic, correlation IDs
- **Background Processing**: Async OCR, extraction, and catalog registration
- **Database Migrations**: Alembic migrations for schema versioning

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

# Copy example environment file and update with secure credentials
cp .env.example .env
# Edit .env and set secure passwords!

# Run migrations
export DMS_POSTGRES_DSN="postgresql+psycopg://dms_user:YOUR_PASSWORD@localhost:5432/dms"
alembic upgrade head

# Start services
docker compose up --build
```

**IMPORTANT**: Before running docker-compose, update `.env` with secure passwords. The service validates credentials and will refuse to start with default passwords.

Key environment variables:

- `DMS_POSTGRES_DSN`
- `DMS_REDIS_URL`
- `DMS_NEO4J_URI`, `DMS_NEO4J_USER`, `DMS_NEO4J_PASSWORD`
- `DMS_STORAGE_ROOT` (local filesystem path)
- `DMS_EXTRACT_URL` (HTTP base for the Extract service, e.g., `http://extract:8081`)
- `DMS_CATALOG_URL` (HTTP base for the Catalog service, e.g., `http://catalog:8084`)
- When `DMS_EXTRACT_URL` is provided the service will first attempt DeepSeek OCR for image uploads before running text extraction and catalog registration.

## Authentication

The service supports two authentication modes:

### JWT Authentication
```bash
# Enable JWT auth
export DMS_REQUIRE_AUTH=true
export DMS_JWT_SECRET="your-secret-key-here"

# Make authenticated request
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" http://localhost:8080/documents/
```

### API Key Authentication
```bash
# Set valid API keys (comma-separated)
export DMS_API_KEYS="key1,key2,key3"

# Make authenticated request
curl -H "X-API-Key: key1" http://localhost:8080/documents/
```

### Development Mode
By default, authentication is optional (`DMS_REQUIRE_AUTH=false`). Set to `true` in production.

## Database Migrations

The service uses Alembic for database schema versioning:

```bash
# Run all pending migrations
alembic upgrade head

# Create a new migration
alembic revision --autogenerate -m "description"

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history
```

## Health Checks

- **`GET /healthz`**: Basic liveness check (always returns 200 if service is running)
- **`GET /healthz/detailed`**: Full dependency health check (Postgres, Redis, Neo4j)
- **`GET /readyz`**: Kubernetes readiness probe (checks critical dependencies)
- **`GET /livez`**: Kubernetes liveness probe

## Tests

```bash
cd services/dms
PYTHONPATH=$(pwd) pytest
```

## API Endpoints

- `POST /documents/`: Upload a document
- `GET /documents/`: List all documents (paginated)
- `GET /documents/{id}`: Get document details
- `GET /documents/{id}/status`: Get processing status
- `GET /documents/{id}/results`: Get extraction results
- `GET /documents/{id}/intelligence`: Get AI-generated intelligence
