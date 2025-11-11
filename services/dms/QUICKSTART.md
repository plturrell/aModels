# DMS Quick Start Guide

Get the Document Management Service running in under 5 minutes.

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- PostgreSQL 15+ (for local development without Docker)

## Option 1: Docker Compose (Recommended)

### Step 1: Configure Environment

```bash
cd services/dms

# Copy and edit environment file
cp .env.example .env

# Generate secure passwords
echo "POSTGRES_PASSWORD=$(openssl rand -base64 24)" >> .env
echo "NEO4J_AUTH=neo4j/$(openssl rand -base64 24)" >> .env
echo "DMS_JWT_SECRET=$(openssl rand -hex 32)" >> .env
```

### Step 2: Update DMS_POSTGRES_DSN

Edit `.env` and set:
```bash
DMS_POSTGRES_DSN=postgresql+psycopg://dms_user:YOUR_POSTGRES_PASSWORD@postgres:5432/dms
```

Replace `YOUR_POSTGRES_PASSWORD` with the password from `POSTGRES_PASSWORD`.

### Step 3: Start Services

```bash
docker compose up --build
```

Wait for all health checks to pass (about 30-60 seconds).

### Step 4: Run Migrations

In a new terminal:
```bash
# Set connection string for migration tool
export DMS_POSTGRES_DSN="postgresql+psycopg://dms_user:YOUR_PASSWORD@localhost:5432/dms"

# Run migrations
alembic upgrade head
```

### Step 5: Test

```bash
# Health check
curl http://localhost:8080/healthz

# Upload a document
curl -X POST http://localhost:8080/documents/ \
  -F "name=Test Document" \
  -F "description=My first upload" \
  -F "file=@/path/to/file.txt"

# List documents
curl http://localhost:8080/documents/
```

## Option 2: Local Development

### Step 1: Install Dependencies

```bash
cd services/dms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Start Dependencies

```bash
# Start only the databases
docker compose up postgres redis neo4j
```

### Step 3: Configure

```bash
export DMS_POSTGRES_DSN="postgresql+psycopg://dms_user:password@localhost:5432/dms"
export DMS_REDIS_URL="redis://localhost:6379/0"
export DMS_NEO4J_URI="neo4j://localhost:7687"
export DMS_NEO4J_USER="neo4j"
export DMS_NEO4J_PASSWORD="your-password"
export DMS_STORAGE_ROOT="./data/documents"
```

### Step 4: Run Migrations

```bash
alembic upgrade head
```

### Step 5: Start Service

```bash
uvicorn app.main:app --reload --port 8080
```

## Enabling Authentication

### Development (Optional Auth)

```bash
# In .env or environment
DMS_REQUIRE_AUTH=false  # Default
```

All endpoints accessible without authentication.

### Production (Required Auth)

```bash
# In .env
DMS_REQUIRE_AUTH=true
DMS_JWT_SECRET=$(openssl rand -hex 32)
DMS_API_KEYS=key1,key2,key3
```

### Creating JWT Tokens

```python
from app.core.auth import create_access_token

token = create_access_token(
    data={"sub": "user123", "username": "john@example.com"}
)
print(f"Token: {token}")
```

### Making Authenticated Requests

```bash
# With JWT
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8080/documents/

# With API Key
curl -H "X-API-Key: key1" \
  http://localhost:8080/documents/
```

## Common Tasks

### View Logs

```bash
docker compose logs -f dms
```

### Restart Service

```bash
docker compose restart dms
```

### Stop Everything

```bash
docker compose down
```

### Database Shell

```bash
docker compose exec postgres psql -U dms_user -d dms
```

### Redis Shell

```bash
docker compose exec redis redis-cli
```

### Neo4j Browser

Open http://localhost:7474 in your browser.

## Troubleshooting

### Service won't start - "default credentials" error

**Problem**: Config validation rejected weak passwords.

**Solution**: Update `.env` with secure passwords:
```bash
POSTGRES_PASSWORD=$(openssl rand -base64 24)
NEO4J_AUTH=neo4j/$(openssl rand -base64 24)
```

### Migration failed - connection refused

**Problem**: PostgreSQL not ready yet.

**Solution**: Wait 10 seconds after `docker compose up`, then run migration.

### Health check fails - Neo4j unreachable

**Problem**: Neo4j still initializing.

**Solution**: Check logs:
```bash
docker compose logs neo4j
```

Wait for "Started" message, usually 20-30 seconds.

### Upload fails - authentication required

**Problem**: `DMS_REQUIRE_AUTH=true` but no credentials provided.

**Solution**: Either:
1. Set `DMS_REQUIRE_AUTH=false` for development
2. Or provide authentication headers (see above)

### Cannot find module 'app'

**Problem**: Python can't find app module.

**Solution**: Run from correct directory:
```bash
cd services/dms
PYTHONPATH=$(pwd) uvicorn app.main:app
```

## Next Steps

- Read [README.md](README.md) for full documentation
- Check [SECURITY.md](SECURITY.md) for production deployment
- Review [API documentation](http://localhost:8080/docs) (Swagger UI)
- Configure external services (Extract, Catalog) in `.env`

## Support

- Check logs: `docker compose logs dms`
- Health status: `curl http://localhost:8080/healthz/detailed`
- API docs: http://localhost:8080/docs
