# DMS Port Configuration

## Summary

Added DMS service to main docker-compose.yml with port configuration to avoid conflicts.

**Status:** ✅ **DMS Service Added**

---

## Port Configuration

### Local Development (Docker Compose)

- **DMS Service:** `8096:8080` (host:container)
  - Host port: `8096` (default, configurable via `DMS_PORT` env var)
  - Container port: `8080` (internal FastAPI port)
  - Note: Port 8090 is used by `search-inference` service
- **Graph-Server:** `8080:8080` (host:container)
- **Catalog:** `8084:8084` (host:container)

**Note:** DMS uses port 8090 on host by default to avoid conflict with graph-server on 8080.

### Production/EC2 Configuration

For production environments where DMS should be on port 8080:

**Option 1: Use DMS_PORT environment variable**
```bash
DMS_PORT=8080 docker compose -f infrastructure/docker/brev/docker-compose.yml up -d dms
```

**Option 2: Change graph-server port**
If graph-server is not needed on port 8080, change its port mapping in docker-compose.yml:
```yaml
graph:
  ports:
    - "8081:8080"  # Change from 8080:8080
```

**Option 3: Use reverse proxy**
Configure nginx/traefik to route:
- `/dms/*` → DMS service (port 8080)
- `/graph/*` → Graph-server (port 8080)
- `/catalog/*` → Catalog service (port 8084)

---

## Service Configuration

### Environment Variables

- `DMS_POSTGRES_DSN`: PostgreSQL connection string
- `DMS_REDIS_URL`: Redis connection URL
- `DMS_NEO4J_URI`: Neo4j connection URI
- `DMS_NEO4J_USER`: Neo4j username
- `DMS_NEO4J_PASSWORD`: Neo4j password
- `DMS_STORAGE_ROOT`: Document storage path
- `DMS_EXTRACT_URL`: Extract service URL (defaults to `http://extract-service:8082`)
- `DMS_CATALOG_URL`: Catalog service URL (defaults to `http://catalog:8084`)
- `DMS_PORT`: Host port for DMS (defaults to `8090`)

### Dependencies

- `postgres`: Database for document metadata
- `redis`: Cache and task queue
- `neo4j`: Graph database for relationships
- `extract`: Extract service for OCR and processing
- `catalog`: Catalog service for metadata registration

---

## Browser Shell Configuration

Update browser shell defaults to use correct DMS endpoint:

**Local Development:**
```bash
SHELL_DMS_ENDPOINT=http://localhost:8096 make shell-serve
```

Or use the default (if Makefile is updated):
```bash
make shell-serve  # Uses http://localhost:8096 for DMS
```

**Production/EC2:**
```bash
SHELL_DMS_ENDPOINT=http://ec2-54-197-215-253.compute-1.amazonaws.com:8080 \
SHELL_AGENTFLOW_ENDPOINT=http://ec2-54-197-215-253.compute-1.amazonaws.com:8001 \
make shell-serve
```

---

## Verification

### Check DMS is running:
```bash
docker ps | grep dms-service
```

### Check DMS health:
```bash
curl http://localhost:8096/docs  # Swagger UI (local default)
curl http://localhost:8096/documents  # Documents endpoint
# Or for production (if DMS_PORT=8080):
curl http://localhost:8080/docs  # Swagger UI
curl http://localhost:8080/documents  # Documents endpoint
```

### Check DMS logs:
```bash
docker logs dms-service
```

---

## Next Steps

1. ✅ DMS service added to docker-compose.yml
2. ⏸️ Update browser shell Makefile defaults (if needed)
3. ⏸️ Configure production EC2 to expose DMS on port 8080
4. ⏸️ Test document upload and listing via browser shell

---

**Status:** ✅ **DMS Service Configured**  
**Created:** 2025-11-06

