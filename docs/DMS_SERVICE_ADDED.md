# DMS Service Added to Docker Compose

## Summary

✅ **DMS service successfully added to main docker-compose.yml**

The DMS (Document Management Service) FastAPI application is now integrated into the main Docker Compose stack.

---

## Configuration

### Port Mapping

- **Local Development:** `8096:8080` (host:container)
  - Default: `http://localhost:8096`
  - Configurable via `DMS_PORT` environment variable
- **Production/EC2:** Set `DMS_PORT=8080` to use port 8080

### Service Dependencies

- ✅ PostgreSQL (document metadata)
- ✅ Redis (cache and task queue)
- ✅ Neo4j (relationship graph)
- ✅ Extract Service (`http://extract-service:8082`)
- ✅ Catalog Service (`http://catalog:8084`)

### Environment Variables

- `DMS_POSTGRES_DSN`: PostgreSQL connection
- `DMS_REDIS_URL`: Redis connection
- `DMS_NEO4J_URI`, `DMS_NEO4J_USER`, `DMS_NEO4J_PASSWORD`: Neo4j connection
- `DMS_STORAGE_ROOT`: Document storage path
- `DMS_EXTRACT_URL`: Extract service URL (defaults to `http://extract-service:8082`)
- `DMS_CATALOG_URL`: Catalog service URL (defaults to `http://catalog:8084`)
- `DMS_PORT`: Host port (defaults to `8096`)

---

## Usage

### Start DMS Service

```bash
cd /home/aModels
docker compose -f infrastructure/docker/brev/docker-compose.yml up -d dms
```

### Check DMS Status

```bash
# Check if running
docker ps | grep dms-service

# Check logs
docker logs dms-service

# Check Swagger UI
curl http://localhost:8096/docs

# Check documents endpoint
curl http://localhost:8096/documents
```

### Production Configuration

For EC2/production where DMS should be on port 8080:

```bash
DMS_PORT=8080 docker compose -f infrastructure/docker/brev/docker-compose.yml up -d dms
```

**Note:** If graph-server is also using port 8080, you'll need to either:
1. Change graph-server port
2. Use a reverse proxy
3. Run DMS and graph-server on different hosts

---

## Browser Shell Integration

Update browser shell to use DMS:

**Local:**
```bash
SHELL_DMS_ENDPOINT=http://localhost:8096 make shell-serve
```

**Production/EC2:**
```bash
SHELL_DMS_ENDPOINT=http://ec2-54-197-215-253.compute-1.amazonaws.com:8080 \
SHELL_AGENTFLOW_ENDPOINT=http://ec2-54-197-215-253.compute-1.amazonaws.com:8001 \
make shell-serve
```

---

## Next Steps

1. ✅ DMS service added and running
2. ⏸️ Update browser shell Makefile defaults (if needed)
3. ⏸️ Configure production EC2 port 8080
4. ⏸️ Test document upload and listing
5. ⏸️ Verify OCR integration with extract-service
6. ⏸️ Verify catalog registration

---

## Files Modified

- `/home/aModels/infrastructure/docker/brev/docker-compose.yml` - Added DMS service
- `/home/aModels/services/dms/Dockerfile` - Fixed build context paths
- `/home/aModels/docs/DMS_PORT_CONFIGURATION.md` - Port configuration guide

---

**Status:** ✅ **DMS Service Running**  
**Local Port:** `8096` (configurable)  
**Container Port:** `8080`  
**Created:** 2025-11-06

