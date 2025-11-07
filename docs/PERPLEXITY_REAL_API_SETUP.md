# Perplexity Real API Setup ✅

## Status: **REAL API INTEGRATION COMPLETE**

✅ **Orchestration Server**: Created and running  
✅ **Gateway Proxy**: Updated to proxy to orchestration  
✅ **Real Endpoints**: All Perplexity endpoints now use real pipeline  

---

## What Changed

### 1. Created Orchestration Server
**File**: `services/orchestration/cmd/server/main.go`

- HTTP server that exposes all Perplexity API endpoints
- Uses the real `PerplexityHandler` from `services/orchestration/api`
- Runs on port 8080 (configurable via `ORCHESTRATION_PORT`)

### 2. Updated Gateway to Proxy
**File**: `services/gateway/main.py`

- Changed from mock responses to real proxy
- All endpoints now forward to orchestration service
- Added `ORCHESTRATION_URL` environment variable (default: `http://localhost:8080`)

### 3. All Endpoints Now Real

All Perplexity endpoints now use the real pipeline:
- `POST /api/perplexity/process` - Real document processing
- `GET /api/perplexity/status/{id}` - Real status tracking
- `GET /api/perplexity/results/{id}` - Real results
- `GET /api/perplexity/results/{id}/intelligence` - Real intelligence
- `GET /api/perplexity/history` - Real request history
- `POST /api/perplexity/search` - Real search
- Plus all other endpoints (export, batch, cancel, learning, graph, domain, catalog)

---

## Running the Services

### 1. Start Orchestration Server

```bash
cd services/orchestration

# Set API key
export PERPLEXITY_API_KEY="your-api-key"
export ORCHESTRATION_PORT="8080"  # Optional

# Build and run
go build -o bin/orchestration-server ./cmd/server/main.go
./bin/orchestration-server
```

Or directly:
```bash
go run ./cmd/server/main.go
```

### 2. Start Gateway (if not already running)

```bash
cd services/gateway

# Set orchestration URL (if different)
export ORCHESTRATION_URL="http://localhost:8080"  # Optional

# Run
python3 main.py
```

### 3. Access Browser Shell

```
http://localhost:5174
```

---

## Environment Variables

### Orchestration Server
- `PERPLEXITY_API_KEY` - **Required**: Your Perplexity API key
- `ORCHESTRATION_PORT` - Port to listen on (default: 8080)
- `DEEP_RESEARCH_URL` - Deep Research service (default: http://localhost:8085)
- `UNIFIED_WORKFLOW_URL` - Unified Workflow (default: http://graph-service:8081)
- `CATALOG_URL` - Catalog service (default: http://catalog:8080)
- `TRAINING_URL` - Training service (default: http://training:8080)
- `LOCALAI_URL` - Local AI service (default: http://localai:8080)
- `SEARCH_URL` - Search service (default: http://search:8080)
- `EXTRACT_URL` - Extract service (default: http://extract:8081)

### Gateway
- `ORCHESTRATION_URL` - Orchestration service URL (default: http://localhost:8080)

---

## Testing

### 1. Health Check
```bash
curl http://localhost:8080/healthz
```

### 2. Process Documents
```bash
curl -X POST http://localhost:8000/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest AI research",
    "limit": 5
  }'
```

### 3. Check Status
```bash
curl http://localhost:8000/api/perplexity/status/{request_id}
```

---

## Architecture

```
Browser Shell (5174)
    ↓
Gateway (8000)
    ↓
Orchestration Server (8080)
    ↓
Perplexity Pipeline
    ↓
[OCR → Catalog → Training → LocalAI → Search]
```

---

## Summary

✅ **No more mock data!**  
✅ **Real Perplexity API integration**  
✅ **Full pipeline processing**  
✅ **All endpoints functional**  

**The Browser Shell now uses the real Perplexity pipeline!** 🎉

