# Backend Connection Setup Guide

## Overview

The browser shell frontend connects to backend services through a Go shell server that acts as a proxy. This guide explains how to set up and run the backend services so the frontend can connect.

## Architecture

```
Frontend (React/TypeScript)
    ↓ (API_BASE = "" or shell server URL)
Shell Server (Go) - Port 4173
    ↓ (Proxies requests)
Gateway Service (Python/FastAPI) - Port 8000
    ↓ (Routes to backend services)
Backend Services:
    - Search Inference (Port 8090)
    - Graph Service (Port 8081)
    - Extract Service (Port 9002)
    - Catalog Service (Port 8084)
    - LocalAI (Port 8080)
    - DMS (Port 8080)
    - AgentFlow (Port 9001)
```

## Environment Variables

### Shell Server Configuration

The shell server uses these environment variables to configure proxies:

```bash
# Gateway URL (main backend entry point)
SHELL_GATEWAY_URL=http://localhost:8000
# or
GATEWAY_URL=http://localhost:8000

# Individual service endpoints (optional, defaults to gateway + path)
SHELL_DMS_ENDPOINT=http://localhost:8080
SHELL_AGENTFLOW_ENDPOINT=http://localhost:9001
SHELL_LOCALAI_URL=http://localhost:8080
SHELL_SEARCH_ENDPOINT=http://localhost:8000/search  # or http://localhost:8090

# Frontend API base (empty = relative to shell server)
VITE_SHELL_API=""
```

### Gateway Service Configuration

The gateway service needs these environment variables:

```bash
# Gateway port
GATEWAY_PORT=8000

# Backend service URLs
SEARCH_INFERENCE_URL=http://localhost:8090
GRAPH_SERVICE_URL=http://localhost:8081
EXTRACT_URL=http://localhost:9002
CATALOG_URL=http://localhost:8084
LOCALAI_URL=http://localhost:8080
AGENTFLOW_URL=http://localhost:9001
DEEP_RESEARCH_URL=http://localhost:8085
PERPLEXITY_API_KEY=your_key_here  # Optional
```

## Starting Services

### 1. Start Gateway Service

```bash
cd services/gateway

# Install dependencies
pip install -r requirements.txt

# Start gateway
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or using Python directly:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Start Shell Server

```bash
cd services/browser/shell

# Build shell server (if not already built)
cd cmd/server
go build -o server main.go
cd ../../

# Set environment variables
export SHELL_GATEWAY_URL=http://localhost:8000
export VITE_SHELL_API=""

# Start shell server
./cmd/server/server -addr :4173
```

Or using Make (if Makefile exists):
```bash
make shell-serve
```

### 3. Start Frontend (Development)

```bash
cd services/browser/shell/ui

# Install dependencies (if not done)
npm install

# Set environment variable
export VITE_SHELL_API=""

# Start dev server
npm run dev
```

### 4. Build and Serve Frontend (Production)

```bash
cd services/browser/shell/ui

# Build
npm run build

# The shell server will serve from ui/dist/
# Just start the shell server after building
```

## Proxy Routes

The shell server proxies these routes:

| Frontend Path | Proxy Target | Description |
|--------------|-------------|-------------|
| `/search/*` | `GATEWAY_URL/search/*` or `SEARCH_ENDPOINT/*` | Search endpoints |
| `/dms/*` | `DMS_ENDPOINT/*` | Document Management |
| `/agentflow/*` | `AGENTFLOW_ENDPOINT/*` | AgentFlow/LangFlow |
| `/localai/*` | `LOCALAI_URL/*` | LocalAI chat |

## Testing Connection

### 1. Test Gateway Health

```bash
curl http://localhost:8000/healthz
```

Expected response:
```json
{
  "status": "ok",
  "services": {
    "gateway": "ok",
    ...
  }
}
```

### 2. Test Shell Server Proxy

```bash
# Test search proxy
curl http://localhost:4173/search/unified \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'
```

### 3. Test from Frontend

Open browser console and check:
- `API_BASE` should be empty string (relative) or shell server URL
- Network requests should go to shell server
- Shell server should proxy to gateway

## Troubleshooting

### Issue: Frontend can't connect to backend

**Check**:
1. Is gateway service running? `curl http://localhost:8000/healthz`
2. Is shell server running? `curl http://localhost:4173/api/localai/models`
3. Check `VITE_SHELL_API` - should be empty for relative paths
4. Check browser console for CORS errors

### Issue: 502 Bad Gateway

**Check**:
1. Gateway service is running on port 8000
2. `SHELL_GATEWAY_URL` is set correctly
3. Backend services are accessible from gateway

### Issue: CORS Errors

**Solution**: Gateway should have CORS middleware configured. Check `main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue: Search endpoints not working

**Check**:
1. `SHELL_SEARCH_ENDPOINT` or `SHELL_GATEWAY_URL` is set
2. Gateway `/search/*` endpoints are working
3. Search inference service is running (if used directly)

## Quick Start Script

Create `start-backend.sh`:

```bash
#!/bin/bash

# Start Gateway
cd services/gateway
export GATEWAY_PORT=8000
export SEARCH_INFERENCE_URL=http://localhost:8090
export GRAPH_SERVICE_URL=http://localhost:8081
export EXTRACT_URL=http://localhost:9002
export CATALOG_URL=http://localhost:8084
export LOCALAI_URL=http://localhost:8080
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
GATEWAY_PID=$!

# Start Shell Server
cd ../browser/shell
export SHELL_GATEWAY_URL=http://localhost:8000
export VITE_SHELL_API=""
./cmd/server/server -addr :4173 &
SHELL_PID=$!

echo "Gateway PID: $GATEWAY_PID"
echo "Shell Server PID: $SHELL_PID"
echo "Gateway: http://localhost:8000"
echo "Shell Server: http://localhost:4173"

# Wait for interrupt
trap "kill $GATEWAY_PID $SHELL_PID" EXIT
wait
```

## Docker Compose (Future)

For production, consider using Docker Compose to orchestrate all services:

```yaml
version: '3.8'
services:
  gateway:
    build: ./services/gateway
    ports:
      - "8000:8000"
    environment:
      - GATEWAY_PORT=8000
      - SEARCH_INFERENCE_URL=http://search-inference:8090
      # ... other env vars
  
  shell-server:
    build: ./services/browser/shell
    ports:
      - "4173:4173"
    environment:
      - SHELL_GATEWAY_URL=http://gateway:8000
    depends_on:
      - gateway
```

## Next Steps

1. ✅ Set up environment variables
2. ✅ Start gateway service
3. ✅ Start shell server
4. ✅ Test connections
5. ✅ Verify frontend can make API calls
6. ✅ Test search, narrative, and dashboard generation

