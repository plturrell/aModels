# Backend Connection Testing Guide

## Prerequisites

1. ✅ Gateway dependencies installed: `pip install -r requirements.txt`
2. ✅ Shell server built: `go build -o server cmd/server/main.go`
3. ✅ Frontend built: `npm run build` (in ui/)

## Step-by-Step Testing

### Step 1: Start Gateway Service

```bash
cd services/gateway
export GATEWAY_PORT=8000
export SEARCH_INFERENCE_URL=http://localhost:8090
export GRAPH_SERVICE_URL=http://localhost:8081
export EXTRACT_URL=http://localhost:9002
export CATALOG_URL=http://localhost:8084
export LOCALAI_URL=http://localhost:8080

# Start gateway
./start.sh
# Or: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Verify**: 
```bash
curl http://localhost:8000/healthz
```

Expected: JSON response with service status

### Step 2: Start Shell Server

```bash
cd services/browser/shell
export SHELL_GATEWAY_URL=http://localhost:8000
export VITE_SHELL_API=""

# Start shell server
./cmd/server/server -addr :4173
```

**Verify**:
```bash
curl http://localhost:4173/api/localai/models
```

Expected: JSON response or error (if LocalAI not running, that's OK)

### Step 3: Test Search Proxy

```bash
# Test unified search through shell server
curl http://localhost:4173/search/unified \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5}'
```

Expected: Search results or error message (if search services not running)

### Step 4: Test Frontend Connection

1. Open browser: `http://localhost:4173`
2. Open browser console (F12)
3. Check `API_BASE` value (should be empty string)
4. Try a search query
5. Check network tab for requests to `/search/unified`

### Step 5: Test Narrative Generation

```bash
curl http://localhost:4173/search/narrative \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "search_results": {
      "combined_results": [
        {"source": "test", "id": "1", "content": "Test content", "score": 0.9}
      ],
      "metadata": {"sources_queried": 1, "sources_successful": 1}
    }
  }'
```

### Step 6: Test Dashboard Generation

```bash
curl http://localhost:4173/search/dashboard \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "query": "test query",
    "search_results": {
      "combined_results": [
        {"source": "test", "id": "1", "content": "Test content", "score": 0.9}
      ],
      "visualization": {
        "source_distribution": {"test": 1},
        "score_statistics": {"average": 0.9, "min": 0.9, "max": 0.9, "count": 1}
      }
    }
  }'
```

## Troubleshooting

### Gateway Not Starting

**Error**: `ModuleNotFoundError: No module named 'fastapi'`

**Solution**:
```bash
cd services/gateway
pip install -r requirements.txt
```

### Shell Server Not Starting

**Error**: `bind: address already in use`

**Solution**:
```bash
# Find process using port 4173
lsof -i :4173
# Kill it
kill -9 <PID>
```

### 502 Bad Gateway

**Cause**: Gateway not running or wrong URL

**Solution**:
1. Check gateway is running: `curl http://localhost:8000/healthz`
2. Check `SHELL_GATEWAY_URL` environment variable
3. Check shell server logs

### CORS Errors

**Cause**: Frontend trying to connect directly to gateway

**Solution**:
- Use shell server as proxy (set `VITE_SHELL_API=""`)
- Or ensure gateway CORS allows frontend origin

### Search Endpoints Not Working

**Cause**: Search services not running or wrong URLs

**Solution**:
1. Check `SEARCH_INFERENCE_URL` in gateway
2. Check search inference service is running
3. Gateway will gracefully handle missing services

## Expected Behavior

### With All Services Running

- ✅ Search returns results from multiple sources
- ✅ Narrative generation works
- ✅ Dashboard generation works
- ✅ Charts render in UI
- ✅ Export to PowerPoint works

### With Partial Services

- ⚠️ Search may return partial results
- ⚠️ Some sources may show errors
- ✅ Gateway handles missing services gracefully
- ✅ UI shows error messages for failed sources

## Next Steps After Testing

1. ✅ Verify all endpoints respond
2. ✅ Test with real search queries
3. ✅ Test narrative generation
4. ✅ Test dashboard generation
5. ✅ Test PowerPoint export
6. ✅ Test chart rendering in UI

