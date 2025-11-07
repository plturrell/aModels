# Starting the Perplexity API Backend

## Issue

The Browser Shell is working, but the Perplexity API backend is not running. You're seeing:
```
ERR_CONNECTION_REFUSED on port 8080
```

---

## Solution: Start the API Backend

The Perplexity API is part of the orchestration service. You need to start the orchestration API server.

### Option 1: Check if Orchestration Service is Running

The Perplexity API endpoints are likely part of a larger orchestration/gateway service. Check:

1. **Gateway Service**: Usually runs on port 8000 or 8080
2. **Orchestration Service**: May have its own server

### Option 2: Start the Service

```bash
# Navigate to orchestration service
cd services/orchestration

# Check for main.go or server file
ls -la *.go

# Or check for docker-compose
cd ../..
ls -la infrastructure/docker/compose.yml
```

---

## Quick Fix: Mock/Disable API Calls

If you just want to see the UI working without the backend:

1. **Open Browser DevTools** (F12)
2. **Go to Network tab**
3. **Check "Disable cache"**
4. The UI should still load, just without data

---

## Expected API Endpoints

The Perplexity module expects these endpoints on `http://localhost:8080`:

- `GET /api/perplexity/history` - Request history
- `GET /api/perplexity/status/{id}` - Request status
- `GET /api/perplexity/results/{id}` - Results
- `GET /api/perplexity/results/{id}/intelligence` - Intelligence data
- `POST /api/perplexity/process` - Submit new query
- `POST /api/perplexity/search` - Search documents

---

## Configuration

The API base URL is configured in:
- Environment variable: `VITE_PERPLEXITY_API_BASE`
- Default: `http://localhost:8080`
- Can be changed in `.env` file in `services/browser/shell/ui/`

---

## Next Steps

1. **Find the orchestration/gateway service** that hosts the Perplexity API
2. **Start that service** on port 8080
3. **Or change the API URL** in `.env` to point to where the service actually runs

---

**The Browser Shell UI is working! Just need the API backend running.** ✅

