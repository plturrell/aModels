# Orchestration Server

HTTP server that exposes the Perplexity API endpoints and other orchestration services.

## Building

```bash
cd services/orchestration
go build -o bin/orchestration-server ./cmd/server
```

## Running

```bash
# Set required environment variables
export PERPLEXITY_API_KEY="your-api-key"
export ORCHESTRATION_PORT="8080"  # Optional, defaults to 8080

# Run the server
./bin/orchestration-server
```

Or directly with Go:

```bash
go run ./cmd/server
```

## Environment Variables

- `PERPLEXITY_API_KEY` - Required: Your Perplexity API key
- `ORCHESTRATION_PORT` - Optional: Port to listen on (default: 8080)
- `DEEP_RESEARCH_URL` - Optional: Deep Research service URL (default: http://localhost:8085)
- `UNIFIED_WORKFLOW_URL` - Optional: Unified Workflow URL (default: http://graph-service:8081)
- `CATALOG_URL` - Optional: Catalog service URL (default: http://catalog:8080)
- `TRAINING_URL` - Optional: Training service URL (default: http://training:8080)
- `LOCALAI_URL` - Optional: Local AI service URL (default: http://localai:8080)
- `SEARCH_URL` - Optional: Search service URL (default: http://search:8080)
- `EXTRACT_URL` - Optional: Extract service URL (default: http://extract:8081)

## Endpoints

All endpoints are prefixed with `/api/perplexity`:

- `POST /api/perplexity/process` - Process documents
- `GET /api/perplexity/status/{request_id}` - Get status
- `GET /api/perplexity/results/{request_id}` - Get results
- `GET /api/perplexity/results/{request_id}/intelligence` - Get intelligence
- `GET /api/perplexity/history` - Get request history
- `POST /api/perplexity/search` - Search documents
- `GET /api/perplexity/results/{request_id}/export` - Export results
- `POST /api/perplexity/batch` - Batch process
- `DELETE /api/perplexity/jobs/{request_id}` - Cancel job
- `GET /api/perplexity/learning/report` - Get learning report
- `POST /api/perplexity/graph/{request_id}/query` - Query knowledge graph
- `GET /api/perplexity/graph/{request_id}/relationships` - Get relationships
- `GET /api/perplexity/domains/{domain}/documents` - Get domain documents
- `POST /api/perplexity/catalog/search` - Search catalog

## Health Check

- `GET /healthz` - Health check endpoint

