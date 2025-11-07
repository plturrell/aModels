# Connection Troubleshooting Guide

## Issue: "All connection attempts failed" in /search/unified

### Root Cause

The unified search endpoint attempts to connect to multiple backend services:
- Search Inference Service (port 8090)
- Extract Service / Knowledge Graph (port 9002)
- Catalog Service (port 8084)
- Perplexity AI (external API)

When these services are not running or not accessible, the gateway returns "All connection attempts failed" errors.

### Solution Options

#### Option 1: Start Required Services

Start the backend services that unified search depends on:

```bash
# Search Inference Service
# (Start the search-inference service on port 8090)

# Extract Service (Knowledge Graph)
# (Start the extract service on port 9002)

# Catalog Service
# (Start the catalog service on port 8084)
```

#### Option 2: Configure Service URLs

Set environment variables to point to running services:

```bash
export SEARCH_INFERENCE_URL=http://localhost:8090
export EXTRACT_URL=http://localhost:9002
export CATALOG_URL=http://localhost:8084
```

#### Option 3: Use Mock/Stub Responses (Development)

For development, you can modify the gateway to return mock responses when services are unavailable. The current implementation already handles this gracefully - it returns empty results with error messages in the `sources` object.

### Current Behavior

The unified search endpoint:
1. ✅ Attempts to connect to each service
2. ✅ Catches connection errors gracefully
3. ✅ Returns partial results if some services fail
4. ✅ Includes error messages in response metadata

**Response Format**:
```json
{
  "query": "test",
  "sources": {
    "inference": {"error": "All connection attempts failed"},
    "knowledge_graph": {"error": "All connection attempts failed"},
    "catalog": {"error": "All connection attempts failed"}
  },
  "combined_results": [],
  "metadata": {
    "sources_queried": 3,
    "sources_successful": 0,
    "sources_failed": 3
  }
}
```

### Testing Without Backend Services

The unified search will still work, but will return empty results. This is expected behavior when services are not running.

### Enabling Services

To get actual search results, you need to:

1. **Start Search Inference Service**:
   ```bash
   # Navigate to search-inference service directory
   # Start the service on port 8090
   ```

2. **Start Extract Service**:
   ```bash
   # Navigate to extract service directory
   # Start the service on port 9002
   ```

3. **Start Catalog Service**:
   ```bash
   # Navigate to catalog service directory
   # Start the service on port 8084
   ```

### Health Check Endpoint

Use the gateway health check to see which services are available:

```bash
curl http://localhost:8000/healthz
```

This will show the status of all backend services.

### Recommended Approach

For development/testing:
1. Start gateway service (required)
2. Start shell server (required for frontend)
3. Backend services are optional - unified search will work with partial results

For production:
1. Start all required backend services
2. Configure service URLs via environment variables
3. Monitor health check endpoint

