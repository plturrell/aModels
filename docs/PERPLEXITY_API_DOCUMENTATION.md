# Perplexity Integration API Documentation

## Overview

The Perplexity Integration API provides a comprehensive interface for processing documents from Perplexity through a full pipeline including OCR, catalog registration, training export, local AI storage, and search indexing.

**Base URL**: `http://localhost:8080/api/perplexity`

**OpenAPI Specification**: See `services/orchestration/api/perplexity_openapi.yaml`

---

## Authentication

Currently, authentication is handled via environment variables. Set `PERPLEXITY_API_KEY` to authenticate with the Perplexity API.

---

## Endpoints

### 1. Process Documents

**POST** `/api/perplexity/process`

Process documents from Perplexity API. Supports both synchronous and asynchronous processing.

#### Request Body

```json
{
  "query": "latest research on AI",
  "model": "sonar",
  "limit": 5,
  "include_images": true,
  "async": false,
  "webhook_url": "https://your-app.com/webhook",
  "config": {}
}
```

#### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query for Perplexity API |
| `model` | string | No | Perplexity model (default: "sonar") |
| `limit` | integer | No | Max documents (1-100, default: 10) |
| `include_images` | boolean | No | Include images in processing |
| `async` | boolean | No | Process asynchronously |
| `webhook_url` | string | No | Webhook URL for completion notification |
| `config` | object | No | Additional configuration |

#### Response (Synchronous)

**Status**: `200 OK`

```json
{
  "status": "completed",
  "request_id": "req_1234567890",
  "query": "latest research on AI",
  "statistics": {
    "documents_processed": 5,
    "documents_succeeded": 5,
    "documents_failed": 0,
    "steps_completed": 10
  },
  "processing_time_ms": 1234,
  "document_ids": ["doc_001", "doc_002"],
  "document_count": 2,
  "results": {
    "catalog_url": "/api/catalog/documents?source=perplexity&request_id=req_1234567890",
    "search_url": "/api/search?query=latest research on AI",
    "export_url": "/api/perplexity/results/req_1234567890/export"
  },
  "progress": {
    "current_step": "completed",
    "completed_steps": ["connecting", "extracting", "processing_document_1"],
    "total_steps": 10,
    "progress_percent": 100.0,
    "estimated_time_remaining_ms": 0
  },
  "status_url": "/api/perplexity/status/req_1234567890",
  "results_url": "/api/perplexity/results/req_1234567890"
}
```

#### Response (Asynchronous)

**Status**: `202 Accepted`

```json
{
  "status": "queued",
  "request_id": "req_1234567890",
  "query": "latest research on AI",
  "message": "Job submitted for async processing",
  "status_url": "/api/perplexity/status/req_1234567890",
  "results_url": "/api/perplexity/results/req_1234567890"
}
```

---

### 2. Get Status

**GET** `/api/perplexity/status/{request_id}`

Get the current status and progress of a processing request.

#### Response

**Status**: `200 OK`

```json
{
  "request_id": "req_1234567890",
  "query": "latest research on AI",
  "status": "processing",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:01Z",
  "current_step": "processing_document_2",
  "progress": {
    "current_step": "processing_document_2",
    "completed_steps": ["connecting", "extracting", "processing_document_1"],
    "total_steps": 10,
    "progress_percent": 30.0,
    "estimated_time_remaining_ms": 7000
  },
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 1,
    "documents_failed": 0,
    "steps_completed": 3
  }
}
```

---

### 3. Get Results

**GET** `/api/perplexity/results/{request_id}`

Get all processed documents and results for a request.

#### Response

**Status**: `200 OK`

```json
{
  "request_id": "req_1234567890",
  "query": "latest research on AI",
  "status": "completed",
  "statistics": {
    "documents_processed": 5,
    "documents_succeeded": 5,
    "documents_failed": 0
  },
  "documents": [
    {
      "id": "doc_001",
      "title": "AI Research Paper",
      "status": "success",
      "processed_at": "2024-01-01T12:00:00Z",
      "catalog_id": "doc_001",
      "localai_id": "doc_001",
      "search_id": "doc_001"
    }
  ],
  "results": {
    "catalog_url": "/api/catalog/documents?source=perplexity&request_id=req_1234567890",
    "search_url": "/api/search?query=latest research on AI"
  },
  "processing_time_ms": 1234
}
```

---

### 4. Export Results

**GET** `/api/perplexity/results/{request_id}/export?format={json|csv}`

Export processing results in JSON or CSV format.

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `format` | string | No | Export format: `json` or `csv` (default: `json`) |

#### Response (JSON)

**Status**: `200 OK`  
**Content-Type**: `application/json`  
**Content-Disposition**: `attachment; filename=perplexity_results_{request_id}.json`

#### Response (CSV)

**Status**: `200 OK`  
**Content-Type**: `text/csv`  
**Content-Disposition**: `attachment; filename=perplexity_results_{request_id}.csv`

**CSV Format**:
```csv
Document ID,Title,Status,Processed At,Catalog ID,Training Task ID,LocalAI ID,Search ID,Error
doc_001,AI Research Paper,success,2024-01-01T12:00:00Z,doc_001,doc_001,doc_001,doc_001,
```

---

### 5. Cancel Job

**DELETE** `/api/perplexity/jobs/{request_id}`

Cancel a running or queued processing job.

#### Response

**Status**: `200 OK`

```json
{
  "status": "cancelled",
  "request_id": "req_1234567890",
  "message": "Job cancelled successfully"
}
```

---

### 6. Batch Process

**POST** `/api/perplexity/batch`

Process multiple queries in parallel.

#### Request Body

```json
{
  "queries": [
    {
      "query": "latest research on AI",
      "limit": 5
    },
    {
      "query": "machine learning trends",
      "limit": 3
    }
  ],
  "async": true,
  "webhook_url": "https://your-app.com/webhook"
}
```

#### Response

**Status**: `202 Accepted`

```json
{
  "batch_id": "req_batch_1234567890",
  "total_queries": 2,
  "request_ids": ["req_001", "req_002"],
  "status": "processing",
  "status_url": "/api/perplexity/batch/req_batch_1234567890"
}
```

---

### 7. Get History

**GET** `/api/perplexity/history?limit=50&offset=0&status=completed&query=AI`

Get a list of previous processing requests with filtering and pagination.

#### Query Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `limit` | integer | No | Max results (1-1000, default: 50) |
| `offset` | integer | No | Pagination offset (default: 0) |
| `status` | string | No | Filter by status: `pending`, `processing`, `completed`, `failed`, `partial` |
| `query` | string | No | Search query text |

#### Response

**Status**: `200 OK`

```json
{
  "requests": [
    {
      "request_id": "req_1234567890",
      "query": "latest research on AI",
      "status": "completed",
      "created_at": "2024-01-01T12:00:00Z",
      "completed_at": "2024-01-01T12:00:05Z",
      "document_count": 5
    }
  ],
  "total": 100,
  "limit": 50,
  "offset": 0
}
```

---

### 8. Get Learning Report

**GET** `/api/perplexity/learning/report`

Get comprehensive learning metrics and patterns.

#### Response

**Status**: `200 OK`

```json
{
  "metrics": {
    "total_documents_processed": 100,
    "total_patterns_learned": 50,
    "total_improvements_applied": 25,
    "last_improvement_time": "2024-01-01T12:00:00Z",
    "learning_effectiveness": 0.85
  },
  "learned_patterns": {
    "query_optimization_rules": {},
    "domain_detection_patterns": {}
  },
  "recommendations": [
    "Monitor learning effectiveness over time",
    "Refine pattern extraction logic for higher precision"
  ]
}
```

---

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message",
  "message": "Detailed error description",
  "request_id": "req_1234567890"
}
```

### Error Codes

- `400 Bad Request` - Invalid request parameters
- `404 Not Found` - Request or resource not found
- `500 Internal Server Error` - Server error

### Enhanced Error Details

Errors include recovery steps and retryability:

```json
{
  "errors": [
    {
      "step": "connect",
      "message": "connection timeout",
      "timestamp": "2024-01-01T12:00:00Z",
      "error_code": "TIMEOUT_ERROR",
      "recovery_steps": [
        "Check network connectivity",
        "Verify service URLs are accessible",
        "Retry the request"
      ],
      "retryable": true
    }
  ]
}
```

---

## Webhooks

When `async: true` and `webhook_url` is provided, a webhook notification is sent on job completion.

### Webhook Payload

```json
{
  "request_id": "req_1234567890",
  "status": "completed",
  "query": "latest research on AI",
  "timestamp": "2024-01-01T12:00:00Z",
  "statistics": {
    "documents_processed": 5,
    "documents_succeeded": 5,
    "documents_failed": 0
  }
}
```

### Webhook Headers

- `Content-Type: application/json`
- `User-Agent: Perplexity-Integration/1.0`

---

## Code Examples

### cURL

```bash
# Process documents synchronously
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest research on AI",
    "limit": 5
  }'

# Process asynchronously with webhook
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest research on AI",
    "async": true,
    "webhook_url": "https://your-app.com/webhook"
  }'

# Check status
curl http://localhost:8080/api/perplexity/status/req_1234567890

# Get results
curl http://localhost:8080/api/perplexity/results/req_1234567890

# Export as CSV
curl "http://localhost:8080/api/perplexity/results/req_1234567890/export?format=csv" \
  -o results.csv

# Batch process
curl -X POST http://localhost:8080/api/perplexity/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"query": "AI research"},
      {"query": "machine learning"}
    ],
    "async": true
  }'

# Get history
curl "http://localhost:8080/api/perplexity/history?limit=10&status=completed"
```

### JavaScript (Fetch)

```javascript
// Process documents
const response = await fetch('http://localhost:8080/api/perplexity/process', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: 'latest research on AI',
    async: true,
    webhook_url: 'https://your-app.com/webhook'
  })
});

const result = await response.json();
console.log('Request ID:', result.request_id);

// Poll for status
const statusResponse = await fetch(
  `http://localhost:8080/api/perplexity/status/${result.request_id}`
);
const status = await statusResponse.json();
console.log('Progress:', status.progress.progress_percent + '%');
```

### Python (Requests)

```python
import requests

# Process documents
response = requests.post(
    'http://localhost:8080/api/perplexity/process',
    json={
        'query': 'latest research on AI',
        'async': True,
        'webhook_url': 'https://your-app.com/webhook'
    }
)
result = response.json()
request_id = result['request_id']

# Check status
status = requests.get(
    f'http://localhost:8080/api/perplexity/status/{request_id}'
).json()
print(f"Progress: {status['progress']['progress_percent']}%")

# Get results
results = requests.get(
    f'http://localhost:8080/api/perplexity/results/{request_id}'
).json()
print(f"Documents: {len(results['documents'])}")
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
)

func main() {
    // Process documents
    payload := map[string]interface{}{
        "query": "latest research on AI",
        "async": true,
        "webhook_url": "https://your-app.com/webhook",
    }
    
    jsonData, _ := json.Marshal(payload)
    resp, _ := http.Post(
        "http://localhost:8080/api/perplexity/process",
        "application/json",
        bytes.NewBuffer(jsonData),
    )
    
    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)
    requestID := result["request_id"].(string)
    
    // Check status
    statusResp, _ := http.Get(
        fmt.Sprintf("http://localhost:8080/api/perplexity/status/%s", requestID),
    )
    var status map[string]interface{}
    json.NewDecoder(statusResp.Body).Decode(&status)
    fmt.Printf("Status: %v\n", status["status"])
}
```

---

## Rate Limiting

- Maximum 50 queries per batch request
- Maximum 1000 results per history request
- Webhook queue buffer: 100 notifications

---

## Best Practices

1. **Use Async Processing** for long-running operations
2. **Set Webhook URLs** to avoid polling
3. **Check Status Regularly** when polling (recommended: 2-5 second intervals)
4. **Handle Errors Gracefully** using recovery steps
5. **Export Results** for offline analysis
6. **Use Batch Processing** for multiple queries
7. **Filter History** to find specific requests

---

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
- File: `services/orchestration/api/perplexity_openapi.yaml`
- View online: Use Swagger UI or similar tool

To view in Swagger UI:
```bash
# Install swagger-ui
npm install -g swagger-ui-serve

# Serve the spec
swagger-ui-serve services/orchestration/api/perplexity_openapi.yaml
```

---

## Support

- API Documentation: This file
- OpenAPI Spec: `services/orchestration/api/perplexity_openapi.yaml`
- Integration Guide: `docs/PERPLEXITY_INTEGRATION_REVIEW.md`
- Quick Start: `docs/PERPLEXITY_QUICK_START.md`

