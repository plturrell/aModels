# DMS API Documentation

## Overview

The DMS API provides endpoints for processing documents through the full aModels pipeline, tracking status, and accessing intelligence data.

## Base URL

- **Gateway**: `http://localhost:8000`
- **Orchestration Service**: `http://localhost:8080`
- **DMS Service**: `http://localhost:8096`

## Authentication

Currently, no authentication is required. In production, API keys or OAuth tokens should be used.

## Endpoints

### Process Documents

**POST** `/api/dms/process`

Process one or more documents through the full pipeline.

**Request Body:**
```json
{
  "document_id": "doc_123",
  "document_ids": ["doc_123", "doc_456"],
  "async": true,
  "webhook_url": "https://example.com/webhook",
  "config": {
    "include_content": true
  }
}
```

**Response (202 Accepted - Async):**
```json
{
  "status": "queued",
  "request_id": "req_abc123",
  "message": "Job submitted for async processing",
  "status_url": "/api/dms/status/req_abc123",
  "results_url": "/api/dms/results/req_abc123"
}
```

**Response (200 OK - Sync):**
```json
{
  "status": "completed",
  "request_id": "req_abc123",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 1,
    "documents_failed": 0
  },
  "processing_time_ms": 5000,
  "document_ids": ["doc_123"],
  "progress": {
    "current_step": "completed",
    "progress_percent": 100
  },
  "status_url": "/api/dms/status/req_abc123",
  "results_url": "/api/dms/results/req_abc123",
  "intelligence_url": "/api/dms/results/req_abc123/intelligence"
}
```

### Get Processing Status

**GET** `/api/dms/status/{request_id}`

Get the current processing status for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing document doc_123",
  "status": "processing",
  "created_at": "2024-01-01T00:00:00Z",
  "started_at": "2024-01-01T00:00:01Z",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 0,
    "documents_failed": 0,
    "steps_completed": 3
  },
  "current_step": "processing_doc_123_localai",
  "completed_steps": ["connecting", "extracting", "processing_doc_123_catalog"],
  "total_steps": 6,
  "progress_percent": 50.0,
  "estimated_time_remaining_ms": 5000
}
```

### Get Results

**GET** `/api/dms/results/{request_id}`

Get processing results for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing document doc_123",
  "status": "completed",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 1,
    "documents_failed": 0
  },
  "documents": [
    {
      "id": "doc_123",
      "title": "Document Title",
      "status": "succeeded",
      "processed_at": "2024-01-01T00:00:05Z",
      "catalog_id": "doc_123",
      "local_ai_id": "doc_123",
      "search_id": "doc_123"
    }
  ],
  "results": {
    "catalog_url": "/api/catalog/documents?source=dms&request_id=req_abc123",
    "search_url": "/api/search?query=dms&request_id=req_abc123",
    "export_url": "/api/dms/results/req_abc123/export"
  },
  "intelligence": {
    "domains": ["finance"],
    "total_relationships": 5,
    "total_patterns": 3,
    "knowledge_graph_nodes": 10,
    "knowledge_graph_edges": 5
  }
}
```

### Get Intelligence

**GET** `/api/dms/results/{request_id}/intelligence`

Get detailed intelligence data for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing document doc_123",
  "status": "completed",
  "intelligence": {
    "domains": ["finance"],
    "total_relationships": 5,
    "total_patterns": 3,
    "knowledge_graph_nodes": 10,
    "knowledge_graph_edges": 5,
    "workflow_processed": true,
    "summary": "Document processed successfully"
  },
  "documents": [
    {
      "id": "doc_123",
      "title": "Document Title",
      "intelligence": {
        "domain": "finance",
        "domain_confidence": 0.8,
        "relationships": [
          {
            "type": "related_to",
            "target_id": "doc_456",
            "target_title": "Related Document",
            "strength": 0.75
          }
        ],
        "learned_patterns": [
          {
            "type": "column",
            "description": "date_pattern",
            "metadata": {"format": "YYYY-MM-DD"}
          }
        ]
      }
    }
  ]
}
```

### Get History

**GET** `/api/dms/history`

Get processing request history.

**Query Parameters:**
- `limit` (int, default: 50): Maximum number of requests to return
- `offset` (int, default: 0): Pagination offset
- `status` (string, optional): Filter by status (pending, processing, completed, failed)
- `document_id` (string, optional): Filter by document ID

**Response:**
```json
{
  "requests": [
    {
      "request_id": "req_abc123",
      "query": "Processing document doc_123",
      "status": "completed",
      "created_at": "2024-01-01T00:00:00Z",
      "completed_at": "2024-01-01T00:00:05Z",
      "document_count": 1
    }
  ],
  "total": 100,
  "limit": 50,
  "offset": 0
}
```

### Search Documents

**POST** `/api/dms/search`

Search indexed documents.

**Request Body:**
```json
{
  "query": "financial report",
  "request_id": "req_abc123",
  "top_k": 10,
  "filters": {
    "domain": "finance"
  }
}
```

**Response:**
```json
{
  "query": "financial report",
  "results": [
    {
      "document_id": "doc_123",
      "title": "Financial Report 2024",
      "score": 0.95,
      "content": "..."
    }
  ],
  "count": 1
}
```

### Export Results

**GET** `/api/dms/results/{request_id}/export?format={json|csv}`

Export processing results.

**Query Parameters:**
- `format` (string, default: "json"): Export format (json or csv)

**Response:**
- JSON: Full JSON export
- CSV: CSV file download

### Batch Process

**POST** `/api/dms/batch`

Process multiple documents in batch.

**Request Body:**
```json
{
  "document_ids": ["doc_123", "doc_456", "doc_789"],
  "async": true,
  "webhook_url": "https://example.com/webhook",
  "config": {}
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "total_documents": 3,
  "request_ids": ["req_abc123", "req_def456", "req_ghi789"],
  "status": "processing",
  "status_url": "/api/dms/batch/batch_xyz789"
}
```

### Cancel Job

**DELETE** `/api/dms/jobs/{request_id}`

Cancel a running asynchronous job.

**Response:**
```json
{
  "status": "cancelled",
  "request_id": "req_abc123",
  "message": "Job cancelled successfully"
}
```

### Knowledge Graph Query

**POST** `/api/dms/graph/{request_id}/query`

Query the knowledge graph for a request.

**Request Body:**
```json
{
  "query": "MATCH (d:Document) RETURN d LIMIT 10",
  "params": {}
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "MATCH (d:Document) RETURN d LIMIT 10",
  "results": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### Domain Documents

**GET** `/api/dms/domains/{domain}/documents`

Get documents by domain.

**Query Parameters:**
- `limit` (int, default: 50): Maximum number of documents
- `offset` (int, default: 0): Pagination offset

**Response:**
```json
{
  "domain": "finance",
  "documents": [
    {
      "document_id": "doc_123",
      "title": "Financial Report",
      "content": "..."
    }
  ],
  "count": 1,
  "limit": 50,
  "offset": 0
}
```

### Catalog Search

**POST** `/api/dms/catalog/search`

Perform semantic search on the catalog.

**Request Body:**
```json
{
  "query": "financial data",
  "object_class": "DataProduct",
  "property": "description",
  "source": "dms",
  "filters": {}
}
```

**Response:**
```json
{
  "results": [...],
  "count": 5
}
```

### Get Document

**GET** `/api/dms/documents/{document_id}`

Get a specific document by ID.

**Response:**
```json
{
  "id": "doc_123",
  "name": "Document Name",
  "description": "Document description",
  "storage_path": "/path/to/document",
  "catalog_identifier": "cat_123",
  "extraction_summary": "Summary text",
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:05Z"
}
```

## Error Handling

All endpoints return standard HTTP status codes:
- `200 OK`: Success
- `202 Accepted`: Async job submitted
- `400 Bad Request`: Invalid request
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `502 Bad Gateway`: Orchestration service unavailable

Error responses include detailed error information:
```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {}
}
```

## Rate Limiting

Currently, no rate limiting is enforced. In production, rate limits should be implemented.

## Webhooks

When processing documents asynchronously, you can provide a `webhook_url` to receive notifications when processing completes.

**Webhook Payload:**
```json
{
  "request_id": "req_abc123",
  "status": "completed",
  "timestamp": "2024-01-01T00:00:05Z",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 1,
    "documents_failed": 0
  }
}
```

## Best Practices

1. **Use Async Processing**: For large documents or batch operations
2. **Poll Status**: Check status endpoint periodically for async jobs
3. **Handle Errors**: Implement retry logic for transient errors
4. **Use Webhooks**: Subscribe to webhooks for real-time notifications
5. **Export Results**: Save important results for analysis

