# Relational API Documentation

## Overview

The Relational API provides endpoints for processing relational database tables through the full aModels pipeline, tracking status, and accessing intelligence data.

## Base URL

- **Gateway**: `http://localhost:8000`
- **Orchestration Service**: `http://localhost:8080`

## Authentication

Currently, no authentication is required. In production, API keys or OAuth tokens should be used.

## Endpoints

### Process Tables

**POST** `/api/relational/process`

Process one or more tables from a relational database.

**Request Body:**
```json
{
  "table": "users",
  "tables": ["users", "orders", "products"],
  "schema": "public",
  "database_url": "postgres://user:pass@localhost/dbname",
  "database_type": "postgres",
  "async": true,
  "webhook_url": "https://example.com/webhook",
  "config": {
    "limit": 1000
  }
}
```

**Response (202 Accepted - Async):**
```json
{
  "status": "queued",
  "request_id": "req_abc123",
  "message": "Job submitted for async processing",
  "status_url": "/api/relational/status/req_abc123",
  "results_url": "/api/relational/results/req_abc123"
}
```

**Response (200 OK - Sync):**
```json
{
  "status": "completed",
  "request_id": "req_abc123",
  "statistics": {
    "documents_processed": 3,
    "documents_succeeded": 3,
    "documents_failed": 0
  },
  "processing_time_ms": 5000,
  "status_url": "/api/relational/status/req_abc123",
  "results_url": "/api/relational/results/req_abc123",
  "intelligence_url": "/api/relational/results/req_abc123/intelligence"
}
```

### Get Processing Status

**GET** `/api/relational/status/{request_id}`

Get the current processing status for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing table users",
  "status": "processing",
  "created_at": "2024-01-01T00:00:00Z",
  "started_at": "2024-01-01T00:00:01Z",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 0,
    "documents_failed": 0,
    "steps_completed": 3
  },
  "current_step": "processing_users_localai",
  "completed_steps": ["connecting", "extracting", "processing_users_catalog"],
  "total_steps": 6,
  "progress_percent": 50.0,
  "estimated_time_remaining_ms": 5000
}
```

### Get Results

**GET** `/api/relational/results/{request_id}`

Get processing results for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing table users",
  "status": "completed",
  "statistics": {
    "documents_processed": 1,
    "documents_succeeded": 1,
    "documents_failed": 0
  },
  "documents": [
    {
      "id": "public.users",
      "title": "Table: users",
      "status": "succeeded",
      "processed_at": "2024-01-01T00:00:05Z",
      "catalog_id": "public.users",
      "local_ai_id": "public.users",
      "search_id": "public.users"
    }
  ],
  "results": {
    "catalog_url": "/api/catalog/documents?source=relational&request_id=req_abc123",
    "search_url": "/api/search?query=relational&request_id=req_abc123",
    "export_url": "/api/relational/results/req_abc123/export"
  },
  "intelligence": {
    "domains": ["general"],
    "total_relationships": 5,
    "total_patterns": 3,
    "knowledge_graph_nodes": 10,
    "knowledge_graph_edges": 5
  }
}
```

### Get Intelligence

**GET** `/api/relational/results/{request_id}/intelligence`

Get detailed intelligence data for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing table users",
  "status": "completed",
  "intelligence": {
    "domains": ["general"],
    "total_relationships": 5,
    "total_patterns": 3,
    "knowledge_graph_nodes": 10,
    "knowledge_graph_edges": 5,
    "workflow_processed": true,
    "summary": "Table processed successfully"
  },
  "documents": [
    {
      "id": "public.users",
      "title": "Table: users",
      "intelligence": {
        "domain": "general",
        "domain_confidence": 0.8,
        "relationships": [
          {
            "type": "foreign_key",
            "target_id": "public.orders",
            "target_title": "Table: orders",
            "strength": 0.75
          }
        ],
        "learned_patterns": [
          {
            "type": "column",
            "description": "email_pattern",
            "metadata": {"format": "email"}
          }
        ]
      }
    }
  ]
}
```

### Get History

**GET** `/api/relational/history`

Get processing request history.

**Query Parameters:**
- `limit` (int, default: 50): Maximum number of requests to return
- `offset` (int, default: 0): Pagination offset
- `status` (string, optional): Filter by status (pending, processing, completed, failed)
- `table` (string, optional): Filter by table name

**Response:**
```json
{
  "requests": [
    {
      "request_id": "req_abc123",
      "query": "Processing table users",
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

### Search Tables

**POST** `/api/relational/search`

Search indexed tables.

**Request Body:**
```json
{
  "query": "user data",
  "request_id": "req_abc123",
  "top_k": 10,
  "filters": {
    "domain": "general"
  }
}
```

**Response:**
```json
{
  "query": "user data",
  "results": [
    {
      "document_id": "public.users",
      "title": "Table: users",
      "score": 0.95,
      "content": "..."
    }
  ],
  "count": 1
}
```

### Export Results

**GET** `/api/relational/results/{request_id}/export?format={json|csv}`

Export processing results.

**Query Parameters:**
- `format` (string, default: "json"): Export format (json or csv)

**Response:**
- JSON: Full JSON export
- CSV: CSV file download

### Batch Process

**POST** `/api/relational/batch`

Process multiple tables in batch.

**Request Body:**
```json
{
  "tables": ["users", "orders", "products"],
  "schema": "public",
  "database_url": "postgres://user:pass@localhost/dbname",
  "database_type": "postgres",
  "async": true,
  "webhook_url": "https://example.com/webhook",
  "config": {}
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "total_tables": 3,
  "request_ids": ["req_abc123", "req_def456", "req_ghi789"],
  "status": "processing",
  "status_url": "/api/relational/batch/batch_xyz789"
}
```

### Cancel Job

**DELETE** `/api/relational/jobs/{request_id}`

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

**POST** `/api/relational/graph/{request_id}/query`

Query the knowledge graph for a request.

**Request Body:**
```json
{
  "query": "MATCH (t:Table) RETURN t LIMIT 10",
  "params": {}
}
```

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "MATCH (t:Table) RETURN t LIMIT 10",
  "results": {
    "nodes": [...],
    "edges": [...]
  }
}
```

### Domain Tables

**GET** `/api/relational/domains/{domain}/tables`

Get tables by domain.

**Query Parameters:**
- `limit` (int, default: 50): Maximum number of tables
- `offset` (int, default: 0): Pagination offset

**Response:**
```json
{
  "domain": "general",
  "tables": [
    {
      "document_id": "public.users",
      "title": "Table: users",
      "content": "..."
    }
  ],
  "count": 1,
  "limit": 50,
  "offset": 0
}
```

### Catalog Search

**POST** `/api/relational/catalog/search`

Perform semantic search on the catalog.

**Request Body:**
```json
{
  "query": "user data",
  "object_class": "DataProduct",
  "property": "description",
  "source": "relational",
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

When processing tables asynchronously, you can provide a `webhook_url` to receive notifications when processing completes.

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

## Database Connection Formats

### PostgreSQL
```json
{
  "database_url": "postgres://user:password@host:5432/dbname",
  "database_type": "postgres"
}
```

### MySQL
```json
{
  "database_url": "mysql://user:password@host:3306/dbname",
  "database_type": "mysql"
}
```

### SQLite
```json
{
  "database_url": "/path/to/database.db",
  "database_type": "sqlite"
}
```

## Best Practices

1. **Use Async Processing**: For large tables or batch operations
2. **Poll Status**: Check status endpoint periodically for async jobs
3. **Handle Errors**: Implement retry logic for transient errors
4. **Use Webhooks**: Subscribe to webhooks for real-time notifications
5. **Export Results**: Save important results for analysis
6. **Limit Rows**: Use configurable row limits for large tables

