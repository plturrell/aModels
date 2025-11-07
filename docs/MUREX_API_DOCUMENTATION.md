# Murex API Documentation

## Overview

The Murex API provides endpoints for processing Murex trades through the full aModels pipeline, tracking status, accessing intelligence data, and performing ETL to SAP GL.

## Base URL

- **Gateway**: `http://localhost:8000`
- **Orchestration Service**: `http://localhost:8080`

## Authentication

Currently, no authentication is required. In production, API keys or OAuth tokens should be used.

## Endpoints

### Process Trades

**POST** `/api/murex/process`

Process trades from Murex API.

**Request Body:**
```json
{
  "table": "trades",
  "filters": {
    "status": "executed",
    "trade_date_from": "2024-01-01"
  },
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
  "status_url": "/api/murex/status/req_abc123",
  "results_url": "/api/murex/results/req_abc123"
}
```

**Response (200 OK - Sync):**
```json
{
  "status": "completed",
  "request_id": "req_abc123",
  "statistics": {
    "documents_processed": 10,
    "documents_succeeded": 10,
    "documents_failed": 0
  },
  "processing_time_ms": 5000,
  "status_url": "/api/murex/status/req_abc123",
  "results_url": "/api/murex/results/req_abc123",
  "intelligence_url": "/api/murex/results/req_abc123/intelligence"
}
```

### Get Processing Status

**GET** `/api/murex/status/{request_id}`

Get the current processing status for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing Murex trades",
  "status": "processing",
  "created_at": "2024-01-01T00:00:00Z",
  "started_at": "2024-01-01T00:00:01Z",
  "statistics": {
    "documents_processed": 5,
    "documents_succeeded": 3,
    "documents_failed": 0,
    "steps_completed": 3
  },
  "current_step": "processing_trades_localai",
  "completed_steps": ["connecting", "extracting", "processing_trades_catalog"],
  "total_steps": 6,
  "progress_percent": 50.0,
  "estimated_time_remaining_ms": 5000
}
```

### Get Results

**GET** `/api/murex/results/{request_id}`

Get processing results for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing Murex trades",
  "status": "completed",
  "statistics": {
    "documents_processed": 10,
    "documents_succeeded": 10,
    "documents_failed": 0
  },
  "documents": [
    {
      "id": "murex_trades_0",
      "title": "Murex trades: T001",
      "status": "succeeded",
      "processed_at": "2024-01-01T00:00:05Z",
      "catalog_id": "murex_trades_0",
      "local_ai_id": "murex_trades_0",
      "search_id": "murex_trades_0"
    }
  ],
  "results": {
    "catalog_url": "/api/catalog/documents?source=murex&request_id=req_abc123",
    "search_url": "/api/search?query=murex&request_id=req_abc123"
  }
}
```

### Get Intelligence

**GET** `/api/murex/results/{request_id}/intelligence`

Get detailed intelligence data for a request.

**Response:**
```json
{
  "request_id": "req_abc123",
  "query": "Processing Murex trades",
  "status": "completed",
  "intelligence": {
    "domains": ["finance"],
    "total_relationships": 5,
    "total_patterns": 3,
    "knowledge_graph_nodes": 10,
    "knowledge_graph_edges": 5,
    "workflow_processed": true,
    "summary": "Trades processed successfully"
  },
  "documents": [
    {
      "id": "murex_trades_0",
      "title": "Murex trades: T001",
      "intelligence": {
        "domain": "finance",
        "domain_confidence": 0.95,
        "relationships": [],
        "learned_patterns": []
      }
    }
  ]
}
```

### Get History

**GET** `/api/murex/history`

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
      "query": "Processing Murex trades",
      "status": "completed",
      "created_at": "2024-01-01T00:00:00Z",
      "completed_at": "2024-01-01T00:00:05Z",
      "document_count": 10
    }
  ],
  "total": 100,
  "limit": 50,
  "offset": 0
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

## ETL to SAP GL

The Murex pipeline automatically performs ETL transformations to SAP GL:

### Transformation Process

1. **Extract**: Trades extracted from Murex API
2. **Transform**: Fields mapped to SAP GL format
3. **Enrich**: Account mappings applied
4. **Validate**: Data validated against SAP requirements
5. **Load**: Journal entries posted to SAP GL

### Field Mappings

- `trade_id` → `entry_id` (format: JE-{trade_id})
- `trade_date` → `entry_date` (identity)
- `notional_amount` → `debit_amount` (identity)
- `notional_amount` → `credit_amount` (copy)
- `counterparty_id` → `account` (lookup)

## Configuration

### Environment Variables

- `MUREX_BASE_URL`: Murex API base URL (default: https://api.murex.com)
- `MUREX_API_KEY`: Murex API key for authentication
- `MUREX_OPENAPI_SPEC_URL`: OpenAPI specification URL (optional)
- `SAP_GL_URL`: SAP GL endpoint URL for ETL

## Best Practices

1. **Use Async Processing**: For large trade volumes or batch operations
2. **Poll Status**: Check status endpoint periodically for async jobs
3. **Handle Errors**: Implement retry logic for transient errors
4. **Use Webhooks**: Subscribe to webhooks for real-time notifications
5. **Monitor ETL**: Track SAP GL transformation status
6. **Filter Trades**: Use filters to process specific trade subsets

