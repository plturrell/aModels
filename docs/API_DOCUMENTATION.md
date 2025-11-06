# API Documentation

This document provides comprehensive API documentation for all new features implemented in Priority 1-5.

## Table of Contents

1. [AI Agents API](#ai-agents-api)
2. [Digital Twin API](#digital-twin-api)
3. [Discoverability API](#discoverability-api)
4. [Regulatory Specs API](#regulatory-specs-api)

---

## AI Agents API

Base URL: `/api/agents`

### Start Data Ingestion

Start autonomous data ingestion from a source system.

**Endpoint:** `POST /api/agents/ingestion/start`

**Request Body:**
```json
{
  "source_type": "murex|sap_gl|bcrs|rco|axiom",
  "config": {
    "connection_string": "string",
    "additional_config": "any"
  }
}
```

**Response:**
```json
{
  "status": "started",
  "source_type": "murex",
  "message": "Data ingestion started successfully"
}
```

### Get Ingestion Status

Get the status of a running ingestion job.

**Endpoint:** `GET /api/agents/ingestion/{sourceType}/status?source_type=murex`

**Response:**
```json
{
  "source_type": "murex",
  "status": "running|completed|failed",
  "message": "Status information"
}
```

### Learn Mapping Rules

Trigger automatic learning and update of mapping rules.

**Endpoint:** `POST /api/agents/mapping/learn`

**Request Body:**
```json
{
  "patterns": [
    {
      "source_table": "trades",
      "source_columns": ["trade_id", "amount"],
      "target_label": "Trade",
      "target_properties": ["id", "value"],
      "success_count": 10,
      "failure_count": 0
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Mapping rules learning initiated",
  "patterns": 1
}
```

### Detect Anomalies

Run anomaly detection on data points.

**Endpoint:** `POST /api/agents/anomaly/detect`

**Request Body:**
```json
{
  "data_points": [
    {
      "timestamp": "2024-01-01T00:00:00Z",
      "value": 100.0,
      "dimensions": {"field": "value"}
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "anomalies": [
    {
      "id": "anomaly-1",
      "type": "statistical",
      "severity": "high",
      "description": "Value exceeds threshold"
    }
  ],
  "count": 1
}
```

### Generate Test Scenarios

Generate test scenarios for a schema.

**Endpoint:** `POST /api/agents/test/generate`

**Request Body:**
```json
{
  "schema": {
    "fields": [
      {"name": "id", "type": "string"},
      {"name": "value", "type": "number"}
    ]
  },
  "options": {
    "generate_from_schema": true,
    "generate_edge_cases": true,
    "run_tests": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "id": "test-1",
      "name": "Test scenario 1",
      "status": "passed",
      "duration_ms": 100
    }
  ],
  "count": 1
}
```

---

## Digital Twin API

Base URL: `/api/digitaltwin`

### Create Digital Twin

Create a digital twin for a data product.

**Endpoint:** `POST /api/digitaltwin/create`

**Request Body:**
```json
{
  "data_product_id": "product-123",
  "name": "Twin for Product 123"
}
```

**Response:**
```json
{
  "twin": {
    "id": "twin-456",
    "name": "Twin for Product 123",
    "type": "data_product",
    "source_id": "product-123",
    "version": "1.0.0",
    "status": "active"
  },
  "message": "Digital twin created successfully"
}
```

### Start Simulation

Run a simulation on a digital twin.

**Endpoint:** `POST /api/digitaltwin/{id}/simulate`

**Request Body:**
```json
{
  "type": "pipeline",
  "config": {
    "duration": "1s",
    "time_step": "100ms",
    "data_volume": 100
  }
}
```

**Response:**
```json
{
  "simulation": {
    "id": "sim-789",
    "twin_id": "twin-456",
    "type": "pipeline",
    "status": "running"
  },
  "message": "Simulation started successfully"
}
```

### Start Stress Test

Run a stress test on a digital twin.

**Endpoint:** `POST /api/digitaltwin/{id}/stress-test`

**Request Body:**
```json
{
  "config": {
    "duration": "1s",
    "target_rps": 10,
    "max_concurrency": 5,
    "load_profile": {
      "type": "constant",
      "stages": [
        {
          "duration": "1s",
          "target_rps": 10,
          "concurrency": 5
        }
      ]
    }
  }
}
```

**Response:**
```json
{
  "stress_test": {
    "id": "stress-101",
    "twin_id": "twin-456",
    "status": "running"
  },
  "message": "Stress test started successfully"
}
```

### Start Rehearsal

Start a rehearsal mode to validate changes.

**Endpoint:** `POST /api/digitaltwin/{id}/rehearse`

**Request Body:**
```json
{
  "change": {
    "id": "change-1",
    "type": "schema",
    "description": "Add new field",
    "before": {"field": "old"},
    "after": {"field": "new"},
    "priority": "medium",
    "risk": "low"
  },
  "run_simulation": true,
  "run_stress_test": false
}
```

**Response:**
```json
{
  "rehearsal": {
    "id": "rehearsal-202",
    "twin_id": "twin-456",
    "change_id": "change-1",
    "status": "running",
    "recommendation": "safe"
  },
  "message": "Rehearsal started successfully"
}
```

---

## Discoverability API

Base URL: `/api/discover`

### Search

Search across teams for data products.

**Endpoint:** `GET /api/discover/search?q=query&team=team1&team=team2&category=finance&tag=production&sort_by=relevance&limit=20&offset=0`

**Response:**
```json
{
  "results": [
    {
      "id": "product-123",
      "name": "Product Name",
      "team": "team1",
      "category": "finance",
      "tags": ["production"],
      "relevance_score": 0.95
    }
  ],
  "total_count": 1,
  "query": "query",
  "duration_ms": 50
}
```

### Marketplace

List products in the marketplace.

**Endpoint:** `GET /api/discover/marketplace?category=finance&team=team1&sort_by=recent&limit=20&offset=0`

**Response:**
```json
{
  "listings": [
    {
      "product_id": "product-123",
      "name": "Product Name",
      "description": "Product description",
      "team": "team1",
      "category": "finance",
      "tags": ["production"],
      "usage_count": 100,
      "rating": 4.5
    }
  ],
  "count": 1
}
```

### Create Tag

Create a new tag.

**Endpoint:** `POST /api/discover/tags`

**Request Body:**
```json
{
  "name": "production",
  "category": "environment",
  "description": "Production environment tag",
  "parent_tag_id": "optional-parent-id"
}
```

**Response:**
```json
{
  "tag": {
    "id": "tag-123",
    "name": "production",
    "category": "environment",
    "description": "Production environment tag",
    "created_at": "2024-01-01T00:00:00Z",
    "updated_at": "2024-01-01T00:00:00Z"
  },
  "message": "Tag created successfully"
}
```

### Request Access

Request access to a data product.

**Endpoint:** `POST /api/discover/access-request`

**Request Body:**
```json
{
  "product_id": "product-123",
  "requester_id": "user-456",
  "requester_team": "team2",
  "reason": "Need for analysis project",
  "comments": "Additional context"
}
```

**Response:**
```json
{
  "request_id": "request-789",
  "status": "pending",
  "message": "Access request submitted successfully"
}
```

---

## Regulatory Specs API

Base URL: `/api/regulatory`

### Extract MAS 610 Specification

Extract regulatory specification from MAS 610 document.

**Endpoint:** `POST /api/regulatory/extract/mas610`

**Request Body:**
```json
{
  "document_content": "MAS 610 Regulatory Guidelines...",
  "document_source": "mas_610_guidelines.pdf",
  "document_version": "2024.1",
  "user": "user-123"
}
```

**Response:**
```json
{
  "extraction_id": "extract-123",
  "spec": {
    "id": "spec-456",
    "regulatory_type": "mas_610",
    "version": "1.0.0",
    "report_name": "MAS 610 Report",
    "field_count": 50
  },
  "confidence": 0.95,
  "processing_time_ms": 2000,
  "message": "MAS 610 specification extracted successfully"
}
```

### Extract BCBS 239 Specification

Extract regulatory specification from BCBS 239 document.

**Endpoint:** `POST /api/regulatory/extract/bcbs239`

**Request Body:**
```json
{
  "document_content": "BCBS 239 Guidelines...",
  "document_source": "bcbs239_guidelines.pdf",
  "document_version": "2024.1",
  "user": "user-123"
}
```

**Response:**
```json
{
  "extraction_id": "extract-789",
  "spec": {
    "id": "spec-101",
    "regulatory_type": "bcbs239",
    "version": "1.0.0",
    "report_name": "BCBS 239 Report",
    "field_count": 75
  },
  "confidence": 0.92,
  "processing_time_ms": 2500,
  "message": "BCBS 239 specification extracted successfully"
}
```

### Validate Specification

Validate a regulatory specification.

**Endpoint:** `POST /api/regulatory/validate`

**Request Body:**
```json
{
  "spec": {
    "id": "spec-456",
    "regulatory_type": "mas_610",
    "version": "1.0.0",
    "report_structure": {
      "report_name": "MAS 610 Report",
      "report_id": "MAS_610",
      "total_fields": 50,
      "required_fields": 30
    },
    "field_definitions": [
      {
        "field_id": "1.1",
        "field_name": "Total Assets",
        "field_type": "currency",
        "required": true
      }
    ]
  },
  "user": "user-123"
}
```

**Response:**
```json
{
  "valid": true,
  "errors": [],
  "warnings": [],
  "completeness": 1.0,
  "field_count": 50,
  "message": "Validation completed"
}
```

### List Schemas

List all regulatory schemas.

**Endpoint:** `GET /api/regulatory/schemas?regulatory_type=mas_610&limit=100`

**Response:**
```json
{
  "schemas": [
    {
      "id": "spec-456",
      "regulatory_type": "mas_610",
      "version": "1.0.0",
      "report_name": "MAS 610 Report",
      "created_at": "2024-01-01T00:00:00Z"
    }
  ],
  "count": 1
}
```

---

## Error Responses

All endpoints may return the following error responses:

**400 Bad Request:**
```json
{
  "error": "Invalid request",
  "details": "Field 'source_type' is required"
}
```

**404 Not Found:**
```json
{
  "error": "Resource not found",
  "details": "Twin with ID 'twin-456' not found"
}
```

**500 Internal Server Error:**
```json
{
  "error": "Internal server error",
  "details": "Failed to process request"
}
```

---

## Authentication

Currently, authentication is optional and can be enabled via the `ENABLE_AUTH` environment variable. When enabled, protected endpoints require a valid token in the `Authorization` header:

```
Authorization: Bearer <token>
```

---

## Rate Limiting

Rate limiting is not currently implemented but is planned for production readiness.

---

## Versioning

All APIs are currently at version 1. Future versions will be indicated via URL path or headers.

