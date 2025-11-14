# VaultGemma LocalAI v2 API Documentation

**Version:** 2.0.0  
**Status:** ✅ Production Ready  
**Release Date:** November 2025

---

## Overview

The v2 API provides enhanced features over the v1 API, including:

- **Distributed Tracing**: Built-in OpenTelemetry support for request tracking
- **Workflow Tracking**: Associate requests with workflows for better observability
- **Structured Responses**: Consistent, predictable response formats with metadata
- **Enhanced Error Handling**: Detailed error codes and retry guidance
- **Async Support**: Support for asynchronous request processing (coming soon)
- **Batch Operations**: Process multiple requests efficiently (coming soon)

The v1 API remains fully supported for backward compatibility.

---

## Base URL

```
http://localhost:8080/v2
```

---

## Authentication

Currently, no authentication is required. Authentication middleware will be added in a future release.

---

## Endpoints

### 1. Chat Completions

**Endpoint:** `POST /v2/chat/completions`

Create a chat completion with enhanced features including tracing and workflow tracking.

#### Request Body

```json
{
  "model": "auto",
  "messages": [
    {
      "role": "user",
      "content": "Analyze SQL query performance"
    }
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "stream": false,
  "metadata": {
    "user_id": "user_123",
    "session_id": "session_456"
  },
  "workflow_id": "workflow_789",
  "trace_parent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01",
  "domain_filter": ["0x5678-SQLAgent", "0x9ABC-DataAnalysisAgent"]
}
```

#### Request Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model ID or "auto" for automatic domain detection |
| `messages` | array | Yes | Array of message objects with `role` and `content` |
| `max_tokens` | integer | No | Maximum tokens to generate (default: 512) |
| `temperature` | float | No | Sampling temperature 0.0-2.0 (default: 0.7) |
| `top_p` | float | No | Nucleus sampling parameter (default: 0.9) |
| `top_k` | integer | No | Top-k sampling parameter (default: 40) |
| `stream` | boolean | No | Enable streaming responses (default: false) |
| `metadata` | object | No | Additional metadata for the request |
| `workflow_id` | string | No | Workflow ID for tracking related requests |
| `trace_parent` | string | No | W3C Trace Context parent for distributed tracing |
| `domain_filter` | array | No | Limit domain detection to specific domains |
| `async` | boolean | No | Enable async processing (coming soon) |
| `callback_url` | string | No | Callback URL for async results (coming soon) |

#### Response

```json
{
  "id": "v2_req_1730000000000",
  "object": "chat.completion",
  "created": 1730000000,
  "model": "0x5678-SQLAgent",
  "domain": "0x5678-SQLAgent",
  "status": "completed",
  "request_time": "2025-11-14T19:00:00Z",
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  "workflow_id": "workflow_789",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "To analyze SQL query performance..."
      },
      "finish_reason": "stop",
      "metadata": {
        "latency_ms": 234,
        "backend_type": "gguf",
        "cache_hit": false,
        "domain_name": "SQL Agent"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 25,
    "completion_tokens": 150,
    "total_tokens": 175
  },
  "metadata": {
    "latency_ms": 234,
    "backend_type": "gguf",
    "cache_hit": false,
    "domain_name": "SQL Agent",
    "fallback_used": false
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique request identifier |
| `object` | string | Object type: "chat.completion" |
| `created` | integer | Unix timestamp of request creation |
| `model` | string | Model ID that processed the request |
| `domain` | string | Domain that handled the request |
| `status` | string | Request status: "completed", "failed", "pending" |
| `request_time` | string | ISO 8601 timestamp of request |
| `trace_id` | string | OpenTelemetry trace ID for distributed tracing |
| `workflow_id` | string | Workflow ID if provided in request |
| `choices` | array | Array of completion choices |
| `usage` | object | Token usage statistics |
| `metadata` | object | Additional response metadata |

#### Error Response

```json
{
  "error": {
    "code": "model_not_found",
    "message": "Failed to resolve model: model not available",
    "type": "api_error",
    "trace_id": "0af7651916cd43dd8448eb211c80319c",
    "retry_after": 60
  }
}
```

#### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Malformed request body or parameters |
| `model_not_found` | 503 | Requested model is not available |
| `inference_error` | 502 | Error during model inference |
| `rate_limit_exceeded` | 429 | Too many requests |
| `timeout` | 408 | Request timeout |
| `internal_error` | 500 | Internal server error |

---

### 2. List Models

**Endpoint:** `GET /v2/models`

List all available models with enhanced metadata.

#### Response

```json
{
  "object": "list",
  "version": "v2",
  "count": 24,
  "data": [
    {
      "id": "0x5678-SQLAgent",
      "name": "SQL Agent",
      "layer": "layer1",
      "team": "DataTeam",
      "backend": "gguf",
      "tags": ["sql", "database", "query"],
      "keywords": ["sql", "select", "database"],
      "max_tokens": 2048,
      "temperature": 0.1
    }
  ]
}
```

---

### 3. Health Check

**Endpoint:** `GET /v2/health`

Get service health status with v2 features.

#### Response

```json
{
  "status": "ok",
  "version": "v2",
  "timestamp": "2025-11-14T19:00:00Z",
  "service": {
    "name": "vaultgemma-localai",
    "version": "2.0.0",
    "uptime": "2h34m12s"
  },
  "metrics": {
    "requests_total": 1234
  },
  "features": [
    "distributed_tracing",
    "workflow_tracking",
    "async_requests",
    "batch_operations",
    "enhanced_errors"
  ],
  "profiler": {
    "uptime_seconds": 9252,
    "request_count": 1234,
    "error_count": 5,
    "error_rate": 0.004,
    "latency": {
      "avg_ms": 245,
      "min_ms": 45,
      "max_ms": 1203,
      "p50_ms": 210,
      "p95_ms": 580,
      "p99_ms": 980
    }
  }
}
```

---

## Distributed Tracing

The v2 API includes built-in OpenTelemetry support for distributed tracing.

### Enabling Tracing

Set the following environment variables:

```bash
export OTEL_TRACING_ENABLED=1
export OTEL_EXPORTER_JAEGER_ENDPOINT=http://localhost:14268/api/traces
```

### Trace Context Propagation

You can propagate trace context from upstream services using the W3C Trace Context format:

```json
{
  "trace_parent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
}
```

The response will include the `trace_id` for correlation:

```json
{
  "trace_id": "0af7651916cd43dd8448eb211c80319c",
  ...
}
```

---

## Workflow Tracking

Associate requests with workflows for better observability:

```bash
curl -X POST http://localhost:8080/v2/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Analyze sales data"}],
    "workflow_id": "sales-analysis-2024-q4"
  }'
```

All requests with the same `workflow_id` can be tracked together in your observability platform.

---

## Migration from v1 to v2

### Key Differences

| Feature | v1 API | v2 API |
|---------|--------|--------|
| **Response Format** | OpenAI-compatible | Enhanced with metadata |
| **Error Format** | Simple message | Structured with error codes |
| **Tracing** | Not available | Built-in OpenTelemetry |
| **Workflow Tracking** | Not available | Native support |
| **Metadata** | Limited | Comprehensive |
| **Async Support** | Not available | Coming soon |

### Migration Steps

1. **Update endpoint URLs**: Change `/v1/*` to `/v2/*`
2. **Update response parsing**: Handle new response structure
3. **Add workflow tracking**: Include `workflow_id` in requests
4. **Enable tracing**: Set `OTEL_TRACING_ENABLED=1`
5. **Update error handling**: Handle structured error responses

### Example Migration

**v1 Request:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'
```

**v2 Request:**
```bash
curl -X POST http://localhost:8080/v2/chat/completions \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}],
    "workflow_id": "my-workflow",
    "metadata": {"user_id": "user_123"}
  }'
```

---

## Best Practices

1. **Use workflow IDs**: Always include `workflow_id` for better request tracking
2. **Enable tracing**: Set up OpenTelemetry for production deployments
3. **Handle errors**: Check error codes and implement retry logic
4. **Monitor performance**: Use the `/v2/health` endpoint for health checks
5. **Set metadata**: Include relevant context in request metadata

---

## Rate Limiting

Rate limiting is applied at 10 requests per second per client. When rate limited, you'll receive a 429 status code with `retry_after` in seconds.

---

## Support

For issues or questions:
- GitHub Issues: [aModels Repository]
- Documentation: `/home/aModels/services/localai/README.md`
- API v1 Docs: For legacy API documentation
