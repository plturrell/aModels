# Telemetry Exporter Service

Production-ready service for exporting agent telemetry to Signavio Process Intelligence via real APIs. No mocks, no test files - production integration only.

## Overview

The Telemetry Exporter Service provides on-demand API endpoints to:
- Export agent execution telemetry from multiple sources to Signavio
- Discover agent sessions from Extract service and agent telemetry service
- Track export status for idempotency
- Support unified workflow from multiple telemetry sources

## Architecture

- **Unified Discovery**: Coordinates telemetry retrieval from multiple sources
- **Signavio Exporter**: Handles upload to Signavio Ingestion API
- **Telemetry Formatter**: Converts agent telemetry to Signavio format
- **REST API**: On-demand export endpoints

## Configuration

### Environment Variables

```bash
# Server
TELEMETRY_EXPORTER_PORT=8085

# Signavio
SIGNAVIO_API_URL=https://ingestion-eu.signavio.com
SIGNAVIO_API_KEY=your_api_key
SIGNAVIO_TENANT_ID=your_tenant_id
SIGNAVIO_DATASET=agent-telemetry
SIGNAVIO_TIMEOUT=30s
SIGNAVIO_MAX_RETRIES=3

# Source Services
EXTRACT_SERVICE_URL=http://localhost:8081
AGENT_METRICS_BASE_URL=http://localhost:9000  # Optional

# Agent Identification
AGENT_NAME=my-agent  # Optional, defaults to hostname
SERVICE_NAME=my-service  # Alternative to AGENT_NAME
```

### Required Configuration

- `SIGNAVIO_API_KEY`: Signavio API key (required)
- `SIGNAVIO_DATASET`: Dataset name for Signavio (required)
- `EXTRACT_SERVICE_URL`: Extract service URL (required)

## API Endpoints

### Export Single Session

```bash
POST /export/session/{session-id}?source=extract|agent_telemetry

# Optional: Override dataset
curl -X POST http://localhost:8085/export/session/abc123?source=extract \
  -H "Content-Type: application/json" \
  -d '{"dataset": "custom-dataset"}'
```

**Response:**
```json
{
  "status": "success",
  "session_id": "abc123",
  "source": "extract",
  "exported_at": "2025-01-15T10:30:00Z"
}
```

### Export Batch

```bash
POST /export/batch

curl -X POST http://localhost:8085/export/batch \
  -H "Content-Type: application/json" \
  -d '{
    "session_ids": ["session1", "session2", "session3"],
    "source": "extract",
    "dataset": "agent-telemetry"
  }'
```

**Response:**
```json
{
  "status": "success",
  "total_count": 3,
  "exported_count": 3,
  "exported_at": "2025-01-15T10:30:00Z",
  "results": [
    {"session_id": "session1", "status": "ready"},
    {"session_id": "session2", "status": "ready"},
    {"session_id": "session3", "status": "ready"}
  ]
}
```

### Check Export Status

```bash
GET /export/status/{session-id}

curl http://localhost:8085/export/status/abc123
```

**Response:**
```json
{
  "session_id": "abc123",
  "exported": true
}
```

### List Sessions

```bash
GET /export/sessions

curl http://localhost:8085/export/sessions
```

**Response:**
```json
{
  "sessions": [],
  "message": "Session discovery not yet implemented - provide session IDs manually"
}
```

### Health Check

```bash
GET /health

curl http://localhost:8085/health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "telemetry-exporter",
  "signavio": "connected"
}
```

## Usage

### Build

```bash
cd services/telemetry-exporter
go build -o bin/telemetry-exporter ./cmd/telemetry-exporter
```

### Run

```bash
# Using environment variables
export SIGNAVIO_API_KEY=your_key
export SIGNAVIO_DATASET=agent-telemetry
export EXTRACT_SERVICE_URL=http://localhost:8081
./bin/telemetry-exporter

# Or with command-line flags
./bin/telemetry-exporter -port 8085
```

### Export Agent Telemetry

```bash
# Export a single session from Extract service
curl -X POST http://localhost:8085/export/session/session_abc123?source=extract

# Export from agent telemetry service
curl -X POST http://localhost:8085/export/session/session_abc123?source=agent_telemetry

# Export multiple sessions
curl -X POST http://localhost:8085/export/batch \
  -H "Content-Type: application/json" \
  -d '{
    "session_ids": ["session1", "session2"],
    "source": "extract"
  }'
```

## Data Sources

### Extract Service

- **Endpoint**: `GET /signavio/agent-metrics/{session-id}`
- **Source**: `extract`
- Returns agent telemetry with metrics and events

### Agent Telemetry Service

- **Endpoint**: `GET /agent_metrics/{session-id}/events`
- **Source**: `agent_telemetry`
- Returns raw telemetry events

## Telemetry Format

Exported telemetry includes:

- **Basic Fields**: agent_run_id, agent_name, task_id, start_time, end_time, status
- **Metrics**: outcome_summary, latency_ms, notes
- **Enhanced Fields** (as JSON):
  - `tools_used`: Tool usage statistics
  - `llm_calls`: LLM/model call statistics
  - `process_steps`: Process step events

## Idempotency

The service tracks exported sessions to prevent duplicate exports. Re-exporting the same session will be skipped automatically.

## Error Handling

- Comprehensive error handling with retries for Signavio uploads
- Proper HTTP status codes
- Detailed error messages in responses
- Logging for debugging

## Integration

- **Testing Service**: Reuses `testing.SignavioClient` for Signavio uploads
- **Extract Service**: Calls `/signavio/agent-metrics/{session-id}` endpoint
- **Agent Telemetry Service**: Uses agent telemetry client pattern
- **Signavio API**: Production Ingestion API with multipart/form-data

## Production Considerations

- No test files or mock data generated
- All API calls use production endpoints
- Proper error handling and retries
- Idempotent operations
- Graceful shutdown
- Health check endpoint
- Comprehensive logging

## Future Enhancements

- Automatic session discovery from event streams
- Database-backed session tracking
- Scheduled batch exports
- Export status persistence
- Metrics and observability

