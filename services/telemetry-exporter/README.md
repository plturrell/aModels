# Telemetry Exporter Service

The Telemetry Exporter service aggregates and exports OpenTelemetry traces from all agent services to file storage and Signavio Process Intelligence.

## Features

- **File Export**: Exports traces in JSON Lines and Protobuf formats with automatic rotation
- **Signavio Export**: Converts OTLP traces to Signavio format and uploads via API
- **Continuous Export**: Background worker for automatic trace export
- **On-Demand Export**: REST API for manual trace export requests
- **Multi-Exporter Support**: Simultaneously export to multiple destinations
- **LLM Observability**: Integrated support for OpenLLMetry semantic conventions for LLM trace recognition and enrichment

## Configuration

### Environment Variables

#### General Configuration
- `OTEL_EXPORT_MODE`: Export mode - `continuous`, `on-demand`, or `both` (default: `both`)
- `OTEL_EXPORT_FLUSH_INTERVAL`: Flush interval for continuous export (default: `30s`)

#### File Export
- `OTEL_EXPORT_FILE_ENABLED`: Enable file export (default: `true`)
- `OTEL_EXPORT_FILE_PATH`: Directory for trace files (default: `/app/data/traces`)
- `OTEL_EXPORT_FILE_MAX_SIZE`: Maximum file size in bytes before rotation (default: `104857600` = 100MB)
- `OTEL_EXPORT_FILE_MAX_FILES`: Maximum number of files to keep (default: `10`)

#### Signavio Export
- `OTEL_EXPORT_SIGNAVIO_ENABLED`: Enable Signavio export (default: `false`)
- `SIGNAVIO_API_URL`: Signavio ingestion API base URL
- `SIGNAVIO_API_KEY`: Signavio API key
- `SIGNAVIO_TENANT_ID`: Signavio tenant ID
- `SIGNAVIO_DATASET`: Signavio dataset name
- `SIGNAVIO_BATCH_SIZE`: Batch size for uploads (default: `100`)
- `SIGNAVIO_TIMEOUT`: Request timeout (default: `30s`)
- `SIGNAVIO_MAX_RETRIES`: Maximum retry attempts (default: `3`)

## API Endpoints

### POST /api/v1/traces/export
Export traces on demand.

**Request Body:**
```json
{
  "time_range": {
    "start_time": "2024-01-01T00:00:00Z",
    "end_time": "2024-01-01T23:59:59Z"
  },
  "service_filter": ["extract-service", "graph-service"],
  "agent_type_filter": ["langgraph", "deepagents"],
  "export_format": "file",
  "destination": "/app/data/traces/export"
}
```

**Response:**
```json
{
  "export_id": "export-1234567890",
  "status": "pending",
  "export_time": "2024-01-01T12:00:00Z"
}
```

### GET /api/v1/traces/export/{export_id}
Get export status.

**Response:**
```json
{
  "export_id": "export-1234567890",
  "status": "completed",
  "file_location": "/app/data/traces/export-1234567890.jsonl",
  "record_count": 1500,
  "export_time": "2024-01-01T12:00:00Z"
}
```

### GET /health
Health check endpoint.

## File Export Format

### JSON Lines Format
Each line is a JSON-encoded `ExportTraceServiceRequest`:
```json
{"resourceSpans":[...]}
{"resourceSpans":[...]}
```

### Protobuf Format
Length-prefixed protobuf messages:
```
[4-byte length][protobuf message][4-byte length][protobuf message]...
```

## Signavio Integration

The service converts OTLP traces to Signavio telemetry records with:
- Agent run ID, name, and type
- Task ID and description
- Start/end times and latency
- Status and outcome summary
- Tool usage statistics
- LLM call metrics
- Process step information
- Prompt metrics

See [Signavio Integration Guide](./docs/SIGNAVIO_INTEGRATION.md) for detailed configuration.

## LLM Observability

The service integrates with [OpenLLMetry](https://github.com/traceloop/openllmetry) to provide comprehensive observability for LLM operations. LLM spans are automatically detected and enriched with token usage, cost, and performance metrics.

See [LLM Observability Guide](./docs/LLM_OBSERVABILITY.md) for detailed information.

## Development

### Building
```bash
cd services/telemetry-exporter
go build ./cmd/server
```

### Running
```bash
./server
```

### Testing
```bash
go test ./...
```

## Docker

The service is included in `docker-compose.yml` and runs on port 8083.

```bash
docker compose up telemetry-exporter
```
