# Testing with LocalAI and Signavio Export

## Quick Start

### 1. Generate Test Data

Run the test data generator to create Signavio-compatible output:

```bash
cd services/telemetry-exporter
docker run --rm \
  -v $(pwd):/workspace \
  -v $(pwd)/../testing:/workspace/../testing \
  -v $(pwd)/../../pkg:/workspace/../../pkg \
  -w /workspace \
  golang:1.24-alpine \
  sh -c "apk add --no-cache git && go run generate_test_extract.go"
```

This generates:
- `signavio_telemetry_export.csv` - CSV data ready for Signavio upload
- `signavio_telemetry_schema.json` - Avro schema for Signavio
- `signavio_telemetry_record.json` - JSON record for reference

### 2. Upload to Signavio

Use the Signavio Ingestion API to upload the CSV and schema:

```bash
curl -X POST https://ingestion-eu.signavio.com/ingestion/api/v1/datasets/your-dataset/upload \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "X-Tenant-ID: YOUR_TENANT_ID" \
  -F "files=@signavio_telemetry_export.csv" \
  -F "schema=@signavio_telemetry_schema.json" \
  -F "primaryKeys=agent_run_id" \
  -F "delimiter=,"
```

### 3. Verify Export

Check the generated files:

```bash
# View CSV data
head signavio_telemetry_export.csv

# View JSON record
cat signavio_telemetry_record.json | jq .

# View schema
cat signavio_telemetry_schema.json | jq .
```

## File Formats

### CSV Format
The CSV file contains one row per agent execution with:
- Basic fields: agent_run_id, agent_name, task_id, task_description, start_time, end_time, status
- Metrics: latency_ms, outcome_summary, notes
- Enhanced fields: service_name, workflow_name, agent_type, agent_state
- JSON-encoded arrays: tools_used, llm_calls, process_steps, prompt_metrics

### Avro Schema
The schema defines the structure for Signavio Process Intelligence:
- All fields are properly typed
- Optional fields use union types with null
- JSON-encoded fields are stored as strings

### JSON Record
The JSON file shows the full structure of a Signavio telemetry record for reference.

## Integration with Real Traces

To export real traces from the telemetry-exporter service:

1. Ensure services are instrumented and generating traces
2. Traces are automatically exported to files in `/app/data/traces/`
3. Use the export API to convert traces to Signavio format:

```bash
curl -X POST http://telemetry-exporter:8083/api/v1/traces/export \
  -H "Content-Type: application/json" \
  -d '{
    "export_format": "signavio",
    "destination": "agent-telemetry-dataset"
  }'
```

## LocalAI Testing

The service is built with LocalAI support enabled. To verify:

```bash
# Check service logs
docker logs telemetry-exporter-test

# Should see:
# [telemetry-exporter] File exporter enabled
# [telemetry-exporter] Telemetry exporter service starting on :8080
```

No errors about missing LocalAI dependencies should appear.

