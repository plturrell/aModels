# Trace Export Formats

This document describes the export formats supported by the telemetry-exporter service.

## JSON Lines Format

Each line in the file is a complete JSON-encoded `ExportTraceServiceRequest`:

```json
{"resourceSpans":[{"resource":{"attributes":[...]},"scopeSpans":[...]}]}
{"resourceSpans":[{"resource":{"attributes":[...]},"scopeSpans":[...]}]}
```

### Advantages
- Human-readable
- Easy to parse line-by-line
- Can be processed with standard JSON tools
- Supports streaming processing

### Example
```bash
# Read traces
cat traces.jsonl | jq '.resourceSpans[0].scopeSpans[0].spans[0].name'

# Filter by service
cat traces.jsonl | jq 'select(.resourceSpans[0].resource.attributes[] | select(.key=="service.name" and .value.stringValue=="extract-service"))'
```

## Protobuf Format

Length-prefixed protobuf messages in binary format:

```
[4-byte big-endian length][protobuf ExportTraceServiceRequest][4-byte length][...]
```

### Advantages
- Compact binary format
- Fast parsing
- Type-safe
- Standard OTLP format

### Reading Protobuf Files

```python
import struct
from opentelemetry.proto.collector.trace.v1.trace_pb2 import ExportTraceServiceRequest

with open('traces.pb', 'rb') as f:
    while True:
        # Read length
        length_bytes = f.read(4)
        if len(length_bytes) < 4:
            break
        length = struct.unpack('>I', length_bytes)[0]
        
        # Read message
        message_bytes = f.read(length)
        request = ExportTraceServiceRequest()
        request.ParseFromString(message_bytes)
        
        # Process request
        print(request)
```

## File Rotation

Files are automatically rotated when they exceed `OTEL_EXPORT_FILE_MAX_SIZE`:

- Original: `traces.jsonl`, `traces.pb`
- Rotated: `traces-20240101-120000-1.jsonl`, `traces-20240101-120000-1.pb`

Old files are cleaned up when the number of files exceeds `OTEL_EXPORT_FILE_MAX_FILES`.

## Signavio Format

Signavio export uses CSV format with Avro schema:

```csv
agent_run_id,agent_name,task_id,task_description,start_time,end_time,status,latency_ms,...
run-123,extract-agent,task-456,Extract data,1704067200000,1704067201000,success,1000,...
```

Complex fields (tools, LLM calls, process steps) are JSON-encoded strings in CSV columns.

## Comparison

| Format | Size | Readability | Processing Speed | Use Case |
|--------|------|-------------|------------------|----------|
| JSON Lines | Large | High | Medium | Development, debugging |
| Protobuf | Small | Low | Fast | Production, archival |
| Signavio CSV | Medium | Medium | Medium | Process mining, analysis |

## Recommended Usage

- **Development**: Use JSON Lines for easy inspection
- **Production**: Use Protobuf for efficient storage
- **Analysis**: Export to Signavio for process mining
- **Archival**: Use Protobuf for long-term storage

