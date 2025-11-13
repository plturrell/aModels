# OpenTelemetry Trace Export Configuration

This document describes how to configure OpenTelemetry trace export across all agent services.

## Overview

All agent services support OpenTelemetry trace export with full attribute capture. Traces are exported to:
- **File storage**: JSON Lines and Protobuf formats
- **Signavio Process Intelligence**: For process mining and analysis
- **OTLP HTTP**: Standard OpenTelemetry Protocol endpoint

## Service Configuration

### Extract Service

```bash
# Enable tracing
OTEL_TRACES_ENABLED=true

# Exporter configuration
OTEL_EXPORTER_TYPE=otlp  # or "jaeger", "otlp", or comma-separated list
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318

# File export (handled by telemetry-exporter service)
OTEL_EXPORT_FILE_ENABLED=true
OTEL_EXPORT_FILE_PATH=/app/data/traces

# Signavio export (handled by telemetry-exporter service)
OTEL_EXPORT_SIGNAVIO_ENABLED=false

# Agent framework type
AGENT_FRAMEWORK_TYPE=extract
```

### Graph Service (LangGraph)

```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_TYPE=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
OTEL_EXPORT_FILE_ENABLED=true
OTEL_EXPORT_FILE_PATH=/app/data/traces
AGENT_FRAMEWORK_TYPE=langgraph
```

### Regulatory Service (Goose, Deep Research)

```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_TYPE=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
OTEL_EXPORT_FILE_ENABLED=true
OTEL_EXPORT_FILE_PATH=/app/data/traces
AGENT_FRAMEWORK_TYPE=regulatory
```

### DeepAgents Service (Python)

```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
```

### Browser Service (Python)

```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
```

## Telemetry Exporter Service

The telemetry-exporter service aggregates traces from all services and exports them.

### Configuration

```bash
# Export mode
OTEL_EXPORT_MODE=both  # continuous, on-demand, or both

# File export
OTEL_EXPORT_FILE_ENABLED=true
OTEL_EXPORT_FILE_PATH=/app/data/traces
OTEL_EXPORT_FILE_MAX_SIZE=104857600  # 100MB
OTEL_EXPORT_FILE_MAX_FILES=10

# Signavio export
OTEL_EXPORT_SIGNAVIO_ENABLED=true
SIGNAVIO_API_URL=https://ingestion-eu.signavio.com
SIGNAVIO_API_KEY=your-api-key
SIGNAVIO_TENANT_ID=your-tenant-id
SIGNAVIO_DATASET=agent-telemetry
SIGNAVIO_BATCH_SIZE=100
SIGNAVIO_TIMEOUT=30s
SIGNAVIO_MAX_RETRIES=3
```

## Sampling Configuration

Control trace sampling to manage volume:

```bash
# Sample all traces (default)
OTEL_TRACES_SAMPLER_RATIO=1.0

# Sample 10% of traces
OTEL_TRACES_SAMPLER_RATIO=0.1

# Sample 50% of traces
OTEL_TRACES_SAMPLER_RATIO=0.5
```

## Attribute Limits

By default, all services are configured with unlimited span limits for full attribute capture:

- `AttributeValueLengthLimit`: -1 (no limit)
- `AttributeCountLimit`: -1 (no limit)
- `EventCountLimit`: -1 (no limit)
- `LinkCountLimit`: -1 (no limit)

This ensures all agent execution details are captured in traces.

## Best Practices

1. **Set Agent Framework Type**: Always set `AGENT_FRAMEWORK_TYPE` for proper categorization
2. **Use Descriptive Span Names**: Include workflow/agent context in span names
3. **Add Workflow Attributes**: Include `workflow.name`, `workflow.version` for tracking
4. **Record Tool Usage**: Emit `tool.call` and `tool.result` events
5. **Track LLM Calls**: Emit `llm.call` and `llm.response` events with token counts
6. **Monitor File Sizes**: Adjust `OTEL_EXPORT_FILE_MAX_SIZE` based on storage capacity
7. **Configure Signavio Batching**: Set `SIGNAVIO_BATCH_SIZE` for optimal upload performance

## Troubleshooting

### Traces Not Appearing

1. Verify `OTEL_TRACES_ENABLED=true` in all services
2. Check telemetry-exporter service is running
3. Verify OTLP endpoint is accessible
4. Check service logs for export errors

### High Storage Usage

1. Reduce `OTEL_TRACES_SAMPLER_RATIO` to sample fewer traces
2. Decrease `OTEL_EXPORT_FILE_MAX_FILES` to keep fewer files
3. Reduce `OTEL_EXPORT_FILE_MAX_SIZE` for smaller files

### Signavio Upload Failures

1. Verify API credentials are correct
2. Check network connectivity to Signavio
3. Increase `SIGNAVIO_TIMEOUT` for slow networks
4. Reduce `SIGNAVIO_BATCH_SIZE` for large records

