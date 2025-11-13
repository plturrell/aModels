# Signavio Process Intelligence Integration

This document describes how to configure and use the Signavio export functionality for OpenTelemetry agent traces.

## Overview

The telemetry-exporter service converts OpenTelemetry traces to Signavio Process Intelligence format and uploads them via the Signavio Ingestion API. This enables process mining and analysis of agent execution patterns.

## Configuration

### Prerequisites

1. Signavio Process Intelligence account
2. Ingestion API connection configured in Signavio
3. API key and tenant ID

### Environment Variables

```bash
# Enable Signavio export
OTEL_EXPORT_SIGNAVIO_ENABLED=true

# Signavio API configuration
SIGNAVIO_API_URL=https://ingestion-eu.signavio.com
SIGNAVIO_API_KEY=your-api-key
SIGNAVIO_TENANT_ID=your-tenant-id
SIGNAVIO_DATASET=agent-telemetry

# Optional: Batch configuration
SIGNAVIO_BATCH_SIZE=100
SIGNAVIO_TIMEOUT=30s
SIGNAVIO_MAX_RETRIES=3
```

## Data Mapping

### OTLP to Signavio Conversion

The exporter maps OpenTelemetry span attributes to Signavio telemetry records:

| OTLP Attribute | Signavio Field | Description |
|---------------|----------------|-------------|
| `agent.run_id` | `agent_run_id` | Unique identifier for agent execution |
| `agent.name` | `agent_name` | Name of the agent |
| `agent.type` | `agent_type` | Framework type (langgraph, deepagents, goose, etc.) |
| `service.name` | `service_name` | Service that executed the agent |
| `workflow.name` | `workflow_name` | Workflow identifier |
| Span name | `task_description` | Description of the task |
| Span duration | `latency_ms` | Execution latency |
| Span status | `status` | success or error |

### Extracted Metrics

The exporter automatically extracts:
- **Tool Usage**: From `tool.call` and `tool.result` events
- **LLM Calls**: From `llm.call` and `llm.response` events
- **Process Steps**: From `process.step` and `workflow.step` events
- **Prompt Metrics**: From prompt-related attributes

## Signavio Schema

The exported data follows this schema:

```json
{
  "agent_run_id": "string",
  "agent_name": "string",
  "task_id": "string",
  "task_description": "string",
  "start_time": 1234567890123,
  "end_time": 1234567890124,
  "status": "success|error",
  "latency_ms": 1000,
  "service_name": "string",
  "workflow_name": "string",
  "agent_type": "string",
  "tools_used": [...],
  "llm_calls": [...],
  "process_steps": [...],
  "prompt_metrics": {...}
}
```

## Upload Process

1. Traces are collected and converted to Signavio format
2. Records are batched (default: 100 records per batch)
3. Each batch is uploaded as CSV with Avro schema via multipart/form-data
4. Signavio processes and indexes the data for analysis

## Troubleshooting

### Export Not Working

1. Verify `OTEL_EXPORT_SIGNAVIO_ENABLED=true`
2. Check API credentials are correct
3. Verify Signavio dataset exists
4. Check service logs for error messages

### Missing Data

1. Ensure spans have required attributes (`agent.name`, `agent.type`)
2. Verify trace export is enabled in agent services
3. Check batch size - small batches may delay uploads

### Performance Issues

1. Adjust `SIGNAVIO_BATCH_SIZE` for optimal throughput
2. Increase `SIGNAVIO_TIMEOUT` for slow networks
3. Monitor `SIGNAVIO_MAX_RETRIES` for transient failures

## Best Practices

1. **Use Descriptive Agent Names**: Set `agent.name` attribute for clear identification
2. **Include Workflow Context**: Add `workflow.name` and `workflow.version` attributes
3. **Record Tool Usage**: Emit `tool.call` and `tool.result` events for tool tracking
4. **Track LLM Calls**: Emit `llm.call` and `llm.response` events for LLM metrics
5. **Monitor Batch Sizes**: Adjust batch size based on data volume and network conditions

## Example Dashboard Queries

Once data is in Signavio, you can create dashboards for:
- Agent execution frequency by type
- Average latency by workflow
- Tool usage patterns
- LLM call costs and performance
- Error rates by agent type
- Process step dependencies

