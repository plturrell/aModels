# DeepAgents Integration Guide

Comprehensive guide for integrating DeepAgents into aModels services.

## Overview

DeepAgents provides AI-powered capabilities across the aModels platform including:
- Catalog metadata management (deduplication, validation, research)
- Knowledge graph analysis
- Workflow optimization
- Flow analysis and optimization
- GPU orchestration
- Cross-system mapping

## Service Integration Patterns

### Pattern 1: Structured Outputs (Recommended)

Use structured outputs for reliable, parseable responses:

```go
// Go example
request := map[string]interface{}{
    "messages": []Message{
        {Role: "user", Content: "Analyze these elements"},
    },
    "response_format": map[string]interface{}{
        "type": "json_schema",
        "json_schema": jsonSchema,
    },
}

resp, err := http.Post("/invoke/structured", "application/json", body)
// Parse structured_output field
```

```python
# Python example
response = await client.post("/invoke/structured", json={
    "messages": [{"role": "user", "content": "Analyze this"}],
    "response_format": {
        "type": "json_schema",
        "json_schema": json_schema
    }
})
structured = response.json()["structured_output"]
```

### Pattern 2: Tool Invocation

Instruct the agent to use specific tools:

```go
prompt := "Use the check_duplicates tool to analyze these candidate elements..."
// Agent will automatically invoke the tool
```

### Pattern 3: Graceful Degradation

Always handle failures gracefully:

```go
response, err := client.Invoke(ctx, req)
if err != nil || response == nil {
    // Non-fatal: continue without AI enhancement
    return defaultBehavior()
}
// Use AI-enhanced response
```

## Service-Specific Integration

### Catalog Service

**Features:**
- Deduplication (enabled by default)
- Validation (enabled by default)
- Research (enabled by default)

**Configuration:**
```bash
# Disable all AI features
CATALOG_AI_DISABLED=true

# Disable specific features
CATALOG_AI_DEDUPLICATION_DISABLED=true
CATALOG_AI_VALIDATION_DISABLED=true
CATALOG_AI_RESEARCH_DISABLED=true

# Cache configuration
CATALOG_AI_CACHE_TTL=5m  # Default: 5 minutes
```

**Usage:**
```go
// Automatic in bulk registration endpoint
POST /catalog/data-elements/bulk
// AI suggestions included in response
```

### Extract Service

**Features:**
- Knowledge graph analysis
- Schema quality analysis
- Data lineage analysis
- Cross-system mapping suggestions

**Configuration:**
```bash
# Disable DeepAgents
DEEPAGENTS_ENABLED=false

# Service URL
DEEPAGENTS_URL=http://deepagents-service:9004
```

**Usage:**
```go
analysis, err := deepAgentsClient.AnalyzeKnowledgeGraph(ctx, summary, projectID, systemID)
// Structured output in analysis.StructuredOutput
```

### Graph Service

**Features:**
- Workflow state analysis
- Next steps suggestions
- Workflow optimization

**Configuration:**
```bash
DEEPAGENTS_SERVICE_URL=http://deepagents-service:9004
```

**Usage:**
```json
{
  "deepagents_request": {
    "messages": [{"role": "user", "content": "Analyze workflow"}],
    "config": {
      "response_format": {"type": "json"}
    }
  }
}
```

### AgentFlow Service

**Features:**
- Flow execution analysis
- Flow optimization suggestions
- Flow validation
- Flow comparison

**Configuration:**
```bash
DEEPAGENTS_ENABLED=true
DEEPAGENTS_URL=http://deepagents-service:9004
DEEPAGENTS_TIMEOUT_SECONDS=120
```

### GPU Orchestrator

**Features:**
- Intelligent GPU allocation
- Workload analysis
- Resource optimization

**Usage:**
```go
// Automatically uses analyze_workload and query_gpu_status tools
allocation, err := orchestrator.AllocateViaDeepAgents(ctx, serviceName, workloadType, data)
```

## Error Handling

### Standard Pattern

1. **Health Check First**: Quick 5s timeout check
2. **Retry Logic**: 2 retries with exponential backoff
3. **Graceful Degradation**: Return nil, nil on failure (non-fatal)
4. **Logging**: Log warnings but don't break service

### Example

```go
// Health check
if !client.checkHealth(ctx) {
    logger.Printf("DeepAgents unavailable, skipping")
    return nil, nil  // Non-fatal
}

// Retry with backoff
for attempt := 0; attempt <= maxRetries; attempt++ {
    resp, err := client.Do(req)
    if err != nil && attempt < maxRetries {
        time.Sleep(backoff)
        continue
    }
    // Process response
}
```

## Caching

Enable caching for improved performance:

```go
client.SetCache(cache)
// Responses automatically cached with TTL
```

**Cache Key Format:**
```
deepagents:{operation}:{hash}
```

**TTL Configuration:**
```bash
CATALOG_AI_CACHE_TTL=5m  # 5 minutes default
```

## Monitoring

### Metrics Endpoint

```bash
curl http://deepagents-service:9004/metrics
```

**Metrics Available:**
- Request counts by endpoint and status
- Latency statistics (p50, p95, p99)
- Error rates
- Tool usage statistics
- Token usage (input/output/total)
- Response quality metrics

### Prometheus Integration

Metrics can be exported to Prometheus format (future enhancement).

## Best Practices

1. **Always Use Structured Outputs**: More reliable than text parsing
2. **Enable Caching**: Reduces load and improves latency
3. **Health Check First**: Avoid unnecessary requests
4. **Graceful Degradation**: Never break core functionality
5. **Monitor Metrics**: Track usage and performance
6. **Use Tools Explicitly**: Instruct agent to use specific tools
7. **Configure Timeouts**: Set appropriate timeouts for your use case

## Troubleshooting

### Service Not Available
- Check health: `curl http://deepagents-service:9004/healthz`
- Verify environment variables
- Check service logs

### Structured Output Not Working
- Verify JSON schema is valid
- Check response for validation_errors
- Ensure using `/invoke/structured` endpoint

### High Latency
- Enable caching
- Check metrics for bottlenecks
- Consider increasing timeout
- Verify DeepAgents service performance

### Tools Not Available
- Check agent info: `curl http://deepagents-service:9004/agent/info`
- Verify tool imports in agent_factory.py
- Check service configuration

## Examples

See:
- `services/catalog/api/deepagents_client.go` - Catalog integration
- `services/extract/deepagents.go` - Extract integration
- `services/graph/pkg/workflows/deepagents_processor.go` - Graph integration
- `services/agentflow/service/deepagents.py` - AgentFlow integration

