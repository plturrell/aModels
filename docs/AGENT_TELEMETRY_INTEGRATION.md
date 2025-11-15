# Agent Telemetry Integration with OpenLLMetry

This document describes how OpenLLMetry is integrated across all aModels agent services to provide comprehensive LLM observability.

## Overview

OpenLLMetry provides standardized OpenTelemetry semantic conventions for LLM operations. All agent services in aModels are instrumented to emit OpenLLMetry attributes, enabling:

- **Automatic LLM Trace Recognition**: LLM operations are automatically detected and enriched
- **Token Usage Tracking**: Detailed tracking of prompt and completion tokens
- **Cost Monitoring**: Cost calculation based on token usage
- **Performance Metrics**: Latency, throughput, and error rate tracking
- **Model Analytics**: Per-model performance and usage statistics

## Architecture

### Components

1. **OpenLLMetry Submodule**: Reference implementation and semantic conventions
   - Location: `infrastructure/third_party/openllmetry`
   - Provides Python instrumentation packages and semantic convention definitions

2. **Go LLM Observability Package**: Shared helpers for Go services
   - Location: `pkg/observability/llm/attributes.go`
   - Provides `AddLLMRequestAttributes()` and `AddLLMResponseAttributes()` helpers

3. **Telemetry Exporter**: Processes and enriches LLM spans
   - Location: `services/telemetry-exporter`
   - Recognizes LLM spans using OpenLLMetry conventions
   - Enriches Signavio exports with LLM metrics

4. **Browser UI**: Visual dashboard for LLM observability
   - Location: `services/browser/shell/ui/src/modules/LLMObservability`
   - Displays LLM metrics, token usage, costs, and performance

## Service Integration

### Python Services (DeepAgents, Browser)

**Package Installation:**
```bash
pip install opentelemetry-instrumentation-langchain>=0.48.0
pip install opentelemetry-semantic-conventions-ai>=0.4.13
```

**Instrumentation:**
```python
from opentelemetry.instrumentation.langchain import LangchainInstrumentor

# In otel_instrumentation.py
LangchainInstrumentor().instrument()
```

**Automatic Instrumentation:**
- LangChain `ChatOpenAI` calls are automatically instrumented
- All LLM operations emit OpenLLMetry semantic convention attributes
- No code changes required in agent code

**Files Modified:**
- `services/deepagents/requirements.txt` - Added OpenLLMetry packages
- `services/deepagents/otel_instrumentation.py` - Added LangChain instrumentation
- `services/browser/requirements.txt` - Added OpenLLMetry packages
- `services/browser/otel_instrumentation.py` - Added LangChain instrumentation

### Go Services (LocalAI, Extract, Graph, Regulatory)

**Shared Package:**
```go
import "github.com/plturrell/aModels/pkg/observability/llm"

// Add request attributes
llmConfig := llm.LLMRequestConfig{
    System:      "localai",
    Model:       "phi-3.5-mini",
    RequestType: "chat",
    Temperature: 0.7,
    MaxTokens:   512,
}
llm.AddLLMRequestAttributes(span, llmConfig)

// Add response attributes
llmResponse := llm.LLMResponseInfo{
    PromptTokens:     150,
    CompletionTokens: 75,
    TotalTokens:      225,
    FinishReason:     "stop",
}
llm.AddLLMResponseAttributes(span, llmResponse)
```

**LocalAI Client:**
- `services/localai/pkg/client/localai_client.go`
- Automatically adds OpenLLMetry attributes to all LLM calls
- Supports both `Generate()` and `GenerateChat()` methods

**LocalAI Server:**
- `services/localai/pkg/server/api_v2.go`
- Emits OpenLLMetry attributes in response spans
- Includes token usage and finish reason

**Agent Services:**
- `services/extract/internal/observability/tracing.go` - Added LLM helpers
- `services/graph/pkg/observability/tracing.go` - Added LLM helpers
- `services/regulatory/internal/observability/tracing.go` - Added LLM helpers

**Usage in Agent Code:**
```go
import "github.com/plturrell/aModels/services/extract/internal/observability"

// In your agent code
ctx, span := observability.StartSpan(ctx, "llm.call")
defer span.End()

// Add LLM request attributes
observability.AddLLMRequestAttributes(ctx, llmConfig)

// ... make LLM call ...

// Add LLM response attributes
observability.AddLLMResponseAttributes(ctx, llmResponse)
```

## OpenLLMetry Attributes

### Required Attributes

All LLM spans must include:

- `gen_ai.system`: Provider name (e.g., "localai", "openai", "anthropic")
- `gen_ai.request.model`: Model identifier
- `llm.request.type`: Request type ("chat", "completion", "embedding")
- `gen_ai.usage.prompt_tokens`: Input/prompt tokens
- `gen_ai.usage.completion_tokens`: Output/completion tokens
- `llm.usage.total_tokens`: Total tokens

### Optional Attributes

- `gen_ai.request.temperature`: Temperature setting
- `gen_ai.request.top_p`: Top-p sampling parameter
- `llm.top_k`: Top-k sampling parameter
- `gen_ai.request.max_tokens`: Maximum tokens in response
- `llm.is_streaming`: Whether the request is streaming
- `llm.response.finish_reason`: Completion reason (e.g., "stop", "length")
- `gen_ai.usage.cost`: Cost of the operation
- `gen_ai.usage.cache_creation_input_tokens`: Tokens used for cache creation
- `gen_ai.usage.cache_read_input_tokens`: Tokens read from cache

## Telemetry Exporter Processing

The telemetry-exporter service automatically:

1. **Detects LLM Spans**: Uses `ExtractLLMInfo()` to identify LLM operations
2. **Enriches Signavio Exports**: Maps LLM attributes to Signavio telemetry records
3. **Enhances LLM Calls**: Adds token usage and model information to `LLMCalls` field

**Key Files:**
- `services/telemetry-exporter/pkg/llm/llm_trace.go` - LLM span detection
- `services/telemetry-exporter/pkg/signavio/otlp_exporter.go` - LLM attribute mapping

## Browser UI

The LLM Observability module provides:

- **Key Metrics Dashboard**: Total calls, tokens, cost, average latency
- **Model Performance**: Per-model breakdown of calls, tokens, and costs
- **Recent Traces Table**: Detailed view of recent LLM operations
- **Auto-refresh**: Updates every 30 seconds

**Location:** `services/browser/shell/ui/src/modules/LLMObservability/LLMObservabilityModule.tsx`

## Configuration

### Environment Variables

**Python Services:**
```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
TRACELOOP_TRACE_CONTENT=true  # Set to false to disable prompt/completion logging
```

**Go Services:**
```bash
OTEL_TRACES_ENABLED=true
OTEL_EXPORTER_TYPE=otlp
OTEL_EXPORTER_OTLP_ENDPOINT=http://telemetry-exporter:4318
```

## Best Practices

1. **Always Use Helpers**: Use the provided helper functions rather than manually setting attributes
2. **Include All Required Attributes**: Ensure system, model, and request type are always set
3. **Track Token Usage**: Always include prompt, completion, and total tokens
4. **Record Errors**: Use `span.RecordError()` for failed LLM calls
5. **Set Finish Reason**: Include finish reason in response attributes

## Example: Adding LLM Instrumentation to New Service

### Python Service

1. Add to `requirements.txt`:
   ```
   opentelemetry-instrumentation-langchain>=0.48.0
   opentelemetry-semantic-conventions-ai>=0.4.13
   ```

2. Update `otel_instrumentation.py`:
   ```python
   from opentelemetry.instrumentation.langchain import LangchainInstrumentor
   
   LangchainInstrumentor().instrument()
   ```

3. No code changes needed - LangChain calls are automatically instrumented

### Go Service

1. Add replace directive to `go.mod`:
   ```
   replace github.com/plturrell/aModels => ../..
   ```

2. Import the LLM package:
   ```go
   import "github.com/plturrell/aModels/pkg/observability/llm"
   ```

3. Use helpers in your code:
   ```go
   ctx, span := tracer.Start(ctx, "llm.call")
   defer span.End()
   
   llmConfig := llm.NewLLMRequestConfig("phi-3.5-mini", "chat")
   llm.AddLLMRequestAttributes(span, llmConfig)
   
   // ... make LLM call ...
   
   llmResponse := llm.LLMResponseInfo{
       PromptTokens:     int64(promptTokens),
       CompletionTokens: int64(completionTokens),
       TotalTokens:      int64(totalTokens),
       FinishReason:     "stop",
   }
   llm.AddLLMResponseAttributes(span, llmResponse)
   ```

## Troubleshooting

### LLM Spans Not Appearing

1. Verify tracing is enabled: `OTEL_TRACES_ENABLED=true`
2. Check exporter endpoint: `OTEL_EXPORTER_OTLP_ENDPOINT`
3. Verify OpenLLMetry packages are installed (Python) or imported (Go)
4. Check telemetry-exporter logs for processing errors

### Missing Attributes

1. Ensure all required attributes are set using helper functions
2. Check that spans are recording: `span.IsRecording()`
3. Verify attribute keys match OpenLLMetry conventions

### Python Instrumentation Not Working

1. Verify `LangchainInstrumentor().instrument()` is called before any LangChain usage
2. Check that `opentelemetry-instrumentation-langchain` is installed
3. Ensure OpenTelemetry tracer provider is initialized first

## References

- [OpenLLMetry GitHub](https://github.com/traceloop/openllmetry)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/genai/)
- [OpenLLMetry Documentation](https://traceloop.com/docs/openllmetry)
- [Telemetry Exporter LLM Observability Guide](../services/telemetry-exporter/docs/LLM_OBSERVABILITY.md)

