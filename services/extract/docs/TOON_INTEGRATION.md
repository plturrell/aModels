# Token Oriented Object Notation (TOON) Integration

## Overview

The extract service now uses **Token Oriented Object Notation (TOON)** for prompt observability and analysis. This enables structured logging of prompt tokens, template variables, and prompt structure without re-parsing.

## Features

### 1. Token-Aware Prompt Creation
- Automatically detects template syntax (e.g., `{{.variable}}`)
- Creates hierarchical token trees for all prompts
- Supports both template and plain text prompts

### 2. Token Telemetry Logging
- Logs prompt tokens to telemetry service
- Generates deterministic prompt IDs (SHA256 hash)
- Includes template type, variable count, and token structure
- Asynchronous logging to avoid blocking extraction

### 3. Integration Points

#### Main Extraction Endpoint (`/extract`)
- All extraction requests now generate token structures
- Tokens are logged before sending to langextract API
- Works with both template and plain string prompts

## Usage

### Template Prompts
```json
{
  "document": "Text to extract from",
  "prompt_description": "Extract {{.entity_type}} from: {{.document}}"
}
```

### Plain Text Prompts
```json
{
  "document": "Text to extract from",
  "prompt_description": "Extract key entities from the document"
}
```

## Token Structure

Each prompt generates a hierarchical token tree:

```json
{
  "type": "text|template",
  "value": "prompt content",
  "metadata": {
    "length": "123",
    "format": "go-template",
    "variable_count": "2"
  },
  "children": [
    {
      "type": "variable",
      "value": "entity_type",
      "metadata": {
        "name": "entity_type",
        "type": "string"
      }
    }
  ]
}
```

## Telemetry Payload

When telemetry is enabled, each extraction logs:

```json
{
  "operation": "prompt_tokens",
  "input": {
    "prompt_id": "a1b2c3d4e5f6",
    "template_type": "template|text",
    "variable_count": 2,
    "tokens": [...]
  }
}
```

## Configuration

### Enable Telemetry
```yaml
# docker-compose.yml
environment:
  - TELEMETRY_ENABLED=true
  - POSTGRES_LANG_SERVICE_ADDR=postgres-lang-service:50051
```

### Disable Telemetry
```yaml
environment:
  - TELEMETRY_ENABLED=false
```

## Benefits

1. **Observability**: Track prompt usage patterns
2. **Debugging**: Inspect prompt structure without re-parsing
3. **Optimization**: Analyze which prompts are most effective
4. **Cost Tracking**: Monitor token usage per prompt type
5. **A/B Testing**: Compare different prompt structures

## Implementation Details

### Code Location
- Token creation: `cmd/extract/main.go::buildLangextractPayload()`
- Token logging: `cmd/extract/main.go::runExtract()`
- Telemetry client: `pkg/monitoring/telemetry.go`
- Token utilities: `pkg/monitoring/prompt_telemetry.go`

### Build Tags
- Telemetry code uses `// +build !notelemetry` tag
- Remove `-tags notelemetry` from Dockerfile to enable

## Testing

### Test Token Generation
```bash
curl -X POST http://localhost:8083/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Test document",
    "prompt_description": "Extract {{.entity}} from text"
  }'
```

### Verify Telemetry Logs
```bash
docker compose logs extract | grep "prompt_tokens"
```

## Future Enhancements

1. **Token Analytics Dashboard**: Visualize prompt usage
2. **Prompt Optimization**: Auto-suggest improvements
3. **Cost Analysis**: Track token costs per prompt
4. **A/B Testing Framework**: Compare prompt variations
5. **Template Library**: Reusable prompt templates

## References

- [Token-Aware Prompts Guide](../../infrastructure/third_party/orchestration/docs/token_prompt_guide.md)
- [Telemetry Configuration](../internal/config/config.go)


