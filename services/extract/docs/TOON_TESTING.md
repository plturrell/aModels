# TOON Testing Guide

## Testing Token Generation

### 1. Test Plain Text Prompt
```bash
curl -X POST http://localhost:8083/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document": "John Smith works at Acme Corp. Email: john@acme.com. Deadline: 2024-12-31.",
    "prompt_description": "Extract person names, email addresses, and dates from the text."
  }'
```

**Expected**: Token structure with type "text"

### 2. Test Template Prompt
```bash
curl -X POST http://localhost:8083/extract \
  -H "Content-Type: application/json" \
  -d '{
    "document": "Invoice #12345 dated 2024-01-15",
    "prompt_description": "Extract {{.entity_type}} from: {{.document}}"
  }'
```

**Expected**: Token structure with type "template" and variable children

### 3. Verify Token Logging

Check logs for token telemetry:
```bash
docker compose logs extract | grep -i "prompt.*token\|token.*prompt"
```

## Token Structure Examples

### Plain Text Token
```json
{
  "type": "text",
  "value": "Extract key entities from the document",
  "metadata": {
    "length": "42"
  }
}
```

### Template Token
```json
{
  "type": "template",
  "value": "go-template",
  "metadata": {
    "format": "go-template",
    "variable_count": "2",
    "has_source": "true",
    "has_rendered": "true"
  },
  "children": [
    {
      "type": "variable",
      "value": "entity_type",
      "metadata": {
        "name": "entity_type",
        "type": "string"
      },
      "children": [
        {
          "type": "value:string",
          "value": "invoice"
        }
      ]
    },
    {
      "type": "rendered_text",
      "value": "Extract invoice from: Invoice #12345",
      "metadata": {
        "length": "45"
      }
    }
  ]
}
```

## Debugging

### Enable Verbose Logging
Set environment variable:
```yaml
environment:
  - PROMPT_DEBUG=1
```

### Check Token Generation
Add debug logging in `buildLangextractPayload()`:
```go
s.logger.Printf("Generated %d tokens for prompt: %s", len(promptTokens), promptTemplate)
```

### Verify Telemetry Integration
Check if telemetry client is initialized:
```bash
docker compose logs extract | grep "telemetry enabled\|telemetry disabled"
```

## Troubleshooting

### Issue: No tokens generated
- Check if prompt contains template syntax
- Verify orchestration package is imported
- Check build logs for compilation errors

### Issue: Token logging fails
- Verify telemetry is enabled (TELEMETRY_ENABLED=true)
- Check POSTGRES_LANG_SERVICE_ADDR is set
- Review telemetry client initialization logs

### Issue: Service crashes on startup
- Check protobuf version compatibility
- Verify telemetry build tags are correct
- Review panic stack traces in logs

## Performance Testing

### Measure Token Generation Overhead
```bash
time curl -X POST http://localhost:8083/extract \
  -H "Content-Type: application/json" \
  -d '{"document": "test", "prompt_description": "extract"}'
```

### Compare With/Without TOON
- Token generation adds ~1-2ms overhead
- Async logging has zero blocking impact
- Memory overhead: ~1KB per prompt

## Integration Tests

### Test Token Persistence
1. Generate tokens for multiple prompts
2. Verify prompt IDs are deterministic
3. Check token structure consistency

### Test Template Variable Resolution
1. Use template with variables
2. Verify variables are captured in token tree
3. Check metadata includes variable types

## Monitoring

### Key Metrics
- Token generation rate
- Template vs text prompt ratio
- Variable count distribution
- Token logging success rate

### Alerts
- Token generation failures
- Telemetry logging errors
- High token generation latency


