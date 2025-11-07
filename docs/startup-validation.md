# Startup Validation for Environment Variables

This document describes the startup validation implemented for environment variables across lang infrastructure services.

## Overview

All services now validate required environment variables at startup, providing clear error messages if configuration is missing or invalid. This prevents services from starting with incomplete configuration and helps identify configuration issues early.

## Implementation by Service

### Graph Service

**Location**: `services/graph/pkg/config/validation.go`

**Validation Function**: `ValidateGraphService()`

**Required Variables**:
- `EXTRACT_SERVICE_URL` (must be valid URL)
- `AGENTFLOW_SERVICE_URL` (must be valid URL)
- `LOCALAI_URL` (must be valid URL)

**Optional Variables** (validated if set):
- `DEEPAGENTS_SERVICE_URL` (must be valid URL if set)
- `GPU_ORCHESTRATOR_URL` (must be valid URL if set)

**Usage**:
```go
import "github.com/langchain-ai/langgraph-go/pkg/config"

func main() {
    if err := config.ValidateGraphService(); err != nil {
        log.Fatalf("Configuration validation failed: %v", err)
    }
    // ... rest of startup
}
```

### Extract Service

**Location**: `services/extract/internal/config/config.go`

**Validation Method**: `Config.Validate()`

**Required Variables**:
- `PORT` (server port)
- `LANGEXTRACT_API_URL` (LangExtract service URL)
- `LANGEXTRACT_API_KEY` (LangExtract API key)

**Conditionally Required** (if Neo4j is being used):
- `NEO4J_URI` (required if any Neo4j field is set)
- `NEO4J_USERNAME` (required if any Neo4j field is set)
- `NEO4J_PASSWORD` (required if any Neo4j field is set)

**Usage**:
```go
cfg, err := config.LoadConfig()
if err != nil {
    log.Fatalf("failed to load configuration: %v", err)
}
// Validation is called automatically in LoadConfig()
```

### DeepAgents Service

**Location**: `services/deepagents/main.py`

**Validation Function**: `validate_config()`

**Required Variables**:
- `EXTRACT_SERVICE_URL`
- `AGENTFLOW_SERVICE_URL`
- `GRAPH_SERVICE_URL`

**At Least One Required** (LLM provider):
- `ANTHROPIC_API_KEY` OR
- `OPENAI_API_KEY` OR
- `LOCALAI_URL`

**Usage**:
```python
@app.on_event("startup")
async def startup():
    validate_config()  # Raises ValueError if validation fails
    # ... rest of startup
```

### AgentFlow Service

**Location**: `services/agentflow/service/config.py`

**Validation**: Automatic via Pydantic `BaseSettings`

AgentFlow uses Pydantic's `BaseSettings` which automatically validates configuration. No additional validation needed.

### Orchestration Service

**Location**: `services/orchestration/`

**Status**: Internal service, no HTTP interface. Configuration validation handled by individual components.

## Validation Utility (Graph Service)

The Graph service provides a reusable validation utility:

**Location**: `services/graph/pkg/config/validation.go`

**Features**:
- `Require(key)`: Validates that a required variable is set
- `RequireOneOf(keys...)`: Validates that at least one of several variables is set
- `RequireURL(key)`: Validates that a required variable is set and is a valid URL
- `OptionalURL(key)`: Validates that if a variable is set, it's a valid URL
- `Validate()`: Returns all validation errors

**Example**:
```go
v := config.NewValidator()
v.RequireURL("EXTRACT_SERVICE_URL")
v.RequireOneOf("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LOCALAI_URL")
if err := v.Validate(); err != nil {
    log.Fatal(err)
}
```

## Error Messages

Validation errors provide clear, actionable messages:

```
Configuration validation failed:
  EXTRACT_SERVICE_URL: required but not set
  AGENTFLOW_SERVICE_URL: invalid URL: parse "not-a-url": invalid URI for request
  ANTHROPIC_API_KEY or OPENAI_API_KEY or LOCALAI_URL: at least one must be set
```

## Benefits

1. **Early Failure**: Services fail fast at startup with clear error messages
2. **Clear Diagnostics**: Validation errors identify exactly what's missing or invalid
3. **Prevents Runtime Errors**: Catches configuration issues before they cause runtime failures
4. **Consistent Experience**: All services validate configuration in a similar way
5. **URL Validation**: Ensures service URLs are properly formatted

## Migration Notes

### Before

Services would start with missing configuration and fail later during operation:

```go
func main() {
    url := os.Getenv("EXTRACT_SERVICE_URL")  // Could be empty
    // ... service starts
    // ... fails later when trying to use url
}
```

### After

Services validate configuration at startup:

```go
func main() {
    if err := config.ValidateGraphService(); err != nil {
        log.Fatalf("Configuration validation failed: %v", err)
    }
    // ... service starts with validated configuration
}
```

## Testing

To test validation:

1. **Missing Required Variable**:
   ```bash
   unset EXTRACT_SERVICE_URL
   ./graph-server
   # Output: Configuration validation failed: EXTRACT_SERVICE_URL: required but not set
   ```

2. **Invalid URL**:
   ```bash
   export EXTRACT_SERVICE_URL="not-a-url"
   ./graph-server
   # Output: Configuration validation failed: EXTRACT_SERVICE_URL: invalid URL: ...
   ```

3. **Missing LLM Provider**:
   ```bash
   unset ANTHROPIC_API_KEY OPENAI_API_KEY LOCALAI_URL
   python -m deepagents.main
   # Output: Configuration validation failed: At least one LLM provider must be configured
   ```

## Future Enhancements

1. **Type Validation**: Validate numeric ports, boolean flags, etc.
2. **Connection Testing**: Test connectivity to required services
3. **Configuration Schema**: Define configuration schemas in YAML/JSON
4. **Validation Hooks**: Allow services to add custom validation rules
5. **Configuration Documentation**: Auto-generate configuration docs from validation rules

## References

- [Configuration Reference](./configuration-reference.md)
- [Graph Service Validation](../services/graph/pkg/config/validation.go)
- [Extract Service Config](../services/extract/internal/config/config.go)
- [DeepAgents Service](../services/deepagents/main.py)

