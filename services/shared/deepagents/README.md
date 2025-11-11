# Shared DeepAgents Client Library

Standardized Go client library for interacting with the DeepAgents service across all aModels services.

## Features

- Standardized HTTP client with retry logic
- Exponential backoff for resilience
- Health check before requests
- Structured output support
- Graceful degradation (non-fatal failures)
- Circuit breaker pattern support

## Usage

```go
import "github.com/plturrell/aModels/services/shared/deepagents"

// Create client with default config
client := deepagents.NewClient(deepagents.DefaultConfig())

// Or with custom config
client := deepagents.NewClient(deepagents.Config{
    BaseURL: "http://deepagents-service:9004",
    Timeout: 120 * time.Second,
    MaxRetries: 2,
    Logger: logger,
    Enabled: true,
})

// Invoke agent
req := deepagents.InvokeRequest{
    Messages: []deepagents.Message{
        {Role: "user", Content: "Analyze this data"},
    },
}
resp, err := client.Invoke(ctx, req)
if err != nil {
    // Handle error
}

// Invoke with structured output
structuredReq := deepagents.StructuredInvokeRequest{
    Messages: []deepagents.Message{
        {Role: "user", Content: "Provide structured analysis"},
    },
    ResponseFormat: map[string]interface{}{
        "type": "json_schema",
        "json_schema": jsonSchema,
    },
}
structuredResp, err := client.InvokeStructured(ctx, structuredReq)
if err != nil {
    // Handle error
}
```

## Error Handling Pattern

The client follows a graceful degradation pattern:
- Returns `nil, nil` on failure (non-fatal)
- Logs warnings but doesn't break the calling service
- Health check before attempting requests
- Retry with exponential backoff
- Circuit breaker support (can be added)

