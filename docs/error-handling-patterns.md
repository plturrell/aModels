# Error Handling Patterns

This document describes the standardized error handling and logging patterns used across the lang infrastructure services.

## Overview

All integration code should follow consistent patterns for:
- Error handling and retry logic
- Structured logging with correlation IDs
- HTTP request/response logging
- Integration call tracking

## Standardized Utilities

### Location

The standardized utilities are located in:
- `services/graph/pkg/integration/errors.go` - Retry logic and error handling
- `services/graph/pkg/integration/logging.go` - Structured logging with correlation IDs

### Retry Configuration

```go
import "github.com/langchain-ai/langgraph-go/pkg/integration"

config := integration.DefaultRetryConfig()
// Or customize:
config := &integration.RetryConfig{
    MaxRetries:       3,
    InitialDelay:     1 * time.Second,
    MaxDelay:         30 * time.Second,
    BackoffMultiplier: 2.0,
}
```

### Retry Logic

#### For Operations That Return Errors

```go
err := integration.RetryWithBackoff(
    ctx,
    integration.DefaultRetryConfig(),
    logger,
    "operation.name",
    func() error {
        // Your operation here
        return someOperation()
    },
)
```

#### For Operations That Return Results

```go
result, err := integration.RetryWithBackoffResult(
    ctx,
    integration.DefaultRetryConfig(),
    logger,
    "operation.name",
    func() (ResultType, error) {
        // Your operation here
        return someOperation()
    },
)
```

### Retryable Errors

The following errors are automatically retried:
- Network errors (timeout, connection, dial errors)
- HTTP 5xx server errors (500, 502, 503, 504)
- HTTP 429 (rate limiting)

Non-retryable errors (4xx client errors, validation errors) are not retried.

### Structured Logging

#### Operation Logging

```go
import "github.com/langchain-ai/langgraph-go/pkg/integration"

// Start operation (automatically generates correlation ID)
op := integration.StartOperation(ctx, logger, "service.operation")
defer op.End(nil)

// Log messages with correlation ID
op.Log("Processing data: %d items", count)

// End with error (if any)
op.End(err)
```

#### HTTP Request Logging

```go
startTime := time.Now()
resp, err := httpClient.Do(req)
duration := time.Since(startTime)

integration.LogHTTPRequest(ctx, logger, "POST", url, resp.StatusCode, duration, err)
```

#### Integration Call Logging

```go
startTime := time.Now()
result, err := callOtherService(ctx)
duration := time.Since(startTime)

integration.LogIntegrationCall(ctx, logger, "graph-service", "extract-service", "query", duration, err)
```

### Correlation IDs

Correlation IDs are automatically generated and propagated through context:

```go
// Get correlation ID from context (or generate new one)
correlationID := integration.GetCorrelationID(ctx)

// Add correlation ID to context
ctx = integration.WithCorrelationID(ctx, "custom-id")

// Generate and add new correlation ID
ctx = integration.WithNewCorrelationID(ctx)
```

## Error Handling Patterns by Service

### Graph Service

**File**: `services/graph/pkg/workflows/orchestration_processor.go`

**Pattern**:
```go
// Start logged operation
op := integration.StartOperation(ctx, log.Default(), fmt.Sprintf("orchestration.chain.%s", chainName))
defer op.End(nil)

// Execute with retry
err := integration.RetryWithBackoff(
    ctx,
    integration.DefaultRetryConfig(),
    log.Default(),
    fmt.Sprintf("orchestration.chain.%s.execute", chainName),
    func() error {
        result, err := chains.Call(ctx, chain, chainInputs)
        return err
    },
)

if err != nil {
    op.End(err)
    return nil, fmt.Errorf("execute orchestration chain %s: %w", chainName, err)
}

op.End(nil)
```

### Extract Service

**File**: `services/extract/deepagents.go`

**Pattern**: Non-fatal integration with retry
```go
// Retry logic with exponential backoff
var lastErr error
maxRetries := 2
for attempt := 0; attempt <= maxRetries; attempt++ {
    if attempt > 0 {
        backoff := time.Duration(attempt) * time.Second
        // Wait with backoff
    }
    
    resp, err := c.client.Do(req)
    if err != nil {
        lastErr = err
        if attempt < maxRetries {
            continue // Retry
        }
        // Return nil, nil (non-fatal)
        return nil, nil
    }
    
    // Check status code
    if resp.StatusCode != http.StatusOK {
        if resp.StatusCode >= 500 && attempt < maxRetries {
            continue // Retry server errors
        }
        // Return nil, nil (non-fatal)
        return nil, nil
    }
    
    // Success
    return &response, nil
}
```

**Note**: Extract service uses custom retry logic because DeepAgents integration is non-fatal. Consider migrating to standardized utilities in the future.

### Orchestration Service

**File**: `services/orchestration/agent_coordinator.go`

**Pattern**: Retry with exponential backoff
```go
for attempt := 0; attempt <= ac.retryConfig.MaxRetries; attempt++ {
    if attempt > 0 {
        delay := time.Duration(float64(ac.retryConfig.InitialDelay) * 
            pow(ac.retryConfig.BackoffMultiplier, float64(attempt-1)))
        if delay > ac.retryConfig.MaxDelay {
            delay = ac.retryConfig.MaxDelay
        }
        time.Sleep(delay)
    }
    
    result, err = ac.runAgentTask(ctx, agentID, task)
    if err == nil {
        return // Success
    }
}
```

## Log Format

### Operation Logs

```
[START] operation.name [correlation_id=abc123]
[LOG] operation.name [correlation_id=abc123] Processing data: 10 items
[END] operation.name [correlation_id=abc123, duration=1.23s]
[ERROR] operation.name [correlation_id=abc123, duration=0.5s, error=connection timeout]
```

### Retry Logs

```
[RETRY] operation.name: attempt 2/4 after 1s
[RETRY] operation.name: attempt 2 failed: connection timeout
[SUCCESS] operation.name: succeeded on attempt 2
[ERROR] operation.name: failed after 4 retries: connection timeout
```

### HTTP Logs

```
[HTTP] POST http://service:port/endpoint [correlation_id=abc123, status=200, duration=150ms]
[HTTP_ERROR] POST http://service:port/endpoint [correlation_id=abc123, status=503, duration=500ms, error=service unavailable]
```

### Integration Logs

```
[INTEGRATION] graph-service -> extract-service:query [correlation_id=abc123, duration=200ms]
[INTEGRATION_ERROR] graph-service -> extract-service:query [correlation_id=abc123, duration=500ms, error=timeout]
```

## Best Practices

1. **Always Use Correlation IDs**: Start operations with `StartOperation()` to get automatic correlation ID tracking

2. **Use Retry for Network Operations**: Wrap HTTP calls and external service calls with retry logic

3. **Log Operation Boundaries**: Use `StartOperation()` and `op.End()` to track operation duration

4. **Distinguish Retryable vs Non-Retryable**: Use `IsRetryableError()` or `IsRetryableHTTPStatus()` to determine if errors should be retried

5. **Log HTTP Requests**: Use `LogHTTPRequest()` for all HTTP calls to track performance and errors

6. **Log Integration Calls**: Use `LogIntegrationCall()` to track cross-service communication

7. **Context Propagation**: Always pass context through to maintain correlation IDs

8. **Error Wrapping**: Wrap errors with context: `fmt.Errorf("operation failed: %w", err)`

## Migration Guide

### Before (Custom Retry Logic)

```go
var lastErr error
maxRetries := 2
for attempt := 0; attempt <= maxRetries; attempt++ {
    if attempt > 0 {
        backoff := time.Duration(attempt) * time.Second
        time.Sleep(backoff)
    }
    
    result, err := operation()
    if err == nil {
        return result, nil
    }
    lastErr = err
}
return nil, lastErr
```

### After (Standardized)

```go
result, err := integration.RetryWithBackoffResult(
    ctx,
    integration.DefaultRetryConfig(),
    logger,
    "operation.name",
    func() (ResultType, error) {
        return operation()
    },
)
```

## Configuration

### Default Retry Configuration

- **MaxRetries**: 3
- **InitialDelay**: 1 second
- **MaxDelay**: 30 seconds
- **BackoffMultiplier**: 2.0 (exponential backoff)

### Customizing Retry Configuration

```go
customConfig := &integration.RetryConfig{
    MaxRetries:       5,              // More retries for critical operations
    InitialDelay:     500 * time.Millisecond, // Faster initial retry
    MaxDelay:         60 * time.Second,        // Longer max delay
    BackoffMultiplier: 1.5,                    // Less aggressive backoff
}
```

## Examples

### Example 1: Orchestration Chain Execution

```go
op := integration.StartOperation(ctx, logger, "orchestration.chain.knowledge_graph_analyzer")
defer op.End(nil)

var result map[string]any
err := integration.RetryWithBackoff(
    ctx,
    integration.DefaultRetryConfig(),
    logger,
    "orchestration.chain.execute",
    func() error {
        var execErr error
        result, execErr = chains.Call(ctx, chain, chainInputs)
        return execErr
    },
)

if err != nil {
    op.End(err)
    return nil, fmt.Errorf("execute chain: %w", err)
}

op.End(nil)
```

### Example 2: HTTP Service Call

```go
op := integration.StartOperation(ctx, logger, "agentflow.flow.execute")
defer op.End(nil)

var response FlowResponse
err := integration.RetryWithBackoffResult(
    ctx,
    integration.DefaultRetryConfig(),
    logger,
    "agentflow.flow.execute",
    func() (FlowResponse, error) {
        startTime := time.Now()
        resp, err := httpClient.Do(req)
        duration := time.Since(startTime)
        
        integration.LogHTTPRequest(ctx, logger, "POST", endpoint, resp.StatusCode, duration, err)
        
        if err != nil {
            return response, err
        }
        
        if resp.StatusCode != http.StatusOK {
            err := fmt.Errorf("status %d", resp.StatusCode)
            if integration.IsRetryableHTTPStatus(resp.StatusCode) {
                return response, err
            }
            return response, fmt.Errorf("%w (non-retryable)", err)
        }
        
        // Decode response...
        return response, nil
    },
)
```

## References

- [Integration Utilities](../services/graph/pkg/integration/errors.go)
- [Logging Utilities](../services/graph/pkg/integration/logging.go)
- [Graph Service Integration Guide](../services/graph/INTEGRATION.md)

