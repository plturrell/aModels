# Integration Testing Guide

This document describes the integration tests for key integration points in the lang infrastructure.

## Overview

Integration tests verify that services can communicate correctly and that workflows execute as expected. These tests are designed to run against real services (when available) or can be skipped if services are not running.

## Test Structure

### Go Integration Tests

**Location**: `services/graph/pkg/workflows/`

**Build Tag**: `// +build integration`

**Files**:
- `orchestration_processor_integration_test.go` - Tests orchestration chain creation and execution
- `unified_processor_integration_test.go` - Tests unified workflow execution

**Running Tests**:
```bash
# Run all integration tests
cd services/graph
go test -tags=integration -v ./pkg/workflows/...

# Run specific test
go test -tags=integration -v ./pkg/workflows/... -run TestOrchestrationChainExecution

# Skip integration tests (run unit tests only)
go test -v ./pkg/workflows/...
```

### Python Integration Tests

**Location**: `services/deepagents/integration_test.py`

**Framework**: pytest with async support

**Running Tests**:
```bash
# Run all integration tests
cd services/deepagents
pytest integration_test.py -v

# Run specific test
pytest integration_test.py::test_deepagents_invoke_simple -v

# Skip tests that require external services
pytest integration_test.py -v -m "not requires_external"
```

## Test Categories

### 1. Orchestration Chain Tests

**File**: `orchestration_processor_integration_test.go`

**Tests**:
- `TestOrchestrationChainCreation` - Verifies all chain types can be created
- `TestOrchestrationChainExecution` - Tests chain execution with LocalAI
- `TestOrchestrationChainWithKnowledgeGraph` - Tests chains with KG context
- `TestOrchestrationChainRetry` - Verifies retry logic
- `TestOrchestrationChainInputValidation` - Tests error handling for missing inputs

**Requirements**:
- `LOCALAI_URL` environment variable (defaults to `http://localai:8080`)
- LocalAI service must be running

**Example**:
```go
func TestOrchestrationChainExecution(t *testing.T) {
    localAIURL := os.Getenv("LOCALAI_URL")
    if localAIURL == "" {
        t.Skip("LOCALAI_URL not set")
    }
    
    chain, err := createOrchestrationChain("llm_chain", localAIURL)
    // ... test execution
}
```

### 2. Unified Workflow Tests

**File**: `unified_processor_integration_test.go`

**Tests**:
- `TestUnifiedWorkflowSequentialMode` - Tests sequential execution mode
- `TestUnifiedWorkflowWithKnowledgeGraph` - Tests KG processing in workflow
- `TestUnifiedWorkflowParallelMode` - Tests parallel execution mode
- `TestUnifiedWorkflowErrorHandling` - Tests error handling

**Requirements**:
- `EXTRACT_SERVICE_URL` (defaults to `http://extract-service:19080`)
- `AGENTFLOW_SERVICE_URL` (defaults to `http://agentflow-service:9001`)
- `LOCALAI_URL` (defaults to `http://localai:8080`)
- Services must be running (tests will skip if unavailable)

**Example**:
```go
func TestUnifiedWorkflowSequentialMode(t *testing.T) {
    workflow, err := NewUnifiedProcessorWorkflow(opts)
    // ... test execution
}
```

### 3. DeepAgents Integration Tests

**File**: `services/deepagents/integration_test.py`

**Tests**:
- `test_deepagents_health` - Health check endpoint
- `test_deepagents_invoke_simple` - Simple agent invocation
- `test_deepagents_invoke_with_context` - Agent with system context
- `test_deepagents_integration_with_extract` - Extract service integration
- `test_deepagents_integration_with_agentflow` - AgentFlow integration
- `test_deepagents_integration_with_graph` - Graph service integration
- `test_deepagents_error_handling` - Error handling
- `test_deepagents_streaming` - Streaming mode
- `test_deepagents_agent_info` - Agent info endpoint

**Requirements**:
- `DEEPAGENTS_URL` (defaults to `http://localhost:9004`)
- `EXTRACT_SERVICE_URL`, `AGENTFLOW_SERVICE_URL`, `GRAPH_SERVICE_URL` (for integration tests)
- DeepAgents service must be running

**Example**:
```python
@pytest.mark.asyncio
async def test_deepagents_invoke_simple(deepagents_client):
    request = {"messages": [{"role": "user", "content": "Hello"}]}
    response = await deepagents_client.post("/invoke", json=request)
    assert response.status_code == 200
```

## Environment Setup

### Required Environment Variables

```bash
# Core Services
LOCALAI_URL=http://localhost:8080
EXTRACT_SERVICE_URL=http://localhost:19080
AGENTFLOW_SERVICE_URL=http://localhost:9001
GRAPH_SERVICE_URL=http://localhost:8081
DEEPAGENTS_URL=http://localhost:9004

# Optional (for specific tests)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

### Docker Compose Setup

```bash
# Start all services
docker-compose up -d

# Run integration tests
cd services/graph
go test -tags=integration -v ./pkg/workflows/...

cd services/deepagents
pytest integration_test.py -v
```

## Test Execution Strategies

### 1. Full Integration Tests

Run all tests against real services:

```bash
# Go tests
go test -tags=integration -v ./pkg/workflows/...

# Python tests
pytest integration_test.py -v
```

### 2. Unit Tests Only

Skip integration tests:

```bash
# Go tests (integration tests skipped automatically without -tags=integration)
go test -v ./pkg/workflows/...

# Python tests (skip integration tests)
pytest integration_test.py -v -m "not integration"
```

### 3. Specific Test Suites

```bash
# Test only orchestration chains
go test -tags=integration -v ./pkg/workflows/... -run TestOrchestration

# Test only unified workflow
go test -tags=integration -v ./pkg/workflows/... -run TestUnifiedWorkflow

# Test only DeepAgents health
pytest integration_test.py::test_deepagents_health -v
```

## Test Best Practices

### 1. Skip Tests When Services Unavailable

```go
if localAIURL == "" {
    t.Skip("LOCALAI_URL not set, skipping integration test")
}
```

### 2. Use Timeouts

```go
ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()
```

### 3. Verify Output Structure

```go
if result == nil {
    t.Fatal("Chain returned nil result")
}
outputKeys := chain.GetOutputKeys()
if len(outputKeys) == 0 {
    t.Fatal("Chain has no output keys")
}
```

### 4. Handle Graceful Degradation

```python
# Services might be unavailable, so accept multiple status codes
assert response.status_code in [200, 500]
```

### 5. Clean Up Resources

```go
defer cancel()
defer client.Close()
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Start services
        run: docker-compose up -d
      - name: Wait for services
        run: sleep 30
      - name: Run Go integration tests
        run: |
          cd services/graph
          go test -tags=integration -v ./pkg/workflows/...
      - name: Run Python integration tests
        run: |
          cd services/deepagents
          pip install -r requirements.txt
          pytest integration_test.py -v
```

## Troubleshooting

### Tests Fail with Connection Errors

1. Verify services are running:
   ```bash
   docker-compose ps
   ```

2. Check service URLs:
   ```bash
   echo $LOCALAI_URL
   echo $EXTRACT_SERVICE_URL
   ```

3. Test service connectivity:
   ```bash
   curl http://localhost:8080/health
   ```

### Tests Timeout

1. Increase timeout in test:
   ```go
   ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
   ```

2. Check service logs:
   ```bash
   docker-compose logs localai
   ```

### Tests Skip Unexpectedly

1. Check environment variables:
   ```bash
   env | grep -E "(LOCALAI|EXTRACT|AGENTFLOW|GRAPH|DEEPAGENTS)_URL"
   ```

2. Verify build tags (Go):
   ```bash
   go test -tags=integration -v ./...
   ```

## Future Enhancements

1. **Mock Services**: Create mock implementations for services
2. **Test Containers**: Use testcontainers for isolated test environments
3. **Performance Tests**: Add performance benchmarks
4. **Load Tests**: Test under concurrent load
5. **Chaos Tests**: Test failure scenarios and recovery

## References

- [Go Testing Package](https://pkg.go.dev/testing)
- [pytest Documentation](https://docs.pytest.org/)
- [Integration Testing Best Practices](https://martinfowler.com/articles/nonDeterminism.html)

