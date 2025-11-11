# AgentFlow Service Integration Guide

## Overview

The AgentFlow service acts as a bridge to LangFlow, enabling execution of pre-configured workflows from JSON files. It provides both a Go CLI and a FastAPI HTTP service for flow management and execution.

## Service Information

- **Port**: 9001 (FastAPI service)
- **Technology**: Go CLI + Python FastAPI service
- **Base URL**: `http://agentflow-service:9001`

## Integration Points

### 1. Flow Execution

**Purpose**: Execute LangFlow flows defined in JSON format.

**Configuration**:
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
```

**Flow Location**: `services/agentflow/flows/`

**Available Flows**:
- `processes/sgmi_controlm_pipeline.json`
- `processes/sample_reconciliation.json`
- `standards/sample_policy.json`

---

### 2. Graph Service Integration

The graph service can execute AgentFlow flows as part of unified workflows.

**File**: `services/graph/pkg/workflows/agentflow_processor.go`

**Usage**:
```go
afNode := RunAgentFlowFlowNode(agentFlowServiceURL)
result, err := afNode(ctx, state)
```

**State Input**:
```go
state := map[string]any{
    "agentflow_request": map[string]any{
        "flow_id": "processes/sgmi_controlm_pipeline",
        "input_value": "Process SGMI data",
        "inputs": map[string]any{
            "input": "Additional input data",
        },
        "ensure": true, // Ensure flow exists
    },
}
```

---

### 3. DeepAgents Integration

DeepAgents can execute flows via the `run_agentflow_flow` tool.

**Tool**: `services/deepagents/tools/agentflow_tool.py`

**Usage**:
```python
# Agent automatically uses this tool
{
  "messages": [
    {
      "role": "user",
      "content": "Run the SGMI pipeline flow with input 'Process data'"
    }
  ]
}
```

---

## API Endpoints

### Health Check

**Endpoint**: `GET /healthz`

**Response**: HTTP 200 OK

---

### Run Flow

**Endpoint**: `POST /run`

**Request Body**:
```json
{
  "flow_id": "processes/sgmi_controlm_pipeline",
  "input_value": "Process SGMI data",
  "inputs": {
    "additional_param": "value"
  },
  "ensure": true
}
```

**Response**:
```json
{
  "result": {
    "output": "Flow execution result",
    "status": "completed",
    "execution_time": 1.23
  }
}
```

**Example**:
```python
import httpx

client = httpx.Client()
response = client.post(
    "http://agentflow-service:9001/run",
    json={
        "flow_id": "processes/sgmi_controlm_pipeline",
        "input_value": "Process data",
        "inputs": {},
        "ensure": true
    }
)
result = response.json()
```

---

### List Flows

**Endpoint**: `GET /flows`

**Response**:
```json
{
  "flows": [
    {
      "id": "processes/sgmi_controlm_pipeline",
      "name": "SGMI Control-M Pipeline",
      "description": "Processes Control-M jobs and SQL queries"
    }
  ]
}
```

---

### Get Flow

**Endpoint**: `GET /flows/{flow_id}`

**Response**:
```json
{
  "id": "processes/sgmi_controlm_pipeline",
  "name": "SGMI Control-M Pipeline",
  "definition": {...},
  "metadata": {...}
}
```

---

## Go CLI Usage

### Probe Service

```bash
cd services/agentflow
go run ./cmd/flow-run --probe
```

### Run Flow

```bash
go run ./cmd/flow-run \
  --flow-id processes/sample_reconciliation \
  --input 'Reconcile ledger 123'
```

### Sync Flows

```bash
go run ./cmd/flow-run --sync
```

---

## Flow Definition Format

Flows are defined in JSON format following LangFlow specifications:

```json
{
  "name": "Flow Name",
  "description": "Flow description",
  "nodes": [
    {
      "id": "node1",
      "type": "ChatInput",
      "data": {
        "input_value": "{{input}}"
      }
    },
    {
      "id": "node2",
      "type": "LLMChain",
      "data": {
        "model": "gpt-4"
      }
    }
  ],
  "edges": [
    {
      "source": "node1",
      "target": "node2"
    }
  ]
}
```

**Location**: `services/agentflow/flows/`

---

## Integration Examples

### Example 1: Execute Flow from Graph Service

```go
// In unified workflow
state := map[string]any{
    "agentflow_request": &AgentFlowRunRequest{
        FlowID:     "processes/sgmi_controlm_pipeline",
        InputValue: "Process SGMI data",
        Inputs: map[string]any{
            "knowledge_graph_context": kgResults,
            "orchestration_result":    orchResults,
        },
        Ensure: true,
    },
}

afNode := RunAgentFlowFlowNode("http://agentflow-service:9001")
result, err := afNode(ctx, state)
```

### Example 2: Execute Flow from DeepAgents

```python
# Agent automatically uses the tool
{
  "messages": [
    {
      "role": "user",
      "content": "Run the reconciliation flow for ledger 123"
    }
  ]
}
```

### Example 3: Direct HTTP Call

```python
import httpx

client = httpx.Client()

response = client.post(
    "http://agentflow-service:9001/run",
    json={
        "flow_id": "processes/sample_reconciliation",
        "input_value": "Reconcile ledger 123",
        "inputs": {
            "ledger_id": "123",
            "date_range": "2024-01-01 to 2024-01-31"
        },
        "ensure": true
    }
)

result = response.json()
print(result["result"]["output"])
```

---

## Configuration

### Required Environment Variables

```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
```

### Service Configuration

The FastAPI service can be configured via environment variables:

```bash
# Service port (default: 9001)
AGENTFLOW_PORT=9001

# LangFlow backend URL (if using external LangFlow)
LANGFLOW_URL=http://langflow:7860
```

---

## Error Handling

### Flow Not Found

If `ensure: true` and flow doesn't exist, the service will attempt to create it from the flows directory.

**Error Response**:
```json
{
  "error": "Flow not found: processes/nonexistent",
  "code": "FLOW_NOT_FOUND"
}
```

### Execution Errors

**Error Response**:
```json
{
  "error": "Flow execution failed",
  "code": "EXECUTION_ERROR",
  "details": "Error message from LangFlow"
}
```

---

## Integration with Other Services

### From Graph Service

**File**: `services/graph/pkg/workflows/agentflow_processor.go`

The graph service integrates AgentFlow into unified workflows, passing knowledge graph and orchestration results as inputs.

### From DeepAgents

**File**: `services/deepagents/tools/agentflow_tool.py`

DeepAgents can execute flows as part of agent tool calls.

### From Browser Gateway

**Endpoint**: `/agentflow/run`

The browser gateway provides a proxy endpoint for AgentFlow execution.

---

## Best Practices

1. **Use Descriptive Flow IDs**: Use hierarchical naming like `processes/sgmi_pipeline`
2. **Validate Inputs**: Ensure flow inputs match expected format
3. **Handle Errors**: Check for execution errors and handle gracefully
4. **Use Ensure Flag**: Set `ensure: true` to automatically create missing flows
5. **Monitor Execution**: Track flow execution times and results

---

## Troubleshooting

### Flow Not Found

1. Check flow exists in `services/agentflow/flows/`
2. Verify flow ID matches file path (without .json extension)
3. Use `ensure: true` to auto-create flows

### Execution Failures

1. Check LangFlow backend is running (if using external)
2. Verify flow JSON is valid
3. Check input format matches flow expectations
4. Review service logs for detailed errors

### Service Unavailable

1. Check health endpoint: `curl http://agentflow-service:9001/healthz`
2. Verify service is running
3. Check network connectivity

---

## References

- [AgentFlow README](./README.md)
- [LangFlow Documentation](https://docs.langflow.org/)

