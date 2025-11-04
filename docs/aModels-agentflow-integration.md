# aModels AgentFlow/LangFlow Integration Guide

This guide explains how to use **AgentFlow** (flow management service), **LangFlow** (external visual flow builder), and **LangGraph** (workflow orchestration) in aModels, and how they work together.

---

## Three Systems in aModels

### 1. AgentFlow Service

**Purpose:** Manage flow catalogs and sync with external LangFlow instance

**Service:** `services/agentflow/` - AgentFlow service

**What it does:**
- Manages local flow definitions (JSON files)
- Syncs flows to external LangFlow instance
- Executes flows via LangFlow API
- Maintains flow registry (local ↔ remote mappings)

**Endpoints:**
- `GET /flows` - List all flows
- `GET /flows/{id}` - Get flow details
- `POST /flows/{id}/sync` - Sync flow to LangFlow
- `POST /flows/{id}/run` - Run flow via LangFlow

**Example:**
```bash
# Sync a flow to LangFlow
curl -X POST http://localhost:9001/flows/processes/sgmi_controlm_pipeline/sync \
  -H "Content-Type: application/json" \
  -d '{"force": true}'

# Run a flow
curl -X POST http://localhost:9001/flows/processes/sgmi_controlm_pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "input_value": "Process SGMI data",
    "ensure": true
  }'
```

---

### 2. LangFlow (External Service)

**Purpose:** Visual flow builder for creating agent workflows

**What it does:**
- Provides UI for designing flows visually
- Executes flows with LLM agents
- Manages flow state and chat history
- Stores flow definitions remotely

**Integration:**
- AgentFlow syncs local flows to LangFlow
- LangFlow executes flows and returns results
- AgentFlow manages the local ↔ remote mapping

---

### 3. LangGraph Workflows (Graph Service)

**Purpose:** Execute stateful agent workflows with checkpointing

**Service:** `services/graph/` - LangGraph-Go service

**What it does:**
- Executes multi-step workflows as state graphs
- Manages state transitions between nodes
- Provides checkpointing for long-running workflows
- Integrates with extract, postgres, search, localai, **and now AgentFlow**

**New Integration:**
- LangGraph workflows can orchestrate AgentFlow flows
- Quality-based routing using knowledge graph metrics
- Stateful processing of AgentFlow operations

---

## Integration: Using All Three Together

### Workflow: AgentFlow Processing with Knowledge Graph Integration

The `agentflow_processor.go` workflow demonstrates how to use LangGraph to orchestrate AgentFlow flows:

```go
// Workflow steps:
1. query_kg       → Query knowledge graph for flow planning
2. run_flow       → Run AgentFlow flow via LangFlow
3. analyze_result → Analyze execution results
```

**Benefits:**
- Use LangGraph's state management for complex workflows
- Leverage knowledge graph quality metrics for decision-making
- Combine AgentFlow flows with other services in a single workflow
- Checkpoint long-running AgentFlow operations

### Example: Quality-Based Flow Routing

```go
// The workflow automatically:
1. Queries knowledge graph → Gets quality metrics
2. Routes to appropriate AgentFlow flow based on quality:
   - "excellent" → Production flow
   - "fair" → Validation flow
   - "poor" → Review flow
   - "critical" → Skip flow
3. Executes flow via LangFlow
4. Analyzes results and decides next steps
```

---

## AgentFlow Workflow Nodes

### QueryKnowledgeGraphForFlowNode

Queries knowledge graphs to inform flow execution.

**Input State:**
```json
{
  "knowledge_graph_query": "root_node",
  "project_id": "project-123",
  "system_id": "system-456",
  "knowledge_graph": {
    "nodes": [...],
    "edges": [...],
    "quality": {...}
  }
}
```

**Output State:**
```json
{
  "knowledge_graph_query_results": {
    "query": "root_node",
    "node_count": 150,
    "edge_count": 200,
    "quality": {
      "score": 0.85,
      "level": "good"
    }
  }
}
```

### RunAgentFlowFlowNode

Runs an AgentFlow flow via LangFlow.

**Input State:**
```json
{
  "agentflow_request": {
    "flow_id": "processes/sgmi_controlm_pipeline",
    "input_value": "Process SGMI data",
    "inputs": {...},
    "ensure": true
  }
}
```

**Output State:**
```json
{
  "agentflow_result": {
    "local_id": "processes/sgmi_controlm_pipeline",
    "remote_id": "langflow-flow-123",
    "result": {...}
  },
  "agentflow_local_id": "processes/sgmi_controlm_pipeline",
  "agentflow_remote_id": "langflow-flow-123"
}
```

### AnalyzeFlowResultsNode

Analyzes AgentFlow execution results.

**Input State:** Requires `agentflow_result` from RunAgentFlowFlowNode

**Output State:**
```json
{
  "agentflow_success": true,
  "agentflow_analysis": {
    "success": true,
    "result": {...}
  }
}
```

---

## Complete Workflow Example

### Using LangGraph to Orchestrate AgentFlow with Knowledge Graph Integration

```go
package main

import (
    "context"
    "encoding/json"
    "os"
    
    "github.com/langchain-ai/langgraph-go/pkg/workflows"
)

func main() {
    // Create workflow
    workflow, err := workflows.NewAgentFlowProcessorWorkflow(
        workflows.AgentFlowProcessorOptions{
            AgentFlowServiceURL: "http://agentflow-service:9001",
            ExtractServiceURL:   "http://extract-service:19080",
        },
    )
    if err != nil {
        panic(err)
    }

    // Prepare input
    input := map[string]any{
        "knowledge_graph_query": "root_node",
        "project_id": "sgmi-project",
        "knowledge_graph": map[string]any{
            "nodes": []any{...},
            "edges": []any{...},
            "quality": map[string]any{
                "score": 0.85,
                "level": "good",
            },
        },
        "agentflow_request": map[string]any{
            "flow_id": "processes/sgmi_controlm_pipeline",
            "input_value": "Process SGMI data with quality checks",
            "ensure": true,
        },
    }

    // Execute workflow
    result, err := workflow.Invoke(context.Background(), input)
    if err != nil {
        panic(err)
    }

    // Check results
    success := result["agentflow_success"].(bool)
    analysis := result["agentflow_analysis"].(map[string]any)

    fmt.Printf("AgentFlow execution: success=%v\n", success)
    fmt.Printf("Analysis: %v\n", analysis)
}
```

---

## API Endpoints Summary

### AgentFlow Service (Flow Management)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/flows` | GET | List all flows |
| `/flows/{id}` | GET | Get flow details |
| `/flows/{id}/sync` | POST | Sync flow to LangFlow |
| `/flows/{id}/run` | POST | Run flow via LangFlow |

### Gateway (Proxies)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/agentflow/run` | POST | Run AgentFlow flow (proxy to AgentFlow service) |
| `/agentflow/process` | POST | Process AgentFlow flows via LangGraph workflow |

### Graph Service (LangGraph Workflows)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/agentflow/process` | POST | Process AgentFlow flows via LangGraph workflow |
| `/knowledge-graph/process` | POST | Process knowledge graphs via LangGraph workflow |
| `/run` | POST | Run generic LangGraph workflow |

---

## Environment Variables

### AgentFlow Service
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
AGENTFLOW_LANGFLOW_URL=http://langflow-instance:7860
AGENTFLOW_LANGFLOW_API_KEY=your-api-key
AGENTFLOW_FLOWS_DIR=/path/to/flows
```

### Graph Service
```bash
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
EXTRACT_SERVICE_URL=http://extract-service:19080
GRAPH_SERVICE_URL=http://graph-service:8081
```

---

## Best Practices

### 1. Use AgentFlow for Flow Management

When you need to:
- Manage flow definitions locally
- Sync flows to external LangFlow instance
- Execute flows via LangFlow API

**Use:** AgentFlow service endpoints (`/flows/*`)

### 2. Use LangGraph for Workflow Orchestration

When you need to:
- Orchestrate multi-step processes
- Combine AgentFlow flows with other services
- Manage state across operations
- Checkpoint long-running tasks

**Use:** Graph service `/agentflow/process` endpoint

### 3. Combine All Three for Complex Operations

When you need to:
- Query knowledge graphs for flow planning
- Route flows based on data quality
- Execute flows with stateful orchestration
- Analyze results and decide next steps

**Use:** LangGraph workflows that:
- Query knowledge graphs
- Run AgentFlow flows
- Analyze results

---

## Naming Clarification

### AgentFlow vs LangFlow

**AgentFlow** (aModels service):
- Manages flow catalogs
- Syncs flows to LangFlow
- Provides HTTP API for flow operations
- Service: `services/agentflow/`

**LangFlow** (external service):
- Visual flow builder UI
- Executes flows with LLM agents
- Managed separately from aModels
- Accessed via AgentFlow service

**LangGraph** (graph service):
- Go-based workflow orchestration
- Stateful workflow execution
- Integrates with AgentFlow and knowledge graphs
- Service: `services/graph/`

---

## Integration Examples

### Example 1: Knowledge Graph → AgentFlow Flow

```bash
# 1. Process knowledge graph
curl -X POST http://localhost:8081/knowledge-graph/process \
  -d '{"knowledge_graph_request": {"json_tables": ["data/schema.json"]}}'

# 2. Use knowledge graph results to run AgentFlow flow
curl -X POST http://localhost:8081/agentflow/process \
  -d '{
    "knowledge_graph": {...},
    "agentflow_request": {
      "flow_id": "processes/sgmi_controlm_pipeline",
      "input_value": "Process based on knowledge graph"
    }
  }'
```

### Example 2: Quality-Based Flow Routing

The workflow automatically routes based on knowledge graph quality:
- High quality → Production flow
- Medium quality → Validation flow
- Low quality → Review flow
- Critical issues → Skip flow

---

## Future Enhancements

### 1. Direct Knowledge Graph Integration

Currently, knowledge graphs are queried via state. Future enhancement:
- Direct query endpoints in AgentFlow flows
- Knowledge graph components in LangFlow UI
- Automated flow generation from knowledge graphs

### 2. Unified Workflow System

Long-term goal:
- Single workflow system using LangGraph
- AgentFlow flows as LangGraph nodes
- Unified state management and checkpointing

### 3. Intelligent Flow Composition

Future capabilities:
- Auto-generate flows from knowledge graphs
- Use knowledge graph structure to plan workflows
- Dynamic flow routing based on data quality

---

## References

- [AgentFlow and LangFlow Review](./agentflow-langflow-review.md)
- [Graph and LangGraph Review](./graph-langgraph-review.md)
- [aModels Graph Integration](./aModels-graph-integration.md)
- [AgentFlow Service README](../services/agentflow/README.md)
- [Graph Service README](../services/graph/README.md)

