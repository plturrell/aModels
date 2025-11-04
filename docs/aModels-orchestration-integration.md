# aModels Orchestration/LangChain Integration Guide

This guide explains how to use **Orchestration** (Go-native LangChain-like framework), **LangGraph** (workflow orchestration), and how they integrate with knowledge graphs and AgentFlow in aModels.

---

## Three Systems in aModels

### 1. Orchestration Framework (Go LangChain-like)

**Purpose:** Go-native framework for building LLM applications with chains, agents, and tools

**Location:** `infrastructure/third_party/orchestration/`

**What it provides:**
- **Chains** - Link components together (LLM + Prompt + Parser)
- **LLMs** - Standardized LLM interface (LocalAI, Azure, Cohere, etc.)
- **Prompts** - Dynamic prompt templates
- **Agents** - Reasoning engines with tools
- **Memory** - State management for conversations
- **Tools** - Functions agents can use

**Key Interface:**
```go
type Chain interface {
    Call(ctx context.Context, inputs map[string]any, options ...ChainCallOption) (map[string]any, error)
    GetInputKeys() []string
    GetOutputKeys() []string
}
```

**Common Chains:**
- `LLMChain` - Combines LLM + Prompt
- `SequentialChain` - Chains multiple operations in sequence
- `QuestionAnsweringChain` - Q&A with document retrieval
- `SQLDatabaseChain` - SQL query generation and execution

**Current Usage:**
- Optional in extract service (currently disabled)
- Not integrated with LangGraph workflows
- No integration with knowledge graphs or AgentFlow

---

### 2. LangGraph Workflows (Graph Service)

**Purpose:** Execute stateful agent workflows with checkpointing

**Service:** `services/graph/` - LangGraph-Go service

**What it does:**
- Executes multi-step workflows as state graphs
- Manages state transitions between nodes
- Provides checkpointing for long-running workflows
- **Now integrates with orchestration chains**

---

### 3. Knowledge Graphs (Extract Service)

**Purpose:** Process and store data relationships, schema, and metadata

**Service:** `services/extract/` - Extract service

**What it does:**
- Processes JSON tables, Hive DDLs, SQL queries
- Creates knowledge graphs with nodes and edges
- Calculates information theory metrics
- **Now provides context for orchestration chains**

---

## Integration: Using All Three Together

### Workflow: Orchestration Chain Processing with Knowledge Graph Integration

The `orchestration_processor.go` workflow demonstrates how to use LangGraph to orchestrate orchestration chains:

```go
// Workflow steps:
1. query_kg       → Query knowledge graph for chain planning
2. run_chain      → Run orchestration chain with knowledge graph context
3. analyze_result → Analyze execution results
```

**Benefits:**
- Use LangGraph's state management for complex workflows
- Leverage knowledge graph quality metrics for decision-making
- Combine orchestration chains with other services in a single workflow
- Checkpoint long-running orchestration operations

---

## Orchestration Workflow Nodes

### QueryKnowledgeGraphForChainNode

Queries knowledge graphs to inform chain execution.

**Input State:**
```json
{
  "knowledge_graph_query": "root_node",
  "project_id": "project-123",
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
  },
  "chain_inputs": {
    "knowledge_graph_context": {
      "nodes": [...],
      "edges": [...],
      "quality": {...}
    }
  }
}
```

### RunOrchestrationChainNode

Runs an orchestration chain.

**Input State:**
```json
{
  "orchestration_request": {
    "chain_name": "llm_chain",
    "inputs": {
      "question": "What tables are in the knowledge graph?",
      "knowledge_graph_context": {...}
    }
  }
}
```

**Output State:**
```json
{
  "orchestration_result": {
    "text": "The knowledge graph contains 150 nodes and 200 edges...",
    "output_keys": ["text"]
  },
  "orchestration_chain_name": "llm_chain"
}
```

### AnalyzeChainResultsNode

Analyzes orchestration chain execution results.

**Input State:** Requires `orchestration_result` from RunOrchestrationChainNode

**Output State:**
```json
{
  "orchestration_success": true,
  "orchestration_analysis": {
    "success": true,
    "chain_name": "llm_chain",
    "result": {...}
  }
}
```

---

## Complete Workflow Example

### Using LangGraph to Orchestrate Orchestration Chains with Knowledge Graph Integration

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
    workflow, err := workflows.NewOrchestrationProcessorWorkflow(
        workflows.OrchestrationProcessorOptions{
            LocalAIURL:        "http://localai:8080",
            ExtractServiceURL: "http://extract-service:19080",
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
        "orchestration_request": map[string]any{
            "chain_name": "llm_chain",
            "inputs": map[string]any{
                "question": "What tables are in the knowledge graph?",
            },
        },
    }

    // Execute workflow
    result, err := workflow.Invoke(context.Background(), input)
    if err != nil {
        panic(err)
    }

    // Check results
    success := result["orchestration_success"].(bool)
    analysis := result["orchestration_analysis"].(map[string]any)

    fmt.Printf("Orchestration chain execution: success=%v\n", success)
    fmt.Printf("Analysis: %v\n", analysis)
}
```

---

## API Endpoints Summary

### Orchestration Framework (Direct Usage)

**Go Code:**
```go
import orch "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"

// Create chain
chain := orch.NewLLMChain(llm, promptTemplate)

// Execute chain
result, err := orch.Call(ctx, chain, map[string]any{
    "input": "What is the capital of France?",
})
```

### Gateway (Proxies)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/orchestration/process` | POST | Process orchestration chains via LangGraph workflow |

### Graph Service (LangGraph Workflows)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/orchestration/process` | POST | Process orchestration chains via LangGraph workflow |
| `/knowledge-graph/process` | POST | Process knowledge graphs via LangGraph workflow |
| `/agentflow/process` | POST | Process AgentFlow flows via LangGraph workflow |

---

## Environment Variables

### Orchestration Framework
```bash
LOCALAI_URL=http://localai:8080
EXTRACT_SERVICE_URL=http://extract-service:19080
```

### Graph Service
```bash
LOCALAI_URL=http://localai:8080
EXTRACT_SERVICE_URL=http://extract-service:19080
GRAPH_SERVICE_URL=http://graph-service:8081
```

---

## Naming Clarification

### Orchestration vs LangChain vs LangGraph

**Orchestration** (Go framework):
- Go-native LangChain-like framework
- Provides chains, agents, tools, memory
- Similar concept to Python LangChain
- Location: `infrastructure/third_party/orchestration/`

**LangChain** (Python framework):
- Original Python framework (conceptually similar)
- Not used in aModels (Go-only)
- Reference point for understanding Orchestration

**LangGraph** (graph service):
- Go-based workflow orchestration
- Stateful workflow execution with checkpointing
- Integrates with orchestration chains, knowledge graphs, AgentFlow
- Service: `services/graph/`

---

## Integration Examples

### Example 1: Knowledge Graph → Orchestration Chain

```bash
# 1. Process knowledge graph
curl -X POST http://localhost:8081/knowledge-graph/process \
  -d '{"knowledge_graph_request": {"json_tables": ["data/schema.json"]}}'

# 2. Use knowledge graph results to run orchestration chain
curl -X POST http://localhost:8081/orchestration/process \
  -d '{
    "knowledge_graph": {...},
    "orchestration_request": {
      "chain_name": "llm_chain",
      "inputs": {
        "question": "What tables are in the knowledge graph?"
      }
    }
  }'
```

### Example 2: Unified Workflow (All Three Systems)

```bash
# Process with knowledge graph → orchestration chain → AgentFlow flow
curl -X POST http://localhost:8081/orchestration/process \
  -d '{
    "knowledge_graph": {...},
    "orchestration_request": {
      "chain_name": "llm_chain",
      "inputs": {
        "question": "Analyze the knowledge graph and generate a plan"
      }
    },
    "agentflow_request": {
      "flow_id": "processes/sgmi_controlm_pipeline",
      "input_value": "Execute plan from orchestration chain"
    }
  }'
```

---

## Best Practices

### 1. Use Orchestration Chains for LLM Operations

When you need to:
- Execute LLM operations with prompts
- Chain multiple LLM operations together
- Use agents with tools
- Manage conversation memory

**Use:** Orchestration framework directly or via LangGraph workflows

### 2. Use LangGraph for Workflow Orchestration

When you need to:
- Orchestrate multi-step processes
- Combine orchestration chains with other services
- Manage state across operations
- Checkpoint long-running tasks

**Use:** Graph service `/orchestration/process` endpoint

### 3. Combine All Three for Complex Operations

When you need to:
- Query knowledge graphs for chain context
- Execute orchestration chains with knowledge graph insights
- Route chains based on data quality
- Analyze results and decide next steps

**Use:** LangGraph workflows that:
- Query knowledge graphs
- Run orchestration chains
- Analyze results

---

## Current Limitations

### 1. Chain Registry Not Implemented

**Current:** `createOrchestrationChain` returns an error indicating chain creation is not fully implemented.

**Needed:**
- Chain registry/factory
- Chain configuration storage
- Dynamic chain creation

### 2. Extract Service Integration Disabled

**Current:** Orchestration is disabled in extract service (`if false`).

**Needed:**
- Enable orchestration integration
- Implement proper chain creation
- Add error handling and fallback logic

### 3. No AgentFlow Integration

**Current:** Orchestration chains are not integrated with AgentFlow flows.

**Needed:**
- LangFlow components that wrap orchestration chains
- Visual chain composition in LangFlow UI
- AgentFlow flow execution via orchestration chains

---

## Future Enhancements

### 1. Chain Registry

Implement a chain registry that:
- Stores chain configurations
- Provides chain factory methods
- Enables dynamic chain creation
- Supports chain versioning

### 2. Unified Workflow System

Long-term goal:
- Single workflow system using LangGraph
- Orchestration chains as LangGraph nodes
- AgentFlow flows as LangGraph nodes
- Unified state management and checkpointing

### 3. Intelligent Chain Selection

Future capabilities:
- Auto-select chains based on knowledge graph quality
- Use knowledge graph structure to plan workflows
- Dynamic chain routing based on data quality

---

## References

- [Orchestration and LangChain Review](./orchestration-langchain-review.md)
- [Graph and LangGraph Review](./graph-langgraph-review.md)
- [AgentFlow and LangFlow Review](./agentflow-langflow-review.md)
- [aModels Graph Integration](./aModels-graph-integration.md)
- [aModels AgentFlow Integration](./aModels-agentflow-integration.md)
- [Orchestration Framework README](../infrastructure/third_party/orchestration/README.md)

