# Graph Service Integration Guide

## Overview

The Graph service provides a LangGraph-based unified workflow processor that combines Knowledge Graph processing, Orchestration chains, AgentFlow flows, DeepAgents, and GraphRAG queries into a single orchestrated workflow.

## Service Information

- **Port**: 8081
- **Technology**: Go, LangGraph Go SDK
- **Base URL**: `http://graph-service:8081`

## Architecture

The graph service implements:

1. **Unified Workflow**: Combines all lang infrastructure components
2. **Workflow Processors**: Individual processors for each component
3. **State Management**: LangGraph state management with checkpointing
4. **Conditional Routing**: Route based on results and quality metrics

---

## Unified Workflow

**File**: `services/graph/pkg/workflows/unified_processor.go`

The unified workflow supports three execution modes:

### Sequential Mode

Components execute in order:
```
KG Processing → Orchestration → AgentFlow → DeepAgents
```

### Parallel Mode

All components execute simultaneously:
```
[KG, Orchestration, AgentFlow, DeepAgents] → Join Results
```

### Conditional Mode

Route based on results and quality:
```
KG Processing → [Quality Check] → Route to appropriate processor
```

---

## Integration Points

### 1. Knowledge Graph Processing

**File**: `services/graph/pkg/workflows/knowledge_graph_processor.go`

**Purpose**: Process knowledge graphs from extract service.

**Usage**:
```go
kgNode := ProcessKnowledgeGraphNode(extractServiceURL)
result, err := kgNode(ctx, state)
```

**State Input**:
```go
state := map[string]any{
    "knowledge_graph_request": &KnowledgeGraphRequest{
        ProjectID: "sgmi",
        SystemID: "production",
        SqlQueries: []string{"SELECT * FROM table1"},
        JSONTables: []string{...},
        HiveDDLs: []string{...},
        ControlMFiles: []string{...},
    },
}
```

**State Output**:
```go
{
    "knowledge_graph": {
        "nodes": [...],
        "edges": [...],
        "quality": {
            "score": 0.85,
            "level": "good",
            "issues": []
        }
    }
}
```

---

### 2. Orchestration Chain Execution

**File**: `services/graph/pkg/workflows/orchestration_processor.go`

**Purpose**: Execute orchestration chains with KG context.

**Usage**:
```go
orchNode := RunOrchestrationChainNode(localAIURL)
result, err := orchNode(ctx, state)
```

**State Input**:
```go
state := map[string]any{
    "orchestration_request": map[string]any{
        "chain_name": "knowledge_graph_analyzer",
        "inputs": map[string]any{
            "query": "Analyze the knowledge graph",
        },
    },
    "knowledge_graph": {...},  // Enriched automatically
}
```

**Supported Chain Types**:
- `llm_chain`, `default`: Basic LLM chain
- `question_answering`, `qa`: Context-aware Q&A
- `summarization`, `summarize`: Text summarization
- `knowledge_graph_analyzer`, `kg_analyzer`: KG analysis
- `data_quality_analyzer`, `quality_analyzer`: Data quality analysis
- `pipeline_analyzer`, `pipeline`: Pipeline analysis
- `sql_analyzer`, `sql`: SQL analysis
- `agentflow_analyzer`, `agentflow`: AgentFlow analysis

**State Output**:
```go
{
    "orchestration_result": {
        "text": "Analysis results...",
        "output": "..."
    },
    "orchestration_text": "Analysis results...",
    "orchestration_success": true
}
```

---

### 3. AgentFlow Flow Execution

**File**: `services/graph/pkg/workflows/agentflow_processor.go`

**Purpose**: Execute AgentFlow flows with KG and orchestration data.

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
            "knowledge_graph_context": kgResults,
            "orchestration_result": orchResults,
        },
        "ensure": true,
    },
}
```

**State Output**:
```go
{
    "agentflow_result": {
        "output": "Flow execution results...",
        "status": "completed"
    }
}
```

---

### 4. DeepAgents Integration

**File**: `services/graph/pkg/workflows/deepagents_processor.go`

**Purpose**: Execute deep agent analysis.

**Usage**:
```go
deepAgentNode := RunDeepAgentNode(deepAgentsServiceURL)
result, err := deepAgentNode(ctx, state)
```

**State Input**:
```go
state := map[string]any{
    "deepagents_request": map[string]any{
        "messages": [
            {"role": "user", "content": "Analyze the pipeline"}
        ],
    },
    "knowledge_graph": {...},  // Context from previous steps
}
```

---

### 5. GraphRAG Queries

**File**: `services/graph/pkg/workflows/graphrag_processor.go`

**Purpose**: Execute GraphRAG queries for knowledge retrieval.

**Usage**:
```go
graphragNode := ProcessGraphRAGNode(graphragOpts)
result, err := graphragNode(ctx, state)
```

**State Input**:
```go
state := map[string]any{
    "graphrag_request": &GraphRAGRequest{
        Query: "What are the main data pipelines?",
        Strategy: "bfs",  // or "dfs", "hybrid"
        MaxDepth: 3,
        MaxResults: 10,
    },
}
```

---

## API Endpoints

### Unified Workflow

**Endpoint**: `POST /unified/process`

**Request**:
```json
{
  "workflow_mode": "sequential",
  "knowledge_graph_request": {
    "project_id": "sgmi",
    "system_id": "production",
    "sql_queries": ["SELECT * FROM table1"]
  },
  "orchestration_request": {
    "chain_name": "knowledge_graph_analyzer",
    "inputs": {
      "query": "Analyze the knowledge graph"
    }
  },
  "agentflow_request": {
    "flow_id": "processes/sgmi_controlm_pipeline",
    "input_value": "Process data",
    "inputs": {}
  }
}
```

**Response**:
```json
{
  "knowledge_graph": {...},
  "orchestration_result": {...},
  "agentflow_result": {...},
  "unified_workflow_complete": true,
  "unified_workflow_summary": {
    "knowledge_graph_processed": true,
    "orchestration_processed": true,
    "agentflow_processed": true,
    "workflow_mode": "sequential"
  }
}
```

---

### Orchestration Chain

**Endpoint**: `POST /orchestration/process`

**Request**:
```json
{
  "knowledge_graph_query": "MATCH (n:Table) RETURN n LIMIT 10",
  "orchestration_request": {
    "chain_name": "knowledge_graph_analyzer",
    "inputs": {
      "query": "Analyze the knowledge graph"
    }
  }
}
```

**Response**:
```json
{
  "orchestration_result": {
    "text": "Analysis results...",
    "output": "..."
  },
  "orchestration_text": "Analysis results...",
  "orchestration_success": true
}
```

---

### AgentFlow Flow

**Endpoint**: `POST /agentflow/run`

**Request**:
```json
{
  "flow_id": "processes/sgmi_controlm_pipeline",
  "input_value": "Process data",
  "inputs": {
    "knowledge_graph_context": {...}
  },
  "ensure": true
}
```

---

### GraphRAG Query

**Endpoint**: `POST /graphrag/query`

**Request**:
```json
{
  "query": "What are the main data pipelines?",
  "strategy": "bfs",
  "max_depth": 3,
  "max_results": 10
}
```

---

## Integration Examples

### Example 1: Complete Unified Workflow

```go
opts := UnifiedProcessorOptions{
    ExtractServiceURL:   "http://extract-service:19080",
    AgentFlowServiceURL: "http://agentflow-service:9001",
    LocalAIURL:          "http://localai:8080",
}

workflowNode := ProcessUnifiedWorkflowNode(opts)

state := map[string]any{
    "unified_request": map[string]any{
        "workflow_mode": "sequential",
        "knowledge_graph_request": map[string]any{
            "project_id": "sgmi",
            "sql_queries": []string{"SELECT * FROM table1"},
        },
        "orchestration_request": map[string]any{
            "chain_name": "knowledge_graph_analyzer",
            "inputs": map[string]any{
                "query": "Analyze the graph",
            },
        },
    },
}

result, err := workflowNode(ctx, state)
```

### Example 2: Orchestration with KG Context

```go
orchNode := RunOrchestrationChainNode("http://localai:8080")

state := map[string]any{
    "orchestration_request": map[string]any{
        "chain_name": "data_quality_analyzer",
        "inputs": map[string]any{
            "query": "Assess data quality",
        },
    },
    "knowledge_graph": map[string]any{
        "quality": map[string]any{
            "score": 0.85,
            "level": "good",
        },
        "nodes": [...],
        "edges": [...],
    },
}

// KG context is automatically enriched into chain inputs
result, err := orchNode(ctx, state)
```

### Example 3: Parallel Execution

```go
state := map[string]any{
    "unified_request": map[string]any{
        "workflow_mode": "parallel",
        "knowledge_graph_request": {...},
        "orchestration_request": {...},
        "agentflow_request": {...},
    },
}

// All components execute simultaneously
result, err := workflowNode(ctx, state)
```

---

## Configuration

### Required Environment Variables

```bash
# Service URLs
EXTRACT_SERVICE_URL=http://extract-service:19080
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001
LOCALAI_URL=http://localai:8080

# Optional
GPU_ORCHESTRATOR_URL=http://gpu-orchestrator:port
DEEPAGENTS_SERVICE_URL=http://deepagents-service:9004
```

### Checkpoint Configuration

```bash
# SQLite (default for local dev)
CHECKPOINT=sqlite:langgraph.db

# Redis
CHECKPOINT=redis://localhost:6379/0

# HANA (requires -tags hana)
CHECKPOINT=hana
HANA_DSN=hana://user:pass@host:port
```

---

## Error Handling

### Workflow Errors

The service implements error handling at each workflow step:

```go
// Errors are captured in state
state["error"] = err.Error()
state["error_step"] = "orchestration"
```

### Conditional Routing

Errors trigger conditional routing:

```go
// Route based on results
if success, ok := state["orchestration_success"].(bool); ok && !success {
    // Route to error handling
    return []string{"error"}, nil
}
```

---

## Best Practices

1. **Use Appropriate Workflow Mode**:
   - Sequential: When steps depend on previous results
   - Parallel: When steps are independent
   - Conditional: When routing based on results

2. **Enrich Context**: Pass knowledge graph context to orchestration chains for better analysis

3. **Handle Errors**: Check `*_success` flags in workflow results

4. **Use GPU Orchestration**: Enable GPU allocation for compute-intensive workflows

5. **Monitor Performance**: Track workflow execution times and success rates

---

## Troubleshooting

### Workflow Failures

1. Check individual processor logs
2. Verify service URLs are correct
3. Check `*_success` flags in results
4. Review error messages in state

### Orchestration Chain Issues

1. Verify LocalAI is running: `curl http://localai:8080/healthz`
2. Check chain name is supported
3. Review chain input format
4. Check stubs vs real framework (see improvement plan)

### Service Unavailable

1. Check health endpoints for all services
2. Verify network connectivity
3. Review service logs
4. Check configuration variables

---

## References

- [Graph Service README](./README.md)
- [Orchestration Integration Documentation](../../docs/orchestration-langchain-integration.md)

