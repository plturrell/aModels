# aModels Graph Integration Guide

This guide explains how to use both **Knowledge Graphs** (data relationships) and **LangGraph Workflows** (agent orchestration) in aModels, and how they work together.

---

## Two Graph Types in aModels

### 1. Knowledge Graphs (Extract Service)

**Purpose:** Process and store data relationships, schema, and metadata

**Endpoint:** `POST /knowledge-graph` (also available at `/graph` for backward compatibility)

**Service:** `services/extract/` - Extract service

**What it does:**
- Processes JSON tables, Hive DDLs, SQL queries, Control-M files
- Extracts schema information and relationships
- Creates nodes (tables, columns, jobs) and edges (relationships, data flows)
- Calculates information theory metrics (entropy, KL divergence)
- Stores in Neo4j, Glean, HANA, Redis

**Example:**
```bash
curl -X POST http://localhost:19080/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{
    "json_tables": ["data/schema.json"],
    "hive_ddls": ["CREATE TABLE users..."],
    "sql_queries": ["SELECT * FROM users"],
    "project_id": "project-123"
  }'
```

**Response:**
```json
{
  "nodes": [...],
  "edges": [...],
  "metadata_entropy": 2.345,
  "kl_divergence": 0.123,
  "quality": {
    "score": 0.85,
    "level": "good",
    "processing_strategy": "standard"
  },
  "root_node_id": "project:project-123"
}
```

---

### 2. LangGraph Workflows (Graph Service)

**Purpose:** Execute stateful agent workflows with checkpointing

**Endpoint:** `POST /knowledge-graph/process` (workflow orchestration)

**Service:** `services/graph/` - LangGraph-Go service

**What it does:**
- Executes multi-step workflows as state graphs
- Manages state transitions between nodes
- Provides checkpointing for long-running workflows
- Integrates with extract service, postgres, search, localai

**Example:**
```bash
curl -X POST http://localhost:8081/knowledge-graph/process \
  -H "Content-Type: application/json" \
  -d '{
    "knowledge_graph_request": {
      "json_tables": ["data/schema.json"],
      "hive_ddls": ["CREATE TABLE users..."]
    }
  }'
```

---

## Integration: Using Both Together

### Workflow: Knowledge Graph Processing with Quality-Based Routing

The `knowledge_graph_processor.go` workflow demonstrates how to use LangGraph to orchestrate knowledge graph processing:

```go
// Workflow steps:
1. process_kg       → Call extract service /knowledge-graph endpoint
2. analyze_quality  → Analyze quality metrics and decide processing strategy
3. query_kg         → Query the knowledge graph (optional)
```

**Benefits:**
- Use LangGraph's state management for complex workflows
- Leverage knowledge graph quality metrics for decision-making
- Combine multiple services in a single workflow
- Checkpoint long-running knowledge graph operations

### Example: Quality-Based Processing

```go
// The workflow automatically:
1. Processes knowledge graph → Gets quality score
2. Analyzes quality → Determines if processing should continue
3. Routes based on quality:
   - "excellent" or "good" → Continue normal processing
   - "fair" → Add validation step
   - "poor" → Add validation + review
   - "critical" → Skip processing, flag for review
```

---

## Knowledge Graph Workflow Nodes

### ProcessKnowledgeGraphNode

Processes knowledge graphs using the extract service.

**Input State:**
```json
{
  "knowledge_graph_request": {
    "json_tables": ["path/to/file.json"],
    "hive_ddls": ["CREATE TABLE..."],
    "sql_queries": ["SELECT * FROM..."],
    "project_id": "project-123"
  }
}
```

**Output State:**
```json
{
  "knowledge_graph": {
    "nodes": [...],
    "edges": [...],
    "metadata_entropy": 2.345,
    "kl_divergence": 0.123,
    "quality": {
      "score": 0.85,
      "level": "good",
      "processing_strategy": "standard"
    },
    "root_node_id": "project:project-123"
  },
  "knowledge_graph_quality": {...},
  "knowledge_graph_nodes": [...],
  "knowledge_graph_edges": [...],
  "warnings": []
}
```

### AnalyzeKnowledgeGraphQualityNode

Analyzes quality metrics and decides processing strategy.

**Input State:** Requires `knowledge_graph` from ProcessKnowledgeGraphNode

**Output State:**
```json
{
  "should_process_kg": true,
  "should_validate_kg": false,
  "should_review_kg": false,
  "processing_strategy": "standard"
}
```

### QueryKnowledgeGraphNode

Queries the knowledge graph (placeholder for Neo4j integration).

**Input State:**
```json
{
  "knowledge_graph_query": "root_node"
}
```

**Output State:**
```json
{
  "knowledge_graph_query_results": [...]
}
```

---

## Complete Workflow Example

### Using LangGraph to Process Knowledge Graph with Quality Checks

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
    workflow, err := workflows.NewKnowledgeGraphProcessorWorkflow(
        workflows.KnowledgeGraphProcessorOptions{
            ExtractServiceURL: "http://extract-service:19080",
        },
    )
    if err != nil {
        panic(err)
    }

    // Prepare input
    input := map[string]any{
        "knowledge_graph_request": map[string]any{
            "json_tables": []string{"data/schema.json"},
            "hive_ddls": []string{"CREATE TABLE users..."},
            "sql_queries": []string{"SELECT * FROM users"},
            "project_id": "project-123",
        },
    }

    // Execute workflow
    result, err := workflow.Run(context.Background(), input)
    if err != nil {
        panic(err)
    }

    // Check quality
    quality := result["knowledge_graph_quality"].(map[string]any)
    level := quality["level"].(string)
    score := quality["score"].(float64)

    fmt.Printf("Knowledge graph processed: quality=%s (%.2f)\n", level, score)
    
    // Check if processing should continue
    shouldProcess := result["should_process_kg"].(bool)
    if !shouldProcess {
        fmt.Println("WARNING: Knowledge graph quality is too low to process")
        return
    }

    // Continue with processing...
}
```

---

## API Endpoints Summary

### Extract Service (Knowledge Graphs)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/knowledge-graph` | POST | Process knowledge graph (primary) |
| `/graph` | POST | Process knowledge graph (legacy alias) |
| `/extract` | POST | Extract entities from documents |
| `/catalog/projects` | GET | List projects |

### Graph Service (LangGraph Workflows)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/knowledge-graph/process` | POST | Process knowledge graph via LangGraph workflow |
| `/run` | POST | Run generic LangGraph workflow |
| `/runs/{id}` | GET | Get workflow run status |
| `/extract/graph` | GET | Get extract graph via Arrow Flight |
| `/agent/catalog` | GET | Get agent catalog |

---

## Environment Variables

### Extract Service
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
NEO4J_URI=bolt://neo4j:7687
GLEAN_EXPORT_DIR=/data/glean
```

### Graph Service
```bash
EXTRACT_SERVICE_URL=http://extract-service:19080
EXTRACT_GRPC_ADDR=extract-service:9090
EXTRACT_FLIGHT_ADDR=extract-service:8815
POSTGRES_GRPC_ADDR=postgres-service:5432
AGENTSDK_FLIGHT_ADDR=localhost:8815
```

---

## Best Practices

### 1. Use Knowledge Graphs for Data Relationships

When you need to:
- Understand data lineage
- Track schema changes
- Analyze data quality
- Store metadata relationships

**Use:** Extract service `/knowledge-graph` endpoint

### 2. Use LangGraph for Workflow Orchestration

When you need to:
- Execute multi-step processes
- Manage state across operations
- Checkpoint long-running tasks
- Route based on conditions

**Use:** Graph service `/knowledge-graph/process` or custom workflows

### 3. Combine Both for Complex Operations

When you need to:
- Process knowledge graphs with quality checks
- Route processing based on metrics
- Chain multiple services together
- Handle failures gracefully

**Use:** LangGraph workflows that call knowledge graph endpoints

---

## Migration Guide

### From `/graph` to `/knowledge-graph`

The extract service now prefers `/knowledge-graph` but maintains `/graph` for backward compatibility:

**Old (still works):**
```bash
curl -X POST http://localhost:19080/graph -d '...'
```

**New (recommended):**
```bash
curl -X POST http://localhost:19080/knowledge-graph -d '...'
```

**Scripts Updated:**
- `scripts/run_sgmi_full_graph.sh` → Uses `/knowledge-graph`
- `scripts/run_sgmi_pipeline_graph.sh` → Uses `/knowledge-graph`

---

## Future Enhancements

### 1. Neo4j Integration for QueryKnowledgeGraphNode

Currently, `QueryKnowledgeGraphNode` is a placeholder. Future enhancement:
- Integrate with Neo4j to query stored knowledge graphs
- Support Cypher queries
- Enable graph traversal and pattern matching

### 2. Agent Tools for Knowledge Graphs

Create LangGraph tools that:
- Query knowledge graphs to answer questions
- Analyze data lineage
- Validate schema consistency
- Generate documentation from knowledge graphs

### 3. Quality-Based Routing

Enhance workflows to:
- Automatically route to different processors based on quality
- Trigger validation pipelines for low-quality graphs
- Send high-quality graphs to production immediately

---

## References

- [Graph and LangGraph Review](./graph-langgraph-review.md)
- [Information Theory Metrics](./extract-metrics.md)
- [Metrics Interpretation](./metrics-interpretation.md)
- [Extract Service README](../services/extract/README.md)
- [Graph Service README](../services/graph/README.md)

