# LangGraph Optimization and Advanced Features

## Overview

This document describes the LangGraph optimizations and advanced features implemented in `aModels` to enable 10/10 workflow orchestration.

## Key Features Implemented

### 1. Conditional Edges

**Purpose**: Route workflow execution based on dynamic conditions (e.g., data quality).

**Implementation**:
- `QualityRoutingFunc`: Routes knowledge graph processing based on quality metrics
- Conditional edges in `NewKnowledgeGraphProcessorWorkflow`:
  - `reject` → Reject low-quality graphs
  - `skip` → Skip processing for simplified strategy
  - `review` → Flag for human review
  - `validate` → Validate before processing
  - `query` → Proceed directly to query

**Example**:
```go
conditionalEdges := []ConditionalEdgeSpec{
    {
        Source: "analyze_quality",
        PathFunc: QualityRoutingFunc,
        PathMap: map[string]string{
            "reject":  "reject_kg",
            "skip":    "skip_kg",
            "review":  "review_kg",
            "validate": "validate_kg",
            "query":   "query_kg",
        },
    },
}
```

### 2. Node Retry and Timeout Configuration

**Purpose**: Improve reliability and handle transient failures.

**Implementation**:
- Automatic retry with exponential backoff
- Per-node timeout configuration
- Configurable retry delays

**Default Configuration**:
```go
opts := []stategraph.NodeOption{
    stategraph.WithNodeRetries(2),                    // Retry failed nodes
    stategraph.WithNodeTimeout(60 * time.Second),    // 60s timeout per node
    stategraph.WithNodeRetryDelay(2 * time.Second),  // 2s delay between retries
}
```

### 3. Parallel Execution Mode

**Purpose**: Execute independent workflow branches in parallel for improved performance.

**Implementation**:
- Separate nodes for knowledge graph, orchestration, and AgentFlow processing
- Join node to aggregate results from parallel branches
- Configurable execution mode (sequential, parallel, conditional)

**Example**:
```go
// Parallel execution: all three processes run simultaneously
nodes := map[string]stategraph.NodeFunc{
    "process_kg":      ProcessKnowledgeGraphWorkflowNode(extractServiceURL),
    "process_orch":    ProcessOrchestrationWorkflowNode(localAIURL),
    "process_agentflow": ProcessAgentFlowWorkflowNode(agentflowServiceURL),
}

// Join node aggregates results
builder.AddJoinNode("join_results", JoinUnifiedResultsNode(),
    stategraph.WithJoinTimeout(30*time.Second))
```

### 4. Enhanced Workflow Builder

**Purpose**: Simplify workflow creation with advanced options.

**Implementation**:
- `BuildGraph`: Basic workflow builder (backward compatible)
- `BuildGraphWithOptions`: Advanced builder with:
  - Conditional edges support
  - State manager integration
  - Automatic retry/timeout configuration

**Usage**:
```go
workflow, err := BuildGraphWithOptions(
    "entry", "exit",
    nodes,
    edges,
    conditionalEdges,
    stateManager,
)
```

### 5. Quality-Based Routing

**Purpose**: Automatically route workflows based on data quality metrics.

**Quality Levels**:
- **excellent/good**: Direct processing → `query`
- **fair**: Validation required → `validate` → `query`
- **poor**: Review required → `review` → `query`
- **critical**: Reject → `reject`

**Routing Logic**:
```go
func QualityRoutingFunc(ctx context.Context, value any) ([]string, error) {
    state := value.(map[string]any)
    shouldProcess := state["should_process_kg"].(bool)
    processingStrategy := state["processing_strategy"].(string)
    
    if !shouldProcess {
        return []string{"reject"}, nil
    }
    if processingStrategy == "skip" {
        return []string{"skip"}, nil
    }
    // ... additional routing logic
}
```

### 6. Workflow Nodes

**Knowledge Graph Nodes**:
- `ProcessKnowledgeGraphNode`: Processes knowledge graphs
- `AnalyzeKnowledgeGraphQualityNode`: Analyzes quality metrics
- `ValidateKnowledgeGraphNode`: Validates graph structure
- `ReviewKnowledgeGraphNode`: Flags for human review
- `RejectKnowledgeGraphNode`: Rejects low-quality graphs
- `SkipKnowledgeGraphNode`: Skips processing
- `QueryKnowledgeGraphNode`: Queries Neo4j

**Unified Workflow Nodes**:
- `ProcessKnowledgeGraphWorkflowNode`: KG processing (parallel mode)
- `ProcessOrchestrationWorkflowNode`: Orchestration processing (parallel mode)
- `ProcessAgentFlowWorkflowNode`: AgentFlow processing (parallel mode)
- `JoinUnifiedResultsNode`: Aggregates parallel results

## Workflow Patterns

### Sequential Pattern
```
Entry → Node1 → Node2 → Node3 → Exit
```

### Conditional Pattern
```
Entry → Node1 → [Conditional] → Node2a / Node2b → Exit
```

### Parallel Pattern
```
Entry → [Node1, Node2, Node3] → Join → Exit
```

### Loop Pattern
```
Entry → Node1 → [Conditional] → Loop → Node1 / Exit
```

## State Management

### State Persistence
- Optional state manager for checkpointing
- Enables workflow resumption after failures
- Supports Postgres, SQLite, Redis backends

**Usage**:
```go
stateManager := graph.NewStateManager(...)
builder.UseStateManager(stateManager)
```

### State Merging
- Automatic state merging in join nodes
- Preserves all state from parallel branches
- Adds unified summary metadata

## Performance Optimizations

### 1. Parallel Execution
- Independent branches run concurrently
- Reduces total execution time
- Join node synchronizes results

### 2. Retry Logic
- Automatic retry on transient failures
- Exponential backoff between retries
- Configurable retry limits

### 3. Timeout Management
- Per-node timeout configuration
- Prevents hanging workflows
- Graceful timeout handling

### 4. Conditional Short-Circuiting
- Skip unnecessary processing steps
- Route based on early quality checks
- Reduce computation for low-quality data

## Error Handling

### Node-Level Errors
- Retry on transient failures
- Log errors with context
- Propagate fatal errors

### Workflow-Level Errors
- Graceful degradation
- Partial results preservation
- Error state in workflow summary

## Monitoring and Observability

### Workflow Metrics
- Node execution time
- Retry counts
- Success/failure rates
- State transitions

### Logging
- Structured logging at each node
- State transitions logged
- Quality metrics logged

## Usage Examples

### Basic Workflow
```go
workflow, err := workflows.NewKnowledgeGraphProcessorWorkflow(
    workflows.KnowledgeGraphProcessorOptions{
        ExtractServiceURL: "http://extract-service:19080",
    },
)
result, err := workflow.Invoke(ctx, request)
```

### Parallel Workflow
```go
workflow, err := workflows.NewUnifiedProcessorWorkflow(
    workflows.UnifiedProcessorOptions{
        ExtractServiceURL:   "http://extract-service:19080",
        AgentFlowServiceURL: "http://agentflow-service:9001",
        LocalAIURL:          "http://localai:8080",
    },
)
// Workflow automatically uses parallel mode when multiple requests are present
result, err := workflow.Invoke(ctx, request)
```

### Custom Workflow with Conditional Edges
```go
nodes := map[string]stategraph.NodeFunc{
    "process": ProcessNode(),
    "analyze": AnalyzeNode(),
    "validate": ValidateNode(),
    "reject": RejectNode(),
}

edges := []EdgeSpec{
    {From: "process", To: "analyze"},
}

conditionalEdges := []ConditionalEdgeSpec{
    {
        Source: "analyze",
        PathFunc: CustomRoutingFunc,
        PathMap: map[string]string{
            "validate": "validate",
            "reject":   "reject",
        },
    },
}

workflow, err := BuildGraphWithOptions(
    "process", "validate",
    nodes, edges, conditionalEdges, nil,
)
```

## Best Practices

1. **Use Conditional Edges**: Route based on data quality or business logic
2. **Enable Retries**: Configure appropriate retry counts for transient failures
3. **Set Timeouts**: Prevent workflows from hanging indefinitely
4. **Parallel Execution**: Use parallel mode for independent operations
5. **State Management**: Enable checkpointing for long-running workflows
6. **Error Handling**: Implement graceful degradation in nodes
7. **Monitoring**: Log state transitions and metrics

## Future Enhancements

- [ ] Workflow versioning
- [ ] A/B testing support
- [ ] Workflow templates
- [ ] Visual workflow builder
- [ ] Advanced checkpointing strategies
- [ ] Workflow scheduling
- [ ] Dynamic workflow composition

