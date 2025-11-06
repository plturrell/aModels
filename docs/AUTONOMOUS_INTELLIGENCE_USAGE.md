# Autonomous Intelligence Layer - Usage Guide

## Quick Start

### 1. Prerequisites

Ensure all services are running:

```bash
# Deep Research Service
DEEP_RESEARCH_URL=http://localhost:8085

# DeepAgents Service
DEEPAGENTS_URL=http://deepagents-service:9004

# Unified Workflow (Graph Service)
GRAPH_SERVICE_URL=http://graph-service:8081

# Database for Goose migrations
CATALOG_DATABASE_URL=postgres://user:pass@localhost/catalog
```

### 2. Run Database Migrations

```bash
cd services/catalog
goose -dir migrations postgres "postgres://user:pass@localhost/catalog?sslmode=disable" up
```

This will create the following tables:
- `autonomous_task_executions`
- `autonomous_agent_performance`
- `autonomous_learned_patterns`
- `autonomous_agent_interactions`
- `autonomous_optimizations`
- `autonomous_knowledge_base`

### 3. Start Catalog Service

```bash
cd services/catalog
go run main.go
```

The service will start on port 8084 by default.

## API Usage

### Execute Autonomous Task

**Endpoint**: `POST /api/autonomous/execute`

**Request**:
```bash
curl -X POST http://localhost:8084/api/autonomous/execute \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_001",
    "type": "data_quality_analysis",
    "description": "Analyze data quality for customer data",
    "query": "What are the data quality issues for customer data?",
    "context": {
      "domain": "customer",
      "priority": "high"
    },
    "agent_id": "data_quality_agent_001",
    "priority": 1
  }'
```

**Response**:
```json
{
  "task_id": "task_001",
  "success": true,
  "result": {
    "data_quality_issues": [
      "Missing values in customer_name",
      "Duplicate customer IDs"
    ],
    "recommendations": [
      "Implement data validation rules",
      "Add uniqueness constraints"
    ]
  },
  "learned": [
    {
      "id": "lesson_001",
      "type": "success",
      "insight": "Data quality analysis works best with full lineage context",
      "recommendation": "Always include lineage when analyzing data quality",
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ],
  "optimized": [
    {
      "id": "opt_001",
      "type": "query_optimization",
      "description": "Optimized knowledge graph query",
      "impact": 0.15,
      "timestamp": "2025-01-15T10:30:00Z"
    }
  ]
}
```

### Get Performance Metrics

**Endpoint**: `GET /api/autonomous/metrics`

```bash
curl http://localhost:8084/api/autonomous/metrics
```

**Response**:
```json
{
  "average_latency": "150ms",
  "success_rate": 0.95,
  "resource_efficiency": 0.85,
  "optimization_count": 42,
  "last_optimized": "2025-01-15T10:30:00Z"
}
```

### Get Agent Registry

**Endpoint**: `GET /api/autonomous/agents`

```bash
curl http://localhost:8084/api/autonomous/agents
```

**Response**:
```json
{
  "agents": {
    "data_quality_agent_001": {
      "id": "data_quality_agent_001",
      "type": "data_quality",
      "success_rate": 0.95,
      "failure_rate": 0.05,
      "last_updated": "2025-01-15T10:30:00Z"
    }
  },
  "total": 10
}
```

### Get Knowledge Base

**Endpoint**: `GET /api/autonomous/knowledge`

```bash
curl http://localhost:8084/api/autonomous/knowledge
```

**Response**:
```json
{
  "patterns": {
    "pattern_001": {
      "id": "pattern_001",
      "description": "Pattern for successful data quality analysis",
      "success_rate": 0.95,
      "usage_count": 25
    }
  },
  "patterns_count": 25,
  "solutions": {
    "solution_001": {
      "id": "solution_001",
      "problem_type": "data_quality",
      "effectiveness": 0.90,
      "usage_count": 15
    }
  },
  "solutions_count": 15,
  "best_practices": [
    {
      "id": "bp_001",
      "description": "Always include lineage when analyzing data quality",
      "validation_score": 0.95
    }
  ],
  "best_practices_count": 8
}
```

## Go API Usage

### Basic Usage

```go
package main

import (
    "context"
    "log"
    "os"
    
    "github.com/plturrell/aModels/services/catalog/autonomous"
    "github.com/plturrell/aModels/services/catalog/research"
)

func main() {
    logger := log.New(os.Stdout, "[autonomous] ", log.LstdFlags)
    
    // Initialize Deep Research client
    deepResearchClient := research.NewDeepResearchClient(
        "http://localhost:8085",
        logger,
    )
    
    // Initialize Autonomous Intelligence Layer
    intelligenceLayer := autonomous.NewIntelligenceLayer(
        deepResearchClient,
        "http://deepagents-service:9004",
        "http://graph-service:8081",
        true, // Goose enabled
        logger,
    )
    
    // Create and execute task
    ctx := context.Background()
    task := &autonomous.AutonomousTask{
        ID:          "task_001",
        Type:        "data_quality_analysis",
        Description: "Analyze data quality",
        Query:       "What are data quality issues?",
        Context:     map[string]interface{}{
            "domain": "customer",
        },
    }
    
    result, err := intelligenceLayer.ExecuteAutonomousTask(ctx, task)
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Task completed: %v", result.Success)
}
```

### Integrated System Usage

```go
package main

import (
    "context"
    "database/sql"
    "log"
    "os"
    
    _ "github.com/lib/pq"
    "github.com/plturrell/aModels/services/catalog/autonomous"
    "github.com/plturrell/aModels/services/catalog/research"
)

func main() {
    logger := log.New(os.Stdout, "[autonomous] ", log.LstdFlags)
    
    // Connect to database
    db, err := sql.Open("postgres", "postgres://user:pass@localhost/catalog?sslmode=disable")
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()
    
    // Initialize Deep Research client
    deepResearchClient := research.NewDeepResearchClient(
        "http://localhost:8085",
        logger,
    )
    
    // Initialize Integrated Autonomous System
    system := autonomous.NewIntegratedAutonomousSystem(
        deepResearchClient,
        "http://deepagents-service:9004",
        "http://graph-service:8081",
        db,
        logger,
    )
    
    // Run Goose migrations
    ctx := context.Background()
    if err := system.RunGooseMigration(ctx, "migrations"); err != nil {
        log.Printf("Migration warning: %v", err)
    }
    
    // Execute task with database tracking
    task := &autonomous.AutonomousTask{
        ID:          "task_001",
        Type:        "data_quality_analysis",
        Description: "Analyze data quality",
        Query:       "What are data quality issues?",
        Context:     map[string]interface{}{
            "domain": "customer",
        },
    }
    
    result, err := system.ExecuteWithGooseMigration(ctx, task)
    if err != nil {
        log.Fatal(err)
    }
    
    log.Printf("Task completed: %v", result.Success)
    log.Printf("Lessons learned: %d", len(result.Learned))
    log.Printf("Optimizations: %d", len(result.Optimized))
}
```

## Task Types

### Data Quality Analysis

```json
{
  "type": "data_quality_analysis",
  "query": "What are the data quality issues for customer data?",
  "context": {
    "domain": "customer",
    "include_lineage": true,
    "include_quality": true
  }
}
```

### Data Lineage Discovery

```json
{
  "type": "lineage_discovery",
  "query": "What is the data lineage for customer transactions?",
  "context": {
    "start_node": "customer_transactions",
    "include_transformations": true
  }
}
```

### Schema Mapping

```json
{
  "type": "schema_mapping",
  "query": "Map source schema to target schema",
  "context": {
    "source_schema": "murex",
    "target_schema": "data_warehouse"
  }
}
```

### Anomaly Detection

```json
{
  "type": "anomaly_detection",
  "query": "Detect anomalies in customer data",
  "context": {
    "domain": "customer",
    "threshold": 0.05
  }
}
```

## Monitoring and Metrics

### Performance Metrics

The system tracks:
- Average task execution latency
- Success rate
- Resource efficiency
- Optimization count
- Last optimization time

### Agent Performance

Track individual agent performance:
- Success/failure rates
- Execution count
- Last execution time
- Performance history

### Knowledge Base Analytics

Monitor knowledge base:
- Pattern usage count
- Solution effectiveness
- Best practice validation scores

## Troubleshooting

### Deep Research Not Available

If Deep Research is unavailable, the system will:
- Continue execution without research context
- Log a warning message
- Complete task execution

### DeepAgents Not Available

If DeepAgents is unavailable:
- Task execution will fail if planning is required
- Error will be returned in response

### Unified Workflow Not Available

If Unified Workflow is unavailable:
- Task execution will fail
- Error will be returned in response

### Database Not Available

If database is unavailable:
- Task execution will continue
- Execution won't be recorded in database
- Warning will be logged

## Best Practices

1. **Always provide context**: Include relevant context in task requests
2. **Monitor metrics**: Regularly check performance metrics
3. **Review lessons learned**: Learn from system insights
4. **Apply optimizations**: Review and apply system optimizations
5. **Use appropriate task types**: Choose the right task type for your use case

## Next Steps

1. **Run Tests**: Execute integration tests
2. **Monitor Performance**: Track metrics over time
3. **Review Knowledge Base**: Learn from patterns and solutions
4. **Optimize Tasks**: Apply system optimizations
5. **Scale Agents**: Register and track multiple agents

