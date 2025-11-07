# Orchestration Service Integration Guide

## Overview

The Orchestration service provides agent coordination and management for complex workflows. It implements a multi-agent system with specialized agents for data ingestion, mapping, anomaly detection, and test generation.

## Service Information

- **Technology**: Go
- **Purpose**: Agent coordination, not direct HTTP service (used internally)

## Architecture

The orchestration service provides:

1. **Agent Coordinator**: Manages multiple agents in workflows
2. **Agent Factory**: Creates and manages agent instances
3. **Specialized Agents**: Domain-specific agents for different tasks
4. **Digital Twin Integration**: Simulation and stress testing capabilities

---

## Agent Types

### 1. Data Ingestion Agent

**Purpose**: Autonomous data ingestion from source systems.

**Supported Sources**:
- Murex
- SAP GL
- BCRS
- RCO
- Axiom

**File**: `services/orchestration/agents/data_ingestion_agent.go`

**Usage**:
```go
agent := NewDataIngestionAgent(
    "ingestion-1",
    "murex",
    connector,
    mapper,
    graphClient,
    logger,
)

err := agent.Ingest(ctx, config)
```

**Configuration**:
```go
config := map[string]interface{}{
    "source_url": "http://murex:8080",
    "credentials": map[string]string{
        "username": "user",
        "password": "pass",
    },
    "tables": []string{"table1", "table2"},
}
```

---

### 2. Mapping Rule Agent

**Purpose**: Maps source schemas to knowledge graph schemas and learns from patterns.

**File**: `services/orchestration/agents/mapping_rule_agent.go`

**Usage**:
```go
agent := NewMappingRuleAgent(store, graphClient, logger)

// Learn from patterns
patterns := []MappingPattern{
    {
        SourceField: "customer_id",
        TargetField: "customer.id",
        Confidence: 0.95,
    },
}

err := agent.LearnAndUpdate(ctx, patterns)
```

---

### 3. Anomaly Detection Agent

**Purpose**: Detects anomalies in data.

**File**: `services/orchestration/agents/anomaly_detection_agent.go`

**Usage**:
```go
agent := NewAnomalyDetectionAgent(graphClient, logger)

dataPoints := []DataPoint{
    {Value: 100.0, Timestamp: time.Now()},
    {Value: 150.0, Timestamp: time.Now()},
}

anomalies, err := agent.DetectAnomalies(ctx, dataPoints)
```

---

### 4. Test Generation Agent

**Purpose**: Generates and runs test scenarios.

**File**: `services/orchestration/agents/test_generation_agent.go`

**Usage**:
```go
agent := NewTestGenerationAgent(orchestrator, logger)

options := TestGenOptions{
    Schema: schema,
    TestCount: 10,
    RunTests: true,
}

results, err := agent.GenerateAndRunTests(ctx, schema, options)
```

---

## Agent Coordinator

**Purpose**: Coordinates multiple agents in workflows.

**File**: `services/orchestration/agent_coordinator.go`

### Register Agent

```go
coordinator := NewAgentCoordinator(logger)

agent := coordinator.RegisterAgent("agent-1", "data_ingestion")
```

### Start Agent

```go
task := map[string]any{
    "source_type": "murex",
    "config": map[string]any{
        "source_url": "http://murex:8080",
    },
}

err := coordinator.StartAgent(ctx, "agent-1", task)
```

### Send Message Between Agents

```go
message := map[string]any{
    "type": "data_ready",
    "data": ingestionResults,
}

err := coordinator.SendMessage("agent-1", "agent-2", "data_ready", message)
```

### Get Agent Status

```go
status, err := coordinator.GetAgentStatus("agent-1")
```

---

## Agent System Integration

**File**: `services/orchestration/agents/integration.go`

The `AgentSystem` integrates all agents into a coordinated system:

```go
system := NewAgentSystem(
    coordinator,
    factory,
    ingestionAgents,
    mappingAgent,
    anomalyAgent,
    testAgent,
    logger,
)

// Run ingestion
err := system.RunIngestion(ctx, "murex", config)

// Run anomaly detection
anomalies, err := system.RunAnomalyDetection(ctx, dataPoints)

// Generate tests
results, err := system.GenerateTests(ctx, schema, options)

// Update mapping rules
err := system.UpdateMappingRules(ctx, patterns)
```

---

## Digital Twin Integration

**Purpose**: Create digital twins for simulation and stress testing.

**File**: `services/orchestration/digitaltwin/integration.go`

### Create Digital Twin System

```go
system := NewDigitalTwinSystem(db, logger)

// Create twin from data product
twin, err := system.CreateTwinFromDataProduct(ctx, dataProductID)
```

### Run Simulation

```go
simulationEngine := system.GetSimulationEngine()

scenario := SimulationScenario{
    Duration: 24 * time.Hour,
    Events: []Event{...},
}

results, err := simulationEngine.RunSimulation(ctx, twinID, scenario)
```

### Stress Testing

```go
stressTester := system.GetStressTester()

config := StressTestConfig{
    LoadLevel: "high",
    Duration: 1 * time.Hour,
}

results, err := stressTester.RunStressTest(ctx, twinID, config)
```

---

## API Handlers

### Agents Handler

**File**: `services/orchestration/api/agents_handler.go`

Provides HTTP endpoints for agent management (if exposed as HTTP service).

**Endpoints**:
- `POST /agents/register`: Register a new agent
- `POST /agents/{id}/start`: Start an agent
- `GET /agents/{id}/status`: Get agent status
- `POST /agents/{from}/message/{to}`: Send message between agents

---

## Integration with Other Services

### From Extract Service

The extract service uses orchestration for chain matching:

**File**: `services/extract/orchestration_integration.go`

```go
matcher := NewOrchestrationChainMatcher(logger)
matcher.SetExtractServiceURL(extractServiceURL)

chainName, score, err := matcher.MatchChainToTask(
    taskDescription,
    tableName,
    classification,
)
```

### From Graph Service

The graph service can use orchestration agents in workflows (currently uses stubs, see improvement plan).

---

## Configuration

### Environment Variables

```bash
# Graph service URL (for knowledge graph operations)
GRAPH_SERVICE_URL=http://graph-service:8081

# Database connection (for digital twins)
DATABASE_URL=postgres://user:pass@localhost/db

# Logging
LOG_LEVEL=info
```

---

## Error Handling

### Agent Errors

Agents implement retry logic with exponential backoff:

```go
retryConfig := &RetryConfig{
    MaxRetries:       3,
    InitialDelay:     2 * time.Second,
    MaxDelay:         60 * time.Second,
    BackoffMultiplier: 2.0,
}
```

### Agent Status

Agents track their status:
- `idle`: Not running
- `running`: Currently executing
- `completed`: Successfully finished
- `failed`: Execution failed
- `retrying`: Retrying after failure

---

## Best Practices

1. **Use Agent Factory**: Create agents through the factory for proper initialization
2. **Monitor Agent Status**: Regularly check agent status for long-running tasks
3. **Handle Messages**: Implement message handlers for agent communication
4. **Use Shared State**: Leverage state manager for cross-agent data sharing
5. **Retry Logic**: Configure appropriate retry settings for resilience

---

## Troubleshooting

### Agent Not Starting

1. Check agent is registered: `coordinator.GetAgentStatus(agentID)`
2. Verify task configuration is valid
3. Check logs for initialization errors

### Agent Communication Failures

1. Verify both agents are registered
2. Check message channel is not full
3. Review message format matches expectations

### Digital Twin Issues

1. Verify database connection
2. Check twin exists: `twinManager.GetTwin(twinID)`
3. Review simulation scenario configuration

---

## References

- [Lang Infrastructure Review](../../docs/lang-infrastructure-review.md)
- [Orchestration Framework](../../infrastructure/third_party/orchestration/README.md)

