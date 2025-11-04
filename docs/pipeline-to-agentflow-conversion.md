# Control-M → SQL → Tables Pipeline to AgentFlow Conversion

## Overview

This document explains how to convert a Control-M → SQL → Tables pipeline extracted from Neo4j/Postgres knowledge graphs into a LangFlow flow with logical agents split across AgentFlow.

## Process Flow

```
1. Extract Knowledge Graph (Postgres/Neo4j)
   ↓
2. Query Neo4j for Control-M → SQL → Tables relationships
   ↓
3. Generate LangFlow Flow JSON with Agent nodes
   ↓
4. Import Flow into AgentFlow/LangFlow service
```

## Architecture

### Components

1. **Extract Service** - Stores knowledge graph in Neo4j
   - Control-M jobs (type: `control-m-job`)
   - SQL queries (type: `sql-query` or `sql`)
   - Tables (type: `table`)
   - Columns (type: `column`)
   - Relationships: `DATA_FLOW`, `HAS_COLUMN`, etc.

2. **Graph Service** - Pipeline conversion
   - `QueryPipelineFromGraph()` - Queries Neo4j for pipeline segments
   - `GenerateLangFlowFlow()` - Creates LangFlow flow JSON
   - `CreateFlowInAgentFlow()` - Imports flow into AgentFlow service

3. **AgentFlow Service** - LangFlow integration
   - `/flows/{flow_id}/sync` - Imports flow into LangFlow

## API Endpoint

### POST `/pipeline/to-agentflow`

Convert a Control-M → SQL → Tables pipeline from the knowledge graph into a LangFlow flow.

**Request:**
```json
{
  "project_id": "project-123",
  "system_id": "system-456",
  "flow_name": "SGMI Control-M Pipeline",
  "flow_id": "sgmi_controlm_pipeline",
  "force": false
}
```

**Response:**
```json
{
  "flow_id": "sgmi_controlm_pipeline",
  "flow_name": "SGMI Control-M Pipeline",
  "segments": [
    {
      "controlm_job": {
        "id": "control-m:JOB001",
        "name": "ETL_JOB_001",
        "description": "Daily ETL job",
        "command": "/scripts/etl.sh",
        "application": "ETL",
        "properties": {...}
      },
      "sql_queries": [
        {
          "id": "sql-query-1",
          "query": "SELECT * FROM source_table",
          "type": "SELECT",
          "properties": {...}
        }
      ],
      "source_tables": [
        {
          "id": "table-1",
          "name": "source_table",
          "type": "table",
          "columns": [...],
          "properties": {...}
        }
      ],
      "target_tables": [
        {
          "id": "table-2",
          "name": "target_table",
          "type": "table",
          "properties": {...}
        }
      ],
      "data_flow_path": [
        {
          "source": "source_column",
          "target": "target_column",
          "relationship": "DATA_FLOW"
        }
      ]
    }
  ],
  "flow_json": {
    "name": "SGMI Control-M Pipeline",
    "description": "Generated from Control-M → SQL → Tables pipeline",
    "data": {
      "nodes": [...],
      "edges": [...],
      "viewport": {...}
    }
  },
  "created": true,
  "result": {
    "local_id": "sgmi_controlm_pipeline",
    "remote_id": "langflow-flow-id",
    "status": "synced"
  }
}
```

## Agent Types

The generated flow splits the pipeline into logical agents:

### 1. Control-M Agent
- **Type**: `ControlMAgent`
- **Purpose**: Represents Control-M job execution
- **Data**: Job name, command, application, schedule, conditions

### 2. SQL Agent
- **Type**: `SQLAgent`
- **Purpose**: Executes SQL queries
- **Data**: SQL query text, query type (SELECT, INSERT, etc.)

### 3. Table Agent
- **Type**: `TableAgent`
- **Purpose**: Manages table operations
- **Data**: Source tables, target tables, data flow paths

### 4. Quality Agent
- **Type**: `QualityAgent`
- **Purpose**: Monitors data quality
- **Data**: Data flow monitoring, quality metrics

## Flow Structure

### Node Structure
```json
{
  "id": "controlm_agent_0",
  "type": "ControlMAgent",
  "template": "ControlM Agent",
  "data": {
    "type": "ControlMAgent",
    "node": {"template": "ControlM Agent"},
    "name": "ETL_JOB_001",
    "display_name": "ETL_JOB_001",
    "description": "Daily ETL job",
    "command": "/scripts/etl.sh",
    "application": "ETL",
    "properties": {...}
  },
  "position": {
    "x": 0.0,
    "y": 0.0
  }
}
```

### Edge Structure
```json
{
  "id": "edge_controlm_agent_0_sql_agent_0_0",
  "source": "controlm_agent_0",
  "target": "sql_agent_0_0",
  "sourceHandle": "output",
  "targetHandle": "input",
  "type": "default"
}
```

## Neo4j Query Strategy

### Step 1: Find Control-M Jobs
```cypher
MATCH (job:Node)
WHERE job.type = 'control-m-job'
RETURN job
LIMIT 50
```

### Step 2: For Each Job, Find Related SQL and Tables
```cypher
MATCH (job:Node {id: $job_id})
OPTIONAL MATCH (job)-[r1:RELATIONSHIP]->(sql:Node)
WHERE sql.type = 'sql-query' OR sql.type = 'sql'
OPTIONAL MATCH (sql)-[r2:RELATIONSHIP]->(table:Node {type: 'table'})
OPTIONAL MATCH (table)-[r3:RELATIONSHIP {label: 'HAS_COLUMN'}]->(col:Node {type: 'column'})
OPTIONAL MATCH (col)-[r4:RELATIONSHIP {label: 'DATA_FLOW'}]->(targetCol:Node {type: 'column'})
OPTIONAL MATCH (targetCol)<-[r5:RELATIONSHIP {label: 'HAS_COLUMN'}]-(targetTable:Node {type: 'table'})
RETURN sql, table, col, targetCol, targetTable
LIMIT 50
```

## Usage Examples

### Basic Conversion

```bash
curl -X POST http://localhost:8081/pipeline/to-agentflow \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "sgmi",
    "system_id": "production",
    "flow_name": "SGMI Pipeline",
    "flow_id": "sgmi_pipeline"
  }'
```

### Force Update Existing Flow

```bash
curl -X POST http://localhost:8081/pipeline/to-agentflow \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "sgmi",
    "system_id": "production",
    "flow_name": "SGMI Pipeline",
    "flow_id": "sgmi_pipeline",
    "force": true
  }'
```

### Via Gateway

```bash
curl -X POST http://localhost:8000/pipeline/to-agentflow \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "sgmi",
    "system_id": "production",
    "flow_name": "SGMI Pipeline"
  }'
```

## Agent Responsibilities

### Control-M Agent
- **Triggers**: Job scheduling, conditions
- **Outputs**: Job completion status, execution context

### SQL Agent
- **Inputs**: Job context, table references
- **Actions**: Execute SQL queries, transform data
- **Outputs**: Query results, execution metrics

### Table Agent
- **Inputs**: SQL query results
- **Actions**: Manage table operations, data flow
- **Outputs**: Table status, data flow information

### Quality Agent
- **Inputs**: Data flow paths, table metadata
- **Actions**: Monitor data quality, validate transformations
- **Outputs**: Quality metrics, warnings, recommendations

## Integration with LangGraph Workflows

The generated flow can be orchestrated via LangGraph workflows:

```go
// In LangGraph workflow
state["pipeline_request"] = map[string]any{
    "project_id": "sgmi",
    "system_id": "production",
    "flow_name": "SGMI Pipeline",
}

// Call conversion endpoint
result := callPipelineConversion(state)
state["agentflow_flow_id"] = result["flow_id"]
```

## Error Handling

### No Pipeline Segments Found
- **Status**: 404 Not Found
- **Response**: `{"error": "no pipeline segments found"}`

### Neo4j Query Failed
- **Status**: 500 Internal Server Error
- **Response**: `{"error": "query pipeline: <error details>"}`

### Flow Creation Failed
- **Status**: 200 OK (with `"created": false`)
- **Response**: Includes flow JSON but notes creation failure

## Configuration

### Environment Variables

```bash
# Extract service for Neo4j queries
EXTRACT_SERVICE_URL=http://extract-service:19080

# AgentFlow service for flow creation
AGENTFLOW_SERVICE_URL=http://agentflow-service:9001

# Neo4j connection (for extract service)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Workflow

1. **Extract Knowledge Graph**
   ```bash
   POST /knowledge-graph
   {
     "control_m_files": ["path/to/controlm.xml"],
     "sql_queries": ["SELECT * FROM..."],
     "project_id": "sgmi",
     "system_id": "production"
   }
   ```

2. **Convert to AgentFlow Flow**
   ```bash
   POST /pipeline/to-agentflow
   {
     "project_id": "sgmi",
     "system_id": "production",
     "flow_name": "SGMI Pipeline"
   }
   ```

3. **Run Flow**
   ```bash
   POST /agentflow/run
   {
     "flow_id": "sgmi_pipeline",
     "inputs": {...}
   }
   ```

## Benefits

✅ **Logical Separation**: Each agent has clear responsibilities
✅ **Reusability**: Agents can be reused across different flows
✅ **Maintainability**: Easy to update individual agents
✅ **Observability**: Each agent can be monitored independently
✅ **Scalability**: Agents can be scaled independently

## Next Steps

1. **Custom Agent Templates**: Define custom agent templates in LangFlow
2. **Agent Configuration**: Add agent-specific configuration
3. **Agent Chaining**: Enable more complex agent relationships
4. **Agent Monitoring**: Add monitoring and logging per agent

