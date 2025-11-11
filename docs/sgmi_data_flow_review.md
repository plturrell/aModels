# SGMI Data Flow Review

## Overview

This document provides a comprehensive review of the SGMI (SAP Global Master Index) data flow from extraction through storage systems (Postgres, Redis, Neo4j) into the training service, including integration with AgentFlow and Open Deep Research.

## Data Flow Architecture

### Phase 1: Extraction (Extract Service)

**Entry Point**: `services/extract/scripts/run_sgmi_full_graph.sh`

The extraction process begins with the SGMI full graph script which:
1. Validates required input files (JSON tables, Hive DDLs, Control-M XML)
2. Builds view lineage metadata using Python script
3. Submits payload to extract service `/knowledge-graph` endpoint
4. Validates response and checks Postgres replication

**Key Code References**:
- `services/extract/main.go:1387-1467`: Main knowledge graph extraction handler
- `services/extract/main.go:940-965`: JSON table schema extraction
- `services/extract/main.go:967-1020`: Hive DDL parsing
- `services/extract/main.go:1022-1120`: Control-M XML parsing
- `services/extract/advanced_extraction.go`: Advanced extraction (table sequences, parameters, classifications)

**Data Transformations**:
1. **JSON Tables** → Nodes (tables, columns) + Edges (relationships)
2. **Hive DDLs** → Nodes (tables, views, columns) + Edges (dependencies, lineage)
3. **Control-M XML** → Nodes (jobs, schedules) + Edges (dependencies, flows)
4. **Information Theory Metrics** → Calculated and stored in root node properties

**Error Handling**:
- File validation before processing
- Graceful degradation for missing files
- Error logging with context
- HTTP status validation in response

### Phase 2: Data Population

#### 2.1 Postgres Population

**Entry Point**: `services/extract/schema_replication.go:39-43`

**Process Flow**:
1. `replicateSchema()` called after graph extraction
2. `replicateSchemaToPostgres()` executes batch inserts
3. Tables: `glean_nodes` and `glean_edges`
4. Uses batch processing (default 1000 records) for performance

**Key Code References**:
- `services/extract/schema_replication.go:178-212`: Postgres replication function
- `services/extract/schema_replication.go:214-244`: Batch node insertion
- `services/extract/schema_replication.go:246-276`: Batch edge insertion
- `services/extract/schema_replication.go:278-304`: Table creation

**Data Schema**:
```sql
CREATE TABLE glean_nodes (
    id TEXT PRIMARY KEY,
    kind TEXT,
    label TEXT,
    properties_json JSONB,
    updated_at_utc TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE glean_edges (
    source_id TEXT,
    target_id TEXT,
    label TEXT,
    properties_json JSONB,
    updated_at_utc TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_id, target_id, label)
);
```

**Additional Postgres Operations**:
- Catalog registration via `catalogClient.RegisterDataElementsBulk()` (`main.go:1387-1417`)
- Vector embeddings via `pgvector_persistence.go` (if enabled)
- Schema replication validation in `run_sgmi_full_graph.sh:146-162`

#### 2.2 Redis Population

**Entry Point**: `services/extract/schema_replication.go:27-31`

**Process Flow**:
1. `RedisPersistence.SaveSchema()` stores schema data
2. Entity extraction queue via `extract:entities` list
3. Vector storage for embeddings (if enabled)

**Key Code References**:
- `services/extract/redis.go:199-336`: Schema saving implementation
- `services/extract/redis.go:34-64`: Vector storage
- `services/extract/redis.go:104-197`: Similarity search

**Data Structures**:
- Hash keys: `schema:node:{id}` and `schema:edge:{source_id}:{target_id}:{label}`
- List: `extract:entities` (LPUSH, LTRIM to 9999)
- Vector storage: Hash with `vector` and `metadata` fields

**Error Handling**:
- Connection failure logged but non-fatal
- Pipeline operations for batch efficiency
- Graceful degradation if Redis unavailable

#### 2.3 Neo4j Population

**Entry Point**: `services/extract/main.go:1424-1436`

**Process Flow**:
1. `graphPersistence.SaveGraph()` called after extraction
2. Neo4j transaction executes MERGE operations
3. Nodes and edges stored with properties
4. Information theory metrics in root node

**Key Code References**:
- `services/extract/neo4j.go:82-190`: Graph saving implementation
- `services/extract/neo4j.go:113-136`: Node MERGE with properties
- `services/extract/neo4j.go:138-184`: Edge MERGE with relationships

**Cypher Operations**:
```cypher
MERGE (n:Node {id: $id}) 
SET n.type = $type, n.label = $label, 
    n.properties_json = $props, n.updated_at = $updated_at

MATCH (source:Node {id: $source_id}) 
MATCH (target:Node {id: $target_id}) 
MERGE (source)-[r:RELATIONSHIP]->(target) 
SET r.label = $label, r.properties_json = $props, 
    r.updated_at = $updated_at
```

**Data Properties**:
- Node: `id`, `type`, `label`, `properties_json`, `updated_at`, `agent_id`, `domain`
- Edge: `label`, `properties_json`, `updated_at`, `agent_id`, `domain`

**Error Handling**:
- Transaction rollback on failure
- Error logging with node/edge context
- Non-fatal if Neo4j unavailable (logged warning)

### Phase 3: Training Service Consumption

**Entry Point**: `services/training/pipeline.py:1332-1365`

**Process Flow**:
1. Training pipeline calls `_extract_knowledge_graph()`
2. Graph service client queries Neo4j for training data
3. Data retrieved in batches for memory efficiency
4. GNN processing uses graph data for embeddings, classification, link prediction

**Key Code References**:
- `services/training/pipeline.py:1332-1365`: Knowledge graph extraction
- `services/training/graph_client.py:135-293`: Graph data retrieval methods
- `services/training/graph_client.py:233-293`: Streaming methods for batch processing
- `services/training/pipeline.py:696-900`: GNN processing with graph data

**Data Retrieval Methods**:
1. `get_graph_for_training()`: Full graph retrieval with filters
2. `stream_nodes()`: Batch node streaming (default 1000 per batch)
3. `stream_edges()`: Batch edge streaming (default 1000 per batch)
4. `query_neo4j()`: Direct Cypher query execution

**Postgres Integration**:
- Training data extracted from Postgres catalog tables
- Column metadata for relational transformer training
- Historical data queries for temporal analysis

**Redis Integration**:
- GNN embedding cache via `gnn_cache_manager.py`
- Cache keys: `gnn:embedding:{project_id}:{hash}`
- Cache invalidation on graph updates

**Neo4j Integration**:
- Direct Cypher queries via graph service client
- Project/system ID filtering
- Batch processing for large graphs

**Error Handling**:
- Fallback to extract service if graph service unavailable
- Retry logic for transient failures
- Error logging with context
- Graceful degradation for missing data

### Phase 4: AgentFlow Integration

**Entry Point**: `services/graph/pkg/workflows/agentflow_processor.go:44-218`

**Process Flow**:
1. Unified workflow includes AgentFlow processing node
2. Knowledge graph context passed in state
3. AgentFlow flow executed with graph data
4. Results merged back into workflow state

**Key Code References**:
- `services/graph/pkg/workflows/agentflow_processor.go:44-218`: Flow execution
- `services/graph/pkg/workflows/unified_processor.go:462-508`: AgentFlow workflow integration
- `services/agentflow/flows/processes/sgmi_controlm_pipeline.json`: SGMI flow definition

**State Management**:
```go
state := map[string]any{
    "agentflow_request": map[string]any{
        "flow_id": "processes/sgmi_controlm_pipeline",
        "input_value": "Process SGMI data",
        "inputs": map[string]any{...},
    },
    "knowledge_graph": graphData,
}
```

**Knowledge Graph Context**:
- Graph data extracted from Neo4j via graph service
- Passed as input to AgentFlow flows
- Used for pipeline processing and decision-making

**Error Handling**:
- Retry logic with exponential backoff
- Timeout handling (120s default)
- Error logging with correlation ID
- Graceful degradation if AgentFlow unavailable

### Phase 5: Open Deep Research Integration

**Entry Point**: `services/catalog/research/deep_research_tool.py:14-46`

**Process Flow**:
1. Research queries submitted to deep research service
2. SPARQL queries executed against catalog
3. Knowledge graph context used for research
4. Results returned for training data enrichment

**Key Code References**:
- `services/catalog/research/deep_research_tool.py`: Research tool implementation
- `services/gateway/main.py:2727-2746`: Deep research API endpoint
- `services/catalog/autonomous/intelligence_layer.go:283-293`: Deep research integration

**Research Capabilities**:
- Metadata discovery queries
- SPARQL query execution
- Catalog search integration
- Context-aware research

**Integration Points**:
- Catalog service for SPARQL queries
- Knowledge graph for context
- Training service for data enrichment

**Error Handling**:
- HTTP timeout handling (300s)
- Error response formatting
- Graceful degradation if service unavailable

## Data Flow Diagram

```
┌─────────────────┐
│  SGMI Source    │
│  Files (JSON,   │
│  DDL, Control-M)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Extract Service │
│  /knowledge-graph│
│  - Parse files   │
│  - Build graph   │
│  - Calculate     │
│    metrics       │
└────────┬────────┘
         │
         ├─────────────────┬─────────────────┐
         ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Postgres   │  │    Redis     │  │    Neo4j     │
│  - glean_    │  │  - Schema    │  │  - Nodes    │
│    nodes     │  │    storage   │  │  - Edges     │
│  - glean_    │  │  - Vectors   │  │  - Metrics   │
│    edges     │  │  - Cache     │  │              │
│  - Catalog   │  │              │  │              │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┴─────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Training Service    │
              │  - Graph Client      │
              │  - GNN Processing    │
              │  - Pattern Learning  │
              └──────────┬───────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌─────────────────┐            ┌─────────────────┐
│   AgentFlow     │            │ Open Deep       │
│   - Flow exec   │            │ Research        │
│   - Graph ctx   │            │ - SPARQL        │
│   - Results     │            │ - Catalog       │
└─────────────────┘            └─────────────────┘
```

## Data Consistency

### Synchronization Points

1. **Extraction → Storage**: All three systems (Postgres, Redis, Neo4j) updated in parallel
2. **Storage → Training**: Training service queries Neo4j directly, uses Postgres for catalog, Redis for cache
3. **Training → AgentFlow**: Knowledge graph passed via workflow state
4. **Catalog → Deep Research**: SPARQL queries against catalog service

### Consistency Guarantees

- **Eventual Consistency**: All systems updated asynchronously
- **Idempotency**: MERGE operations ensure no duplicates
- **Validation**: Postgres validation in `run_sgmi_full_graph.sh` ensures data integrity

## Error Handling and Resilience

### Extract Service
- File validation before processing
- Graceful degradation for missing files
- Error logging with full context
- HTTP status validation

### Storage Systems
- Non-fatal errors logged but don't stop processing
- Retry logic for transient failures
- Connection pooling for performance
- Transaction rollback on Neo4j failures

### Training Service
- Fallback to extract service if graph service unavailable
- Batch processing for memory efficiency
- Cache invalidation on errors
- Graceful degradation for missing data

### AgentFlow
- Retry with exponential backoff
- Timeout handling
- Error logging with correlation ID

### Open Deep Research
- HTTP timeout handling
- Error response formatting
- Graceful degradation

## Performance Characteristics

### Extraction Phase
- File parsing: Sequential processing
- Graph building: In-memory operations
- Metrics calculation: O(n) where n = number of nodes/edges

### Storage Phase
- Postgres: Batch inserts (1000 records per batch)
- Redis: Pipeline operations for efficiency
- Neo4j: Single transaction for all operations

### Training Phase
- Graph retrieval: Streaming for large graphs
- GNN processing: Parallel processing enabled
- Cache: Redis for fast retrieval

## Known Limitations

1. **No Real-time Sync**: Storage systems updated asynchronously
2. **No Distributed Transactions**: Each system updated independently
3. **Cache Invalidation**: Manual invalidation required
4. **Error Recovery**: Limited automatic retry mechanisms
5. **Data Validation**: Postgres validation only in SGMI script

## Recommendations

1. **Add Data Validation**: Validate data consistency across all systems
2. **Improve Error Recovery**: Add automatic retry mechanisms
3. **Add Monitoring**: Track data flow metrics and errors
4. **Optimize Batch Sizes**: Tune batch sizes based on data volume
5. **Add Caching Strategy**: Implement cache invalidation policies

