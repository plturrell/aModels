# Neo4j Knowledge Graph Schema

## Overview

This document describes the complete schema for the SGMI knowledge graph stored in Neo4j, including all node types, relationship types, and their properties. The schema captures data lineage, ETL transformations, Control-M workflows, and multi-step pipeline information.

## Node Structure

All nodes in the graph share a common base structure:

### Base Node Label: `Node`

**Required Properties:**
- `id` (String): Unique identifier for the node
- `type` (String): Node type (see Node Types below)
- `label` (String): Human-readable label
- `properties_json` (String): JSON string containing all node-specific properties
- `updated_at` (String): ISO 8601 timestamp of last update

**Optional Properties:**
- `agent_id` (String): Agent identifier (if applicable)
- `domain` (String): Domain identifier (if applicable)

## Node Types

### 1. Column Nodes (`type: "column"`)

Represents database/view columns with transformation metadata.

**Properties (stored in `properties_json`):**
- `table_name` (String): Parent table/view name
- `transformation_type` (String): Type of transformation
  - Values: `"aggregation"`, `"join"`, `"filter"`, `"cast"`, `"conditional"`, `"direct_copy"`, `"transformed"`, `"function"`
- `function` (String): Function name if applicable (e.g., `"SUM"`, `"COUNT"`, `"CASE"`, `"CAST"`)
- `expression` (String): SQL expression that generates the column
- `source_columns` (Array[String]): List of source column IDs (format: `"table.column"`)
- `sql_query_id` (String): Link to SQL query in vector storage (format: `"sql:<hash>"`)
- `aggregation_keys` (Array[String]): GROUP BY columns (if aggregated)
- `inferred_type` (String): Inferred data type from expression analysis
- `view_lineage_hash` (String): Hash of view definition (if from view)
- `view_joins` (Array[Object]): Join information from view lineage

**Example:**
```cypher
(:Node {
  id: "my_view.total_amount",
  type: "column",
  label: "total_amount",
  properties_json: '{"table_name":"my_view","transformation_type":"aggregation","function":"SUM","expression":"SUM(orders.amount)","source_columns":["orders.amount"],"sql_query_id":"sql:abc123"}'
})
```

### 2. Table Nodes (`type: "table"`)

Represents database tables.

**Properties (stored in `properties_json`):**
- `schema` (String): Schema name (if applicable)
- `properties` (Object): Table properties from DDL
- `primary_key` (Array[String]): Primary key columns
- `partitioned_by` (Array[Object]): Partitioning information

**Example:**
```cypher
(:Node {
  id: "orders",
  type: "table",
  label: "orders",
  properties_json: '{"schema":"public","primary_key":["order_id"]}'
})
```

### 3. Control-M Job Nodes (`type: "control-m-job"`)

Represents Control-M job definitions.

**Properties (stored in `properties_json`):**
- `job_name` (String): Job name
- `host` (String): Execution host
- `command` (String): Command to execute
- `application` (String): Application name
- `sub_application` (String): Sub-application name
- `priority` (String): Job priority
- `run_as` (String): User to run as
- `created_by` (String): Creator
- Additional Control-M specific properties

**Example:**
```cypher
(:Node {
  id: "control-m:SGM_GBL_DATHSKP",
  type: "control-m-job",
  label: "SGM_GBL_DATHSKP",
  properties_json: '{"job_name":"SGM_GBL_DATHSKP","host":"server1","command":"$script_path/jbs_sgmi_housekeeping_run.sh"}'
})
```

### 4. Control-M Condition Nodes (`type: "control-m-condition"`)

Represents Control-M conditions (events).

**Properties (stored in `properties_json`):**
- `condition_name` (String): Condition name
- `condition_type` (String): Type of condition
- Additional condition-specific properties

### 5. Control-M Calendar Nodes (`type: "control-m-calendar"`)

Represents Control-M calendars.

**Properties (stored in `properties_json`):**
- `calendar_name` (String): Calendar name
- Additional calendar-specific properties

### 6. Place Nodes (`type: "place"`)

Represents Petri net places (states/conditions).

**Properties (stored in `properties_json`):**
- `place_id` (String): Place identifier
- `place_type` (String): Type of place (e.g., `"initial"`, `"final"`)
- `initial_tokens` (Number): Initial token count
- Additional Petri net properties

### 7. Transition Nodes (`type: "transition"`)

Represents Petri net transitions (job executions).

**Properties (stored in `properties_json`):**
- `transition_id` (String): Transition identifier
- `subprocess_count` (Number): Number of subprocesses
- `sql_queries` (Array[String]): Associated SQL queries
- Additional transition properties

## Relationship Structure

All relationships in the graph share a common base structure:

### Base Relationship Type: `RELATIONSHIP`

**Required Properties:**
- `label` (String): Relationship type (see Relationship Types below)
- `properties_json` (String): JSON string containing all relationship-specific properties
- `updated_at` (String): ISO 8601 timestamp of last update

**Optional Properties:**
- `agent_id` (String): Agent identifier (if applicable)
- `domain` (String): Domain identifier (if applicable)

## Relationship Types

### 1. DATA_FLOW (`label: "DATA_FLOW"`)

Represents data flow between columns, capturing ETL transformation logic.

**Direction:** `(source_column)-[:RELATIONSHIP]->(target_column)`

**Properties (stored in `properties_json`):**
- `transformation_type` (String): Type of transformation
  - Values: `"aggregation"`, `"join"`, `"filter"`, `"cast"`, `"conditional"`, `"direct_copy"`, `"transformed"`
- `sql_expression` (String): SQL fragment creating this relationship
- `function` (String): Function applied (e.g., `"SUM"`, `"COUNT"`, `"CASE"`)
- `join_type` (String): JOIN type if applicable (e.g., `"INNER"`, `"LEFT"`, `"RIGHT"`)
- `join_condition` (String): JOIN condition SQL
- `filter_condition` (String): WHERE clause affecting this flow
- `sql_query_id` (String): Link to full SQL query (format: `"sql:<hash>"`)
- `step_order` (Number): Step order in multi-step pipeline (0-based)
- `intermediate_table` (String): Intermediate table in multi-step flow

**Example:**
```cypher
(:Node {id: "orders.amount"})-[:RELATIONSHIP {
  label: "DATA_FLOW",
  properties_json: '{"transformation_type":"aggregation","function":"SUM","sql_expression":"SUM(orders.amount)","sql_query_id":"sql:abc123","step_order":1,"intermediate_table":"staging"}'
}]->(:Node {id: "summary.total_amount"})
```

### 2. HAS_COLUMN (`label: "HAS_COLUMN"`)

Represents table-to-column containment.

**Direction:** `(table)-[:RELATIONSHIP]->(column)`

**Properties (stored in `properties_json`):**
- `source` (String): Source system identifier

**Example:**
```cypher
(:Node {id: "orders", type: "table"})-[:RELATIONSHIP {
  label: "HAS_COLUMN",
  properties_json: '{"source":"sgmi"}'
}]->(:Node {id: "orders.amount", type: "column"})
```

### 3. SCHEDULES (`label: "SCHEDULES"`)

Represents calendar-to-job scheduling relationship.

**Direction:** `(calendar)-[:RELATIONSHIP]->(job)`

**Properties:** Typically empty or minimal

**Example:**
```cypher
(:Node {id: "control-m-calendar:Daily", type: "control-m-calendar"})-[:RELATIONSHIP {
  label: "SCHEDULES"
}]->(:Node {id: "control-m:SGM_GBL_DATHSKP", type: "control-m-job"})
```

### 4. BLOCKS (`label: "BLOCKS"`)

Represents condition-to-job blocking relationship (input conditions).

**Direction:** `(condition)-[:RELATIONSHIP]->(job)`

**Properties (stored in `properties_json`):**
- Condition-specific properties from Control-M

**Example:**
```cypher
(:Node {id: "control-m-cond:START", type: "control-m-condition"})-[:RELATIONSHIP {
  label: "BLOCKS",
  properties_json: '{"condition_name":"START"}'
}]->(:Node {id: "control-m:Job1", type: "control-m-job"})
```

### 5. RELEASES (`label: "RELEASES"`)

Represents job-to-condition release relationship (output conditions).

**Direction:** `(job)-[:RELATIONSHIP]->(condition)`

**Properties (stored in `properties_json`):**
- Condition-specific properties from Control-M

**Example:**
```cypher
(:Node {id: "control-m:Job1", type: "control-m-job"})-[:RELATIONSHIP {
  label: "RELEASES",
  properties_json: '{"condition_name":"DONE"}'
}]->(:Node {id: "control-m-cond:DONE", type: "control-m-condition"})
```

### 6. HAS_PETRI_NET (`label: "HAS_PETRI_NET"`)

Represents root-to-Petri net relationship.

**Direction:** `(root)-[:RELATIONSHIP]->(petri_net_root)`

**Properties (stored in `properties_json`):**
- `petri_net_id` (String): Petri net identifier
- `name` (String): Petri net name

### 7. CONTAINS (`label: "CONTAINS"`)

Represents database-to-table/view containment.

**Direction:** `(database)-[:RELATIONSHIP]->(table|view)`

**Properties (stored in `properties_json`):**
- `source` (String): Source system identifier

### 8. REFERENCES (`label: "REFERENCES"`)

Represents foreign key relationships.

**Direction:** `(table)-[:RELATIONSHIP]->(referenced_table)`

**Properties (stored in `properties_json`):**
- `source` (String): Source system identifier
- `foreign_key` (String): Foreign key name
- `column` (String): Foreign key column
- `referenced_column` (String): Referenced column

## Schema Diagram

```
┌─────────────┐
│   Column    │
│  (source)   │
└──────┬──────┘
       │ DATA_FLOW
       │ [transformation_type, function, sql_expression, ...]
       ▼
┌─────────────┐
│   Column    │
│  (target)   │
└──────┬──────┘
       │ HAS_COLUMN
       ▼
┌─────────────┐
│    Table    │
└─────────────┘

┌─────────────┐      SCHEDULES      ┌─────────────┐
│  Calendar   │─────────────────────>│     Job     │
└─────────────┘                      └──────┬──────┘
                                             │ RELEASES
                                             ▼
                                      ┌─────────────┐
                                      │  Condition  │
                                      └─────────────┘
```

## Common Query Patterns

### Find Column Lineage with Transformations

```cypher
MATCH path = (source:Node {type: "column"})-[r:RELATIONSHIP*]->(target:Node {type: "column"})
WHERE r.label = "DATA_FLOW"
RETURN source, target, r
ORDER BY r.properties_json.step_order
```

### Find Aggregated Columns

```cypher
MATCH (n:Node {type: "column"})
WHERE n.properties_json CONTAINS '"transformation_type":"aggregation"'
RETURN n.id, n.label, n.properties_json
```

### Find Multi-Step Pipeline

```cypher
MATCH path = (start:Node {type: "column"})-[r:RELATIONSHIP*]->(end:Node {type: "column"})
WHERE ALL(rel in relationships(path) WHERE rel.label = "DATA_FLOW")
  AND ALL(rel in relationships(path) WHERE rel.properties_json.step_order IS NOT NULL)
RETURN path
ORDER BY relationships(path)[0].properties_json.step_order
```

### Find Columns with Specific Function

```cypher
MATCH (n:Node {type: "column"})
WHERE n.properties_json CONTAINS '"function":"SUM"'
RETURN n.id, n.label, n.properties_json.expression
```

### Find Join Relationships

```cypher
MATCH (source:Node {type: "column"})-[r:RELATIONSHIP]->(target:Node {type: "column"})
WHERE r.label = "DATA_FLOW"
  AND r.properties_json CONTAINS '"join_type"'
RETURN source, target, r.properties_json.join_type, r.properties_json.join_condition
```

## Indexes and Constraints

### Recommended Indexes

```cypher
// Index on node ID for fast lookups
CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id);

// Index on node type for filtering
CREATE INDEX node_type_index IF NOT EXISTS FOR (n:Node) ON (n.type);

// Index on relationship label for filtering
CREATE INDEX rel_label_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.label);
```

### Recommended Constraints

```cypher
// Ensure unique node IDs
CREATE CONSTRAINT node_id_unique IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;
```

## Notes

1. **Properties Storage**: All node and relationship properties are stored as JSON strings in `properties_json` to handle nested structures and Neo4j type limitations.

2. **Querying Properties**: When querying properties, use JSON functions:
   ```cypher
   WHERE n.properties_json CONTAINS '"transformation_type":"aggregation"'
   ```
   Or parse JSON:
   ```cypher
   WITH n, apoc.convert.fromJsonMap(n.properties_json) AS props
   WHERE props.transformation_type = "aggregation"
   ```

3. **Multi-Step Pipelines**: The `step_order` property on DATA_FLOW edges enables tracking of multi-step ETL transformations. Lower step_order values indicate earlier steps in the pipeline.

4. **Transformation Types**: The `transformation_type` property categorizes how data flows between columns, enabling queries to find specific types of transformations (aggregations, joins, filters, etc.).

5. **SQL Query Linking**: The `sql_query_id` property links graph nodes/edges to full SQL queries stored in vector storage, enabling retrieval of complete transformation logic.

