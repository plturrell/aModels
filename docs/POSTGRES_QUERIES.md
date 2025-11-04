# Postgres Query Guide for SGMI Data

The SGMI data is stored in Postgres in the `glean_nodes` and `glean_edges` tables.

## Connection

```bash
docker exec -it postgres psql -U postgres -d amodels
```

Or from outside:
```bash
psql -h 54.196.0.75 -p 5432 -U postgres -d amodels
# Password: postgres
```

## Basic Queries

### Count Nodes and Edges

```sql
-- Total nodes
SELECT COUNT(*) FROM glean_nodes;

-- Total edges
SELECT COUNT(*) FROM glean_edges;

-- Nodes by type
SELECT kind, COUNT(*) as count 
FROM glean_nodes 
GROUP BY kind 
ORDER BY count DESC;
```

### Sample Data

```sql
-- View sample nodes
SELECT id, kind, label, properties_json 
FROM glean_nodes 
LIMIT 10;

-- View sample edges
SELECT source_id, target_id, label, properties_json 
FROM glean_edges 
LIMIT 10;
```

## Table and Column Queries

### Find All Tables

```sql
SELECT id, label, properties_json 
FROM glean_nodes 
WHERE kind = 'table'
ORDER BY label;
```

### Find Columns for a Specific Table

```sql
-- Get columns for a table
SELECT n.id, n.label, n.properties_json
FROM glean_nodes n
JOIN glean_edges e ON e.target_id = n.id
WHERE e.source_id = 'sgmisit.sgmi_all_f' 
  AND n.kind = 'column'
ORDER BY n.label;
```

### Find Table Dependencies

```sql
-- Find what tables a view depends on
SELECT DISTINCT n.id, n.label, n.kind
FROM glean_nodes n
JOIN glean_edges e ON e.target_id = n.id
WHERE e.source_id = 'vw_sgmi_all_f' 
  AND n.kind = 'table';
```

## View Lineage Queries

### Find Views and Their Dependencies

```sql
-- List all views
SELECT id, label 
FROM glean_nodes 
WHERE kind = 'view'
ORDER BY label;

-- Find all dependencies of a view (tables and columns)
WITH RECURSIVE view_deps AS (
  -- Base: start with view
  SELECT e.target_id as dep_id, e.label as rel_type, 1 as depth
  FROM glean_edges e
  WHERE e.source_id = 'vw_sgmi_all_f'
  
  UNION ALL
  
  -- Recursive: follow edges to dependencies
  SELECT e.target_id, e.label, vd.depth + 1
  FROM glean_edges e
  JOIN view_deps vd ON e.source_id = vd.dep_id
  WHERE vd.depth < 5  -- Limit depth to avoid infinite loops
)
SELECT DISTINCT n.id, n.label, n.kind, vd.depth
FROM view_deps vd
JOIN glean_nodes n ON n.id = vd.dep_id
ORDER BY vd.depth, n.kind, n.label;
```

### Find Column Lineage

```sql
-- Find column dependencies
SELECT 
  e1.source_id as source_column,
  e1.target_id as intermediate,
  e2.target_id as target_column,
  e1.label as rel1,
  e2.label as rel2
FROM glean_edges e1
JOIN glean_edges e2 ON e1.target_id = e2.source_id
WHERE e1.source_id LIKE '%.column%'
  AND e2.target_id LIKE '%.column%'
LIMIT 20;
```

## Control-M Job Queries

### Find Control-M Jobs

```sql
SELECT id, label, properties_json
FROM glean_nodes
WHERE kind = 'control-m-job'
ORDER BY label;
```

### Find Job Dependencies

```sql
-- Find Control-M job dependencies
SELECT 
  n1.id as job_id,
  n1.label as job_name,
  e.label as dependency_type,
  n2.id as depends_on_id,
  n2.label as depends_on_name
FROM glean_nodes n1
JOIN glean_edges e ON e.source_id = n1.id
JOIN glean_nodes n2 ON e.target_id = n2.id
WHERE n1.kind = 'control-m-job'
ORDER BY n1.label, e.label;
```

## Schema Analysis

### Find Tables with Most Columns

```sql
SELECT 
  t.id as table_id,
  t.label as table_name,
  COUNT(c.id) as column_count
FROM glean_nodes t
LEFT JOIN glean_edges e ON e.source_id = t.id
LEFT JOIN glean_nodes c ON e.target_id = c.id AND c.kind = 'column'
WHERE t.kind = 'table'
GROUP BY t.id, t.label
ORDER BY column_count DESC
LIMIT 20;
```

### Find Most Connected Nodes

```sql
-- Nodes with most incoming edges
SELECT 
  n.id,
  n.label,
  n.kind,
  COUNT(e.id) as incoming_edges
FROM glean_nodes n
LEFT JOIN glean_edges e ON e.target_id = n.id
GROUP BY n.id, n.label, n.kind
ORDER BY incoming_edges DESC
LIMIT 20;
```

### Find Most Connected Nodes (Outgoing)

```sql
-- Nodes with most outgoing edges
SELECT 
  n.id,
  n.label,
  n.kind,
  COUNT(e.id) as outgoing_edges
FROM glean_nodes n
LEFT JOIN glean_edges e ON e.source_id = n.id
GROUP BY n.id, n.label, n.kind
ORDER BY outgoing_edges DESC
LIMIT 20;
```

## JSON Property Queries

### Extract Properties from JSON

```sql
-- Extract specific property from JSON
SELECT 
  id,
  label,
  properties_json->>'type' as data_type,
  properties_json->>'nullable' as nullable
FROM glean_nodes
WHERE kind = 'column'
LIMIT 10;

-- Find columns with specific data types
SELECT id, label, properties_json->>'type' as data_type
FROM glean_nodes
WHERE kind = 'column'
  AND properties_json->>'type' = 'string'
LIMIT 20;
```

## Advanced Queries

### Find Paths Between Nodes

```sql
-- Find shortest path between two nodes (up to 3 hops)
WITH RECURSIVE path_finder AS (
  SELECT 
    source_id as start_node,
    target_id as end_node,
    label as edge_label,
    ARRAY[source_id, target_id] as path,
    1 as depth
  FROM glean_edges
  WHERE source_id = 'sgmisit.sgmi_all_f'
  
  UNION ALL
  
  SELECT 
    pf.start_node,
    e.target_id,
    e.label,
    pf.path || e.target_id,
    pf.depth + 1
  FROM path_finder pf
  JOIN glean_edges e ON e.source_id = pf.end_node
  WHERE pf.depth < 3
    AND NOT e.target_id = ANY(pf.path)  -- Avoid cycles
)
SELECT DISTINCT 
  start_node,
  end_node,
  depth,
  path
FROM path_finder
WHERE end_node LIKE '%vw_sgmi%'
ORDER BY depth, start_node;
```

### Find Circular Dependencies

```sql
-- Find potential circular dependencies (same node appears in path)
WITH RECURSIVE cycle_finder AS (
  SELECT 
    source_id,
    target_id,
    ARRAY[source_id, target_id] as path,
    1 as depth
  FROM glean_edges
  
  UNION ALL
  
  SELECT 
    cf.source_id,
    e.target_id,
    cf.path || e.target_id,
    cf.depth + 1
  FROM cycle_finder cf
  JOIN glean_edges e ON e.source_id = cf.target_id
  WHERE cf.depth < 5
    AND NOT e.target_id = ANY(cf.path[1:array_length(cf.path, 1)-1])  -- Allow last node to repeat
)
SELECT DISTINCT source_id, target_id, path, depth
FROM cycle_finder
WHERE source_id = target_id  -- Path returns to start
ORDER BY depth;
```

## Statistics

### Overall Statistics

```sql
-- Overall statistics
SELECT 
  'Nodes' as type,
  COUNT(*) as count,
  COUNT(DISTINCT kind) as unique_types
FROM glean_nodes
UNION ALL
SELECT 
  'Edges' as type,
  COUNT(*) as count,
  COUNT(DISTINCT label) as unique_types
FROM glean_edges;
```

### Data Type Distribution

```sql
-- Column data type distribution
SELECT 
  properties_json->>'type' as data_type,
  COUNT(*) as count
FROM glean_nodes
WHERE kind = 'column'
  AND properties_json->>'type' IS NOT NULL
GROUP BY data_type
ORDER BY count DESC;
```

## Export Queries

### Export to CSV

```sql
-- Export nodes to CSV
\copy (SELECT id, kind, label, properties_json FROM glean_nodes) TO '/tmp/nodes.csv' CSV HEADER;

-- Export edges to CSV
\copy (SELECT source_id, target_id, label, properties_json FROM glean_edges) TO '/tmp/edges.csv' CSV HEADER;
```

### Export for Training

```sql
-- Export table-column relationships for training
SELECT 
  t.id as table_id,
  t.label as table_name,
  c.id as column_id,
  c.label as column_name,
  c.properties_json->>'type' as column_type
FROM glean_nodes t
JOIN glean_edges e ON e.source_id = t.id
JOIN glean_nodes c ON e.target_id = c.id
WHERE t.kind = 'table' 
  AND c.kind = 'column'
ORDER BY t.id, c.label;
```

