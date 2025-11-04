# DATA_FLOW Relationships Exploration Guide

## Overview

The SGMI dataset contains **5,549 DATA_FLOW relationships** that represent data lineage and transformation flows between columns. These relationships show how data moves through the system.

## Quick Access to Neo4j Browser

1. **Open Neo4j Browser**: http://54.196.0.75:7474
2. **Login**:
   - Username: `neo4j`
   - Password: `amodels123`
3. **Connection**: You should see "Connected as user neo4j"

## Visual Exploration Queries

### 1. View Sample DATA_FLOW Relationships

Run this query to see a visual graph of DATA_FLOW relationships:

```cypher
// Visualize DATA_FLOW relationships between columns
MATCH (c1:Node)-[r:RELATIONSHIP]->(c2:Node)
WHERE r.label = 'DATA_FLOW'
RETURN c1, r, c2
LIMIT 50
```

**Alternative (if above doesn't work):**
```cypher
MATCH (c1:Node)-[r:RELATIONSHIP]->(c2:Node)
WHERE r.label = 'DATA_FLOW'
RETURN c1.label as source, c2.label as target
LIMIT 50
```

**What to look for**: 
- Columns (nodes) connected by DATA_FLOW edges
- Self-referential flows (column → same column)
- Cross-column flows within the same table

### 2. Explore Table-to-Table Data Flows

```cypher
// Find data flows between different tables
MATCH (t1:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)<-[r3:RELATIONSHIP]-(t2:Node)
WHERE t1.type = 'table'
  AND t2.type = 'table'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
  AND r3.label = 'HAS_COLUMN'
  AND t1.label <> t2.label
RETURN t1.label as source_table, c1.label as source_column, c2.label as target_column, t2.label as target_table
LIMIT 30
```

**Visual version:**
```cypher
MATCH (t1:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)<-[r3:RELATIONSHIP]-(t2:Node)
WHERE t1.type = 'table'
  AND t2.type = 'table'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
  AND r3.label = 'HAS_COLUMN'
RETURN t1, r1, c1, r2, c2, r3, t2
LIMIT 30
```

**What to look for**: 
- Paths showing data flowing from one table to another
- Columns that are sources/targets of transformations

### 3. Find High-Connectivity Columns

```cypher
// Columns with most DATA_FLOW connections
MATCH (c:Node)-[r:RELATIONSHIP]->()
WHERE c.type = 'column'
  AND r.label = 'DATA_FLOW'
WITH c, count(r) as out_degree
MATCH ()-[r2:RELATIONSHIP]->(c)
WHERE c.type = 'column'
  AND r2.label = 'DATA_FLOW'
RETURN c.label as column, 
       out_degree, 
       count(r2) as in_degree, 
       (out_degree + count(r2)) as total_degree
ORDER BY total_degree DESC
LIMIT 20
```

### 4. Explore Specific Table Data Flows

```cypher
// Replace 'sgmi_all_f' with any table name
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND t.label = '`sgmi_all_f`'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t.label as table, c1.label as source_column, c2.label as target_column
LIMIT 50
```

**Visual version:**
```cypher
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND t.label = '`sgmi_all_f`'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t, r1, c1, r2, c2
LIMIT 50
```

### 5. View Data Flow Statistics by Table

```cypher
// Count DATA_FLOW relationships per table
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t.label as table_name, count(r2) as flow_count
ORDER BY flow_count DESC
LIMIT 20
```

## Text-Based Exploration Queries

### Find Cross-Table Data Flows

```cypher
MATCH (t1:Node {type: 'table'})-[r1:RELATIONSHIP]->(c1:Node {type: 'column'})
      -[r2:RELATIONSHIP]->(c2:Node {type: 'column'})
      <-[r3:RELATIONSHIP]-(t2:Node {type: 'table'})
WHERE r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
  AND r3.label = 'HAS_COLUMN'
  AND t1.label <> t2.label
  AND c1.label <> c2.label
RETURN t1.label as source_table,
       c1.label as source_column,
       c2.label as target_column,
       t2.label as target_table
LIMIT 50
```

### Find Non-Self-Referential Flows

```cypher
MATCH (c1:Node {type: 'column'})-[r:RELATIONSHIP]->(c2:Node {type: 'column'})
WHERE r.label = 'DATA_FLOW'
  AND c1.label <> c2.label
RETURN c1.label as source_column,
       c2.label as target_column,
       count(*) as count
ORDER BY count DESC
LIMIT 30
```

### Analyze DATA_FLOW Patterns

```cypher
// Find columns that are both sources and targets
MATCH (c:Node {type: 'column'})
WHERE (c)-[:RELATIONSHIP {label: 'DATA_FLOW'}]->()
  AND ()-[:RELATIONSHIP {label: 'DATA_FLOW'}]->(c)
RETURN c.label as column,
       size((c)-[:RELATIONSHIP {label: 'DATA_FLOW'}]->()) as outflows,
       size(()-[:RELATIONSHIP {label: 'DATA_FLOW'}]->(c)) as inflows
ORDER BY (outflows + inflows) DESC
LIMIT 20
```

## Understanding the Data Flow Structure

### Types of DATA_FLOW Relationships

1. **Self-Referential Flows** (Column → Same Column)
   - Indicates data transformation/pipeline within the same column
   - Common in ETL processes
   - Example: `sgmi_crm_contract_ref_f`.`crm_contract_ref_no` → `sgmi_crm_contract_ref_f`.`crm_contract_ref_no`

2. **Intra-Table Flows** (Column → Different Column, Same Table)
   - Data transformations within a single table
   - Computed columns, derived values

3. **Inter-Table Flows** (Column → Column, Different Tables)
   - Data lineage between tables
   - ETL pipeline transformations
   - Most valuable for understanding data dependencies

### Key Insights

- **5,549 total DATA_FLOW relationships**
- Most are self-referential (same column name)
- Some represent actual data transformations
- Can be used to understand:
  - ETL pipeline structure
  - Data lineage
  - Transformation dependencies
  - Process understanding

## Exporting DATA_FLOW Data

### Export to CSV via Postgres

```sql
-- Export cross-table DATA_FLOW relationships
COPY (
  SELECT 
    t1.label as source_table,
    c1.label as source_column,
    c2.label as target_column,
    t2.label as target_table
  FROM glean_nodes t1
  JOIN glean_edges e1 ON e1.source_id = t1.id AND e1.label = 'HAS_COLUMN'
  JOIN glean_nodes c1 ON e1.target_id = c1.id
  JOIN glean_edges e2 ON e2.source_id = c1.id AND e2.label = 'DATA_FLOW'
  JOIN glean_nodes c2 ON e2.target_id = c2.id
  JOIN glean_edges e3 ON e3.target_id = c2.id AND e3.label = 'HAS_COLUMN'
  JOIN glean_nodes t2 ON e3.source_id = t2.id
  WHERE t1.kind = 'table' 
    AND t2.kind = 'table'
    AND t1.id <> t2.id
    AND c1.id <> c2.id
) TO '/tmp/data_flows.csv' WITH CSV HEADER;
```

## Next Steps

1. **Visualize in Neo4j Browser**: Use the queries above
2. **Export for Analysis**: Use the Postgres export query
3. **Build Lineage Maps**: Create visualizations of data flow paths
4. **Train Process Models**: Use DATA_FLOW for process understanding training

## Useful Neo4j Browser Tips

1. **Node Styling**: Click on a node to see its properties
2. **Expand**: Click the arrow next to a node to expand relationships
3. **Filter**: Use WHERE clauses to filter specific patterns
4. **Limit**: Always use LIMIT for large queries
5. **Save Queries**: Bookmark useful queries in Neo4j Browser
6. **Export Results**: Click "Export CSV" to download query results

## Example: Full Data Flow Path

```cypher
// Find a complete data flow path from source to target
MATCH path = (t1:Node {type: 'table'})
       -[:RELATIONSHIP {label: 'HAS_COLUMN'}]->(c1:Node {type: 'column'})
       -[:RELATIONSHIP {label: 'DATA_FLOW'}]->(c2:Node {type: 'column'})
       <-[:RELATIONSHIP {label: 'HAS_COLUMN'}]-(t2:Node {type: 'table'})
WHERE t1.label = '`sgmi_all_f`'
  AND t2.label <> t1.label
RETURN path
LIMIT 10
```

This will show you visual paths of how data flows from one table to another through column transformations.

