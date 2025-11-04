# Neo4j Browser Visualization Guide

## Quick Start

### 1. Access Neo4j Browser

- **URL**: http://54.196.0.75:7474
- **Username**: `neo4j`
- **Password**: `amodels123`

### 2. Basic Graph Visualization

Once logged in, you'll see a query editor. Try these queries:

#### See All Nodes (Small Sample)
```cypher
MATCH (n:Node)
RETURN n
LIMIT 25
```

#### See Relationships
```cypher
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
RETURN n, r, m
LIMIT 25
```

## Visualizing DATA_FLOW Relationships

### Simple DATA_FLOW Visualization

```cypher
MATCH path = (c1:Node {type: 'column'})-[r:RELATIONSHIP]->(c2:Node {type: 'column'})
WHERE r.label = 'DATA_FLOW'
RETURN path
LIMIT 50
```

**What you'll see**: 
- Column nodes connected by DATA_FLOW edges
- Click on nodes to see their properties
- Use mouse to drag and rearrange the graph

### Focus on Specific Table

```cypher
// Visualize DATA_FLOW for a specific table's columns
MATCH path = (t:Node {type: 'table', label: '`sgmi_all_f`'})
       -[:RELATIONSHIP {label: 'HAS_COLUMN'}]->(c1:Node {type: 'column'})
       -[r:RELATIONSHIP]->(c2:Node {type: 'column'})
WHERE r.label = 'DATA_FLOW'
RETURN path
LIMIT 100
```

### Table Structure with DATA_FLOW

```cypher
// See table structure and its DATA_FLOW relationships
MATCH (t:Node {type: 'table'})
OPTIONAL MATCH (t)-[r1:RELATIONSHIP]->(c:Node {type: 'column'})
OPTIONAL MATCH (c)-[r2:RELATIONSHIP]->(c2:Node {type: 'column'})
WHERE r1.label = 'HAS_COLUMN' AND r2.label = 'DATA_FLOW'
RETURN t, c, r2, c2
LIMIT 50
```

## Interactive Exploration Tips

### 1. Node Expansion
- Click on any node to select it
- Click the arrow (→) next to a node to expand its relationships
- This will show all connected nodes

### 2. Node Styling
- Neo4j Browser automatically colors nodes by label
- You can customize colors in the settings

### 3. Filtering
- Use WHERE clauses to filter specific patterns
- Example: `WHERE t.label CONTAINS 'sgmi'`

### 4. Limiting Results
- Always use `LIMIT` for large queries
- Start with small limits (10-50) and increase as needed

## Advanced Visualizations

### Multi-Hop Data Flow Paths

```cypher
// Find paths of length 2-4 showing data flow chains
MATCH path = (c1:Node {type: 'column'})-[*2..4]->(c2:Node {type: 'column'})
WHERE ALL(r in relationships(path) WHERE r.label = 'DATA_FLOW')
RETURN path
LIMIT 20
```

### Table-to-Table via DATA_FLOW

```cypher
// Visualize data flow from one table to another
MATCH path = (t1:Node {type: 'table'})
       -[:RELATIONSHIP {label: 'HAS_COLUMN'}]->(c1:Node {type: 'column'})
       -[:RELATIONSHIP {label: 'DATA_FLOW'}]->(c2:Node {type: 'column'})
       <-[:RELATIONSHIP {label: 'HAS_COLUMN'}]-(t2:Node {type: 'table'})
WHERE t1.label <> t2.label
RETURN path
LIMIT 30
```

### High-Connectivity Columns

```cypher
// Find and visualize columns with many DATA_FLOW connections
MATCH (c:Node {type: 'column'})-[r:RELATIONSHIP]->(c2:Node {type: 'column'})
WHERE r.label = 'DATA_FLOW'
WITH c, count(r) as degree
WHERE degree > 5
MATCH path = (c)-[:RELATIONSHIP {label: 'DATA_FLOW'}]->(c2:Node {type: 'column'})
RETURN path
LIMIT 50
```

## Text View vs Graph View

### Switch to Table View
- Click the "Table" button in results
- See data in tabular format
- Useful for exact values and counts

### Switch to Graph View
- Click the "Graph" button
- See visual graph representation
- Best for understanding relationships

## Exporting Results

1. **Export as CSV**: Click "Export CSV" button
2. **Copy Query**: Click on query to copy it
3. **Save as Image**: Right-click graph → Save image

## Example Exploration Workflow

1. **Start Broad**: 
   ```cypher
   MATCH (n:Node) RETURN n LIMIT 25
   ```

2. **Focus on Tables**:
   ```cypher
   MATCH (t:Node {type: 'table'}) RETURN t LIMIT 10
   ```

3. **Expand a Table**:
   ```cypher
   MATCH (t:Node {type: 'table', label: '`sgmi_all_f`'})
          -[r:RELATIONSHIP]->(c:Node {type: 'column'})
   RETURN t, r, c
   LIMIT 50
   ```

4. **Add DATA_FLOW**:
   ```cypher
   MATCH (t:Node {type: 'table', label: '`sgmi_all_f`'})
          -[:RELATIONSHIP {label: 'HAS_COLUMN'}]->(c1:Node {type: 'column'})
          -[r:RELATIONSHIP]->(c2:Node {type: 'column'})
   WHERE r.label = 'DATA_FLOW'
   RETURN t, c1, r, c2
   LIMIT 100
   ```

## Keyboard Shortcuts

- `Ctrl+Enter` or `Cmd+Enter`: Run query
- `Ctrl+Up/Down`: Navigate query history
- `Ctrl+/`: Comment/uncomment
- `Ctrl+L`: Clear editor

## Troubleshooting

### Query Too Slow
- Add `LIMIT` clause
- Use more specific WHERE filters
- Start with smaller patterns

### Too Many Nodes
- Use `LIMIT` to reduce results
- Filter by specific labels or properties
- Use `OPTIONAL MATCH` for sparse patterns

### Graph Too Cluttered
- Click and drag nodes to rearrange
- Use filters to show only relevant relationships
- Focus on specific subgraphs

## Useful Queries Collection

All queries are saved in:
- `docs/DATA_FLOW_EXPLORATION.md` - Detailed DATA_FLOW queries
- `docs/NEO4J_QUERIES.md` - General Neo4j queries

## Next Steps

1. **Explore the graph visually** using the queries above
2. **Export interesting patterns** for further analysis
3. **Build custom queries** for your specific use case
4. **Document findings** in your analysis

Happy exploring! 🚀

