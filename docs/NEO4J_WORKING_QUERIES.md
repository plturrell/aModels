# Working Neo4j Queries for SGMI Data

These queries have been tested and verified to work with the actual data structure.

## Basic Queries

### 1. Count All Nodes
```cypher
MATCH (n:Node)
RETURN count(n)
```

### 2. Count Nodes by Type
```cypher
MATCH (n:Node)
RETURN n.type, count(n) as count
ORDER BY count DESC
```

### 3. See Sample Nodes
```cypher
MATCH (n:Node)
RETURN n
LIMIT 25
```

### 4. Count Relationships
```cypher
MATCH ()-[r:RELATIONSHIP]->()
RETURN count(r)
```

### 5. Count Relationships by Type
```cypher
MATCH ()-[r:RELATIONSHIP]->()
RETURN r.label, count(r) as count
ORDER BY count DESC
```

## Table Queries

### 6. List All Tables
```cypher
MATCH (n:Node)
WHERE n.type = 'table'
RETURN n.label
LIMIT 50
```

### 7. Get Table with Its Columns
```cypher
MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node)
WHERE t.type = 'table'
  AND r.label = 'HAS_COLUMN'
RETURN t.label as table, c.label as column
LIMIT 50
```

### 8. Count Columns per Table
```cypher
MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node)
WHERE t.type = 'table'
  AND r.label = 'HAS_COLUMN'
RETURN t.label as table, count(c) as column_count
ORDER BY column_count DESC
LIMIT 20
```

### 9. Get Specific Table's Columns
```cypher
MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node)
WHERE t.type = 'table'
  AND t.label = '`sgmi_all_f`'
  AND r.label = 'HAS_COLUMN'
RETURN t.label as table, c.label as column
LIMIT 100
```

## DATA_FLOW Queries

### 10. See DATA_FLOW Relationships
```cypher
MATCH (c1:Node)-[r:RELATIONSHIP]->(c2:Node)
WHERE r.label = 'DATA_FLOW'
RETURN c1.label as source, c2.label as target
LIMIT 50
```

### 11. Visualize DATA_FLOW (Graph View)
```cypher
MATCH (c1:Node)-[r:RELATIONSHIP]->(c2:Node)
WHERE r.label = 'DATA_FLOW'
RETURN c1, r, c2
LIMIT 50
```

### 12. Count DATA_FLOW per Column
```cypher
MATCH (c:Node)-[r:RELATIONSHIP]->()
WHERE r.label = 'DATA_FLOW'
RETURN c.label as column, count(r) as flow_count
ORDER BY flow_count DESC
LIMIT 20
```

### 13. DATA_FLOW for Specific Table
```cypher
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND t.label = '`sgmi_all_f`'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t.label as table, c1.label as source_column, c2.label as target_column
LIMIT 50
```

### 14. Visualize Table with DATA_FLOW
```cypher
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND t.label = '`sgmi_all_f`'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t, r1, c1, r2, c2
LIMIT 50
```

## Advanced Queries

### 15. Find Columns with Most Connections
```cypher
MATCH (c:Node)-[r:RELATIONSHIP]->()
WHERE c.type = 'column'
WITH c, count(r) as out_degree
MATCH ()-[r2:RELATIONSHIP]->(c)
WHERE c.type = 'column'
RETURN c.label as column, out_degree, count(r2) as in_degree, (out_degree + count(r2)) as total
ORDER BY total DESC
LIMIT 20
```

### 16. Explore Node Properties
```cypher
MATCH (n:Node)
WHERE n.type = 'table'
RETURN n.label, n.properties_json
LIMIT 5
```

### 17. Find Relationships Between Specific Tables
```cypher
MATCH (t1:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)<-[r3:RELATIONSHIP]-(t2:Node)
WHERE t1.type = 'table'
  AND t2.type = 'table'
  AND r1.label = 'HAS_COLUMN'
  AND r3.label = 'HAS_COLUMN'
RETURN t1.label as source_table, t2.label as target_table, count(*) as connections
LIMIT 20
```

### 18. Path Finding (Simple)
```cypher
MATCH path = (t1:Node)-[*2..4]-(t2:Node)
WHERE t1.type = 'table'
  AND t2.type = 'table'
  AND t1.label <> t2.label
RETURN path
LIMIT 10
```

## Visualization Queries (Best for Graph View)

### 19. Simple Graph View
```cypher
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
RETURN n, r, m
LIMIT 100
```

### 20. Table-Column Graph
```cypher
MATCH (t:Node)-[r:RELATIONSHIP]->(c:Node)
WHERE t.type = 'table'
  AND r.label = 'HAS_COLUMN'
RETURN t, r, c
LIMIT 100
```

### 21. DATA_FLOW Graph
```cypher
MATCH (c1:Node)-[r:RELATIONSHIP]->(c2:Node)
WHERE r.label = 'DATA_FLOW'
RETURN c1, r, c2
LIMIT 100
```

### 22. Combined View (Table → Column → DATA_FLOW)
```cypher
MATCH (t:Node)-[r1:RELATIONSHIP]->(c1:Node)-[r2:RELATIONSHIP]->(c2:Node)
WHERE t.type = 'table'
  AND r1.label = 'HAS_COLUMN'
  AND r2.label = 'DATA_FLOW'
RETURN t, r1, c1, r2, c2
LIMIT 50
```

## Troubleshooting

If a query returns no results:

1. **Check node properties**: Use query #16 to see actual property structure
2. **Use WHERE instead of property matching**: `WHERE n.type = 'table'` works better than `{type: 'table'}`
3. **Check label format**: Labels may have backticks like `` `sgmi_all_f` ``
4. **Start simple**: Use queries #1-5 first to verify data exists
5. **Add LIMIT**: Always use LIMIT to avoid overwhelming results

## Quick Reference

- **All nodes**: `MATCH (n:Node) RETURN n LIMIT 25`
- **All relationships**: `MATCH ()-[r:RELATIONSHIP]->() RETURN r LIMIT 25`
- **Count nodes**: `MATCH (n:Node) RETURN count(n)`
- **Count relationships**: `MATCH ()-[r:RELATIONSHIP]->() RETURN count(r)`

