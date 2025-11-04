# Neo4j Query Guide - Start Here!

## 🚀 Quick Start Queries

### 1. Check What Data You Have
```cypher
// Count total nodes and relationships
MATCH (n:Node)
WITH count(n) AS nodeCount
MATCH ()-[r:RELATIONSHIP]->()
RETURN nodeCount, count(r) AS relationshipCount
```

### 2. See Node Types in Your Graph
```cypher
// List all node types and their counts
MATCH (n:Node)
RETURN n.type AS nodeType, count(n) AS count
ORDER BY count DESC
```

### 3. View Sample Nodes
```cypher
// View first 25 nodes with their properties
MATCH (n:Node)
RETURN n.id AS id, n.type AS type, n.label AS label
LIMIT 25
```

### 4. View Sample Relationships
```cypher
// View first 25 relationships
MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
RETURN source.id AS source, r.label AS relationship, target.id AS target
LIMIT 25
```

## 📊 Explore Your Graph

### 5. Visualize a Small Subgraph
```cypher
// View a connected subgraph (limited to 50 nodes for performance)
MATCH path = (n:Node)-[*1..2]->(m:Node)
RETURN path
LIMIT 50
```

### 6. Find Most Connected Nodes
```cypher
// Find nodes with the most relationships (hubs)
MATCH (n:Node)-[r:RELATIONSHIP]-()
RETURN n.id AS node, n.type AS type, count(r) AS connections
ORDER BY connections DESC
LIMIT 20
```

### 7. View Relationship Types
```cypher
// See what types of relationships exist
MATCH ()-[r:RELATIONSHIP]->()
RETURN r.label AS relationshipType, count(r) AS count
ORDER BY count DESC
```

## 🗄️ Explore Tables and Views (SGMI Data)

### 8. View All Tables
```cypher
// Find all table nodes
MATCH (n:Node)
WHERE toLower(n.type) CONTAINS 'table'
RETURN n.id AS tableName, n.label AS label, n.properties AS properties
LIMIT 50
```

### 9. View All Views
```cypher
// Find all view nodes
MATCH (n:Node)
WHERE toLower(n.type) CONTAINS 'view'
RETURN n.id AS viewName, n.label AS label
LIMIT 50
```

### 10. Find Tables with Most Dependencies
```cypher
// Tables that are referenced by many other nodes
MATCH (n:Node)-[r:RELATIONSHIP]->(target:Node)
WHERE toLower(n.type) CONTAINS 'table'
WITH target, count(r) AS incomingCount
RETURN target.id AS tableName, incomingCount AS dependencies
ORDER BY dependencies DESC
LIMIT 20
```

### 11. View Table Lineage (Upstream)
```cypher
// Find what feeds into a specific table (replace 'your_table_name' with actual table)
MATCH path = (source:Node)-[*1..5]->(target:Node)
WHERE target.id = 'your_table_name' OR toLower(target.id) CONTAINS 'your_table_name'
RETURN path
LIMIT 30
```

### 12. View Table Lineage (Downstream)
```cypher
// Find what a specific table feeds into
MATCH path = (source:Node)-[*1..5]->(target:Node)
WHERE source.id = 'your_table_name' OR toLower(source.id) CONTAINS 'your_table_name'
RETURN path
LIMIT 30
```

## 🔍 Search and Filter

### 13. Search for Specific Table/View
```cypher
// Search for nodes containing specific text
MATCH (n:Node)
WHERE toLower(n.id) CONTAINS 'sgmi' OR toLower(n.label) CONTAINS 'sgmi'
RETURN n.id AS name, n.type AS type, n.label AS label
LIMIT 50
```

### 14. Find Control-M Jobs
```cypher
// Find all Control-M job nodes
MATCH (n:Node)
WHERE toLower(n.type) CONTAINS 'control' OR toLower(n.type) CONTAINS 'job'
RETURN n.id AS jobName, n.label AS label, n.properties AS properties
LIMIT 50
```

### 15. View Column Lineage
```cypher
// Find nodes related to columns (if column lineage exists)
MATCH (n:Node)
WHERE toLower(n.type) CONTAINS 'column'
RETURN n.id AS columnName, n.type AS type, n.label AS label
LIMIT 50
```

## 🔗 Relationship Exploration

### 16. View Relationships Between Specific Types
```cypher
// View relationships between tables and views
MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
WHERE toLower(source.type) CONTAINS 'table' AND toLower(target.type) CONTAINS 'view'
RETURN source.id AS sourceTable, r.label AS relationship, target.id AS targetView
LIMIT 50
```

### 17. Find Paths Between Two Nodes
```cypher
// Find paths between two specific nodes (replace with actual names)
MATCH path = shortestPath((start:Node {id: 'source_node_name'})-[*..10]-(end:Node {id: 'target_node_name'}))
RETURN path
```

### 18. View All Connections for a Specific Node
```cypher
// View all connections for a specific node (replace with actual node ID)
MATCH (n:Node {id: 'your_node_id'})-[r:RELATIONSHIP]-(connected:Node)
RETURN n.id AS node, r.label AS relationship, connected.id AS connectedTo, connected.type AS connectedType
LIMIT 50
```

## 📈 Statistics and Analysis

### 19. Graph Statistics Summary
```cypher
// Comprehensive graph statistics
MATCH (n:Node)
WITH count(n) AS totalNodes
MATCH ()-[r:RELATIONSHIP]->()
WITH totalNodes, count(r) AS totalRelationships
MATCH (n:Node)
WITH totalNodes, totalRelationships, 
     collect(DISTINCT n.type) AS nodeTypes,
     count(DISTINCT n.type) AS uniqueNodeTypes
MATCH ()-[r:RELATIONSHIP]->()
RETURN totalNodes, 
       totalRelationships,
       uniqueNodeTypes,
       count(DISTINCT r.label) AS uniqueRelationshipTypes,
       nodeTypes
```

### 20. Find Orphaned Nodes (No Connections)
```cypher
// Find nodes with no relationships
MATCH (n:Node)
WHERE NOT (n)-[:RELATIONSHIP]-()
RETURN n.id AS orphanNode, n.type AS type, n.label AS label
LIMIT 50
```

### 21. Find Most Complex Relationships
```cypher
// Find nodes with the most complex relationship patterns
MATCH (n:Node)-[r:RELATIONSHIP]-(connected:Node)
WITH n, count(DISTINCT connected) AS uniqueConnections, count(r) AS totalRelationships
WHERE uniqueConnections > 5
RETURN n.id AS node, n.type AS type, uniqueConnections, totalRelationships
ORDER BY uniqueConnections DESC
LIMIT 20
```

## 🎯 SGMI-Specific Queries

### 22. Find SGMI Tables
```cypher
// Find all SGMI-related tables
MATCH (n:Node)
WHERE toLower(n.id) CONTAINS 'sgmi' OR toLower(n.label) CONTAINS 'sgmi'
RETURN n.id AS name, n.type AS type
ORDER BY n.type, n.id
LIMIT 100
```

### 23. View SGMI View Dependencies
```cypher
// Find views that depend on SGMI tables
MATCH (table:Node)-[*1..3]->(view:Node)
WHERE (toLower(table.id) CONTAINS 'sgmi' OR toLower(table.label) CONTAINS 'sgmi')
  AND toLower(view.type) CONTAINS 'view'
RETURN DISTINCT table.id AS sourceTable, view.id AS targetView
LIMIT 50
```

### 24. Find Control-M Job Dependencies
```cypher
// Find Control-M jobs and what they depend on
MATCH (job:Node)-[r:RELATIONSHIP]->(dependsOn:Node)
WHERE toLower(job.type) CONTAINS 'control' OR toLower(job.type) CONTAINS 'job'
RETURN job.id AS jobName, r.label AS relationship, dependsOn.id AS dependsOn, dependsOn.type AS dependsOnType
LIMIT 50
```

## 💡 Tips

1. **Start Small**: Use `LIMIT` clauses to avoid overwhelming the browser
2. **Use Visualizations**: The graph visualization in Neo4j Browser helps understand relationships
3. **Filter by Type**: Use `WHERE` clauses to focus on specific node types
4. **Path Queries**: Use `[*1..5]` for variable-length paths to explore multi-hop relationships
5. **Performance**: For large graphs, use `PROFILE` to understand query performance

## 🚨 If Queries Are Slow

Add `LIMIT` to reduce results:
```cypher
// Always add LIMIT for initial exploration
MATCH (n:Node)
RETURN n
LIMIT 100  // Start with 100, increase if needed
```

Use `PROFILE` to see query execution:
```cypher
PROFILE
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
RETURN n, r, m
LIMIT 50
```

## 📝 Next Steps

1. Run query #1 to see what data you have
2. Run query #2 to see node types
3. Run query #8 or #9 to explore tables/views
4. Use query #5 to visualize a small subgraph
5. Use query #11 or #12 to explore lineage for specific tables

Happy exploring! 🎉

