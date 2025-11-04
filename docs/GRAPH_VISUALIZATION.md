# Graph Visualization Guide

This guide explains how to visualize the processed graph data stored in Neo4j.

## Neo4j Browser (Web UI)

The easiest way to visualize the graph is using Neo4j Browser, which is accessible at:

**URL:** `http://localhost:7474`

**Credentials:**
- Username: `neo4j`
- Password: `amodels123`

### Accessing Neo4j Browser

1. Open your web browser and navigate to `http://localhost:7474`
2. Log in with the credentials above
3. Start exploring with the Cypher queries below

## Example Cypher Queries

### 1. View All Nodes and Relationships (Limited)
```cypher
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
RETURN n, r, m
LIMIT 100
```

### 2. Count Nodes by Type
```cypher
MATCH (n:Node)
RETURN n.type AS nodeType, count(n) AS count
ORDER BY count DESC
```

### 3. View Table Nodes
```cypher
MATCH (n:Node)
WHERE n.type = 'table' OR n.type = 'Table'
RETURN n.id AS tableName, n.label AS label, n.properties AS properties
LIMIT 50
```

### 4. View View Nodes
```cypher
MATCH (n:Node)
WHERE n.type = 'view' OR n.type = 'View'
RETURN n.id AS viewName, n.label AS label, n.properties AS properties
LIMIT 50
```

### 5. View Control-M Job Nodes
```cypher
MATCH (n:Node)
WHERE toLower(n.type) CONTAINS 'control-m' OR toLower(n.type) CONTAINS 'controlm'
RETURN n.id AS jobName, n.label AS label, n.properties AS properties
LIMIT 50
```

### 6. View Relationships by Type
```cypher
MATCH ()-[r:RELATIONSHIP]->()
RETURN r.label AS relationshipType, count(r) AS count
ORDER BY count DESC
```

### 7. View Lineage for a Specific Table
```cypher
MATCH path = (source:Node)-[*1..3]->(target:Node)
WHERE source.id = 'your_table_name'
RETURN path
LIMIT 50
```

### 8. View Upstream Dependencies
```cypher
MATCH path = (source:Node)-[*1..5]->(target:Node)
WHERE target.id = 'your_table_name'
RETURN path
LIMIT 50
```

### 9. View Downstream Dependencies
```cypher
MATCH path = (target:Node)-[*1..5]->(source:Node)
WHERE target.id = 'your_table_name'
RETURN path
LIMIT 50
```

### 10. View Graph Statistics
```cypher
MATCH (n:Node)
WITH count(n) AS nodeCount
MATCH ()-[r:RELATIONSHIP]->()
RETURN nodeCount, count(r) AS relationshipCount
```

### 11. View All Node Types
```cypher
MATCH (n:Node)
RETURN DISTINCT n.type AS nodeType
ORDER BY nodeType
```

### 12. View Relationships with Properties
```cypher
MATCH (source:Node)-[r:RELATIONSHIP]->(target:Node)
WHERE r.properties IS NOT NULL
RETURN source.id AS source, r.label AS relationship, target.id AS target, r.properties AS properties
LIMIT 100
```

## Python Visualization Script

You can also visualize the graph programmatically using Python. Here's an example script:

```python
#!/usr/bin/env python3
"""
Visualize Neo4j graph using networkx and matplotlib
"""
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import os

# Neo4j connection
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "amodels123")

def visualize_graph(limit=100):
    """Visualize a subset of the graph."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    with driver.session() as session:
        # Fetch nodes and relationships
        result = session.run("""
            MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
            RETURN n.id AS source, r.label AS rel, m.id AS target, 
                   n.type AS sourceType, m.type AS targetType
            LIMIT $limit
        """, limit=limit)
        
        # Build networkx graph
        G = nx.DiGraph()
        for record in result:
            source = record["source"]
            target = record["target"]
            rel = record["rel"]
            source_type = record["sourceType"]
            target_type = record["targetType"]
            
            G.add_node(source, type=source_type)
            G.add_node(target, type=target_type)
            G.add_edge(source, target, label=rel)
    
    driver.close()
    
    # Visualize
    plt.figure(figsize=(20, 20))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Color nodes by type
    node_colors = []
    for node in G.nodes():
        node_type = G.nodes[node].get("type", "unknown")
        if "table" in str(node_type).lower():
            node_colors.append("lightblue")
        elif "view" in str(node_type).lower():
            node_colors.append("lightgreen")
        elif "control" in str(node_type).lower():
            node_colors.append("lightcoral")
        else:
            node_colors.append("lightgray")
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=500, font_size=8, arrows=True, edge_color="gray")
    plt.title("Graph Visualization")
    plt.tight_layout()
    plt.savefig("graph_visualization.png", dpi=300, bbox_inches="tight")
    print("Visualization saved to graph_visualization.png")
    plt.show()

if __name__ == "__main__":
    visualize_graph(limit=100)
```

## Graph Service API

The graph service also provides REST endpoints for querying graph data:

- **Graph Query Endpoint:** `http://localhost:8080/graph`
- **Graph Stats Endpoint:** `http://localhost:8080/graph/stats`

You can query the graph via HTTP:

```bash
# Get graph statistics
curl http://localhost:8080/graph/stats

# Query specific nodes
curl -X POST http://localhost:8080/graph/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (n:Node) WHERE n.type = \"table\" RETURN n LIMIT 10"}'
```

## Tips

1. **Start Small:** Use `LIMIT` clauses to avoid overwhelming the browser with too many nodes
2. **Filter by Type:** Use `WHERE` clauses to focus on specific node types (tables, views, jobs)
3. **Path Queries:** Use variable-length paths `[*1..5]` to explore multi-hop relationships
4. **Performance:** For large graphs, use `PROFILE` or `EXPLAIN` to understand query performance
5. **Export:** Neo4j Browser allows you to export query results as CSV or JSON

## Troubleshooting

If Neo4j Browser is not accessible:
1. Check if Neo4j is running: `docker ps | grep neo4j`
2. Check Neo4j logs: `docker logs neo4j`
3. Verify port mapping: `docker port neo4j`
4. Check firewall settings if accessing remotely

