# Quick Start - Accessing Services

## ✅ Neo4j Browser is Now Accessible!

AWS Security Groups have been configured. You can now access Neo4j directly:

### Neo4j Browser
- **URL**: http://54.196.0.75:7474
- **Bolt URL**: bolt://ec2-54-196-0-75.compute-1.amazonaws.com:7687
- **Username**: `neo4j`
- **Password**: `amodels123`

### Other Accessible Services

- **LocalAI**: http://54.196.0.75:8081/v1/models
- **Extract Service**: http://54.196.0.75:8082/healthz
- **AgentFlow**: http://54.196.0.75:8001/healthz
- **Browser Automation**: http://54.196.0.75:8070/healthz
- **Search Inference**: http://54.196.0.75:8090

## Quick Test

Test all services:
```bash
./scripts/test_all_services.sh
```

## Visualizing the Graph

Once in Neo4j Browser, try these queries:

```cypher
// View all nodes and relationships (limited)
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node)
RETURN n, r, m
LIMIT 50

// Count nodes by type
MATCH (n:Node)
RETURN n.type AS nodeType, count(n) AS count
ORDER BY count DESC

// View table nodes
MATCH (n:Node)
WHERE n.type = 'table' OR n.type = 'Table'
RETURN n.id AS tableName, n.label AS label
LIMIT 50
```

## Documentation

- **Service URLs**: `docs/SERVICE_URLS.md` - Complete list of all service URLs
- **Graph Visualization**: `docs/GRAPH_VISUALIZATION.md` - Neo4j Browser guide
- **External Access**: `docs/EXTERNAL_ACCESS.md` - Access options and troubleshooting

## Next Steps

1. Explore the graph data in Neo4j Browser
2. Test other services using the URLs above
3. Run SGMI training pipeline if needed
4. Use Python visualization script: `scripts/visualize_graph.py`

