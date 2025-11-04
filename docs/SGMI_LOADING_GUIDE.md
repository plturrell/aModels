# SGMI Data Loading Guide

## Current Status

❌ **SGMI data is NOT yet loaded:**
- Neo4j: 0 nodes
- Postgres: `glean_nodes` table doesn't exist
- Graph service: Not running

## What Needs to Happen

The SGMI data needs to be:
1. **Extracted** from source files (JSON, HQL, XML)
2. **Submitted** to the graph service (port 19080)
3. **Saved to Neo4j** (via graph service)
4. **Saved to Postgres** (via extract service, if configured)

## Step 1: Start the Graph Service

The graph service is required to receive and process the SGMI data:

```bash
cd /home/aModels/infrastructure/docker/brev
docker compose up -d graph
```

Wait for it to start:
```bash
docker logs graph-server --tail 20
```

## Step 2: Configure Postgres Connection (Optional but Recommended)

To save SGMI data to Postgres, configure the extract service:

**Update docker-compose.yml** to add `POSTGRES_CATALOG_DSN`:

```yaml
extract:
  environment:
    - POSTGRES_CATALOG_DSN=postgresql://postgres:postgres@postgres:5432/amodels?sslmode=disable
```

Then restart extract service:
```bash
docker compose restart extract
```

## Step 3: Run SGMI Extraction

Run the SGMI extraction script:

```bash
cd /home/aModels/services/extract/scripts
./run_sgmi_full_graph.sh http://graph-server:19080/graph
```

Or if running from outside the container:
```bash
./run_sgmi_full_graph.sh http://54.196.0.75:19080/graph
```

## Step 4: Verify Data Loaded

### Check Neo4j
```cypher
// Count nodes
MATCH (n:Node)
RETURN count(n) AS nodeCount

// Check node types
MATCH (n:Node)
RETURN n.type AS nodeType, count(n) AS count
ORDER BY count DESC
```

### Check Postgres (if configured)
```bash
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_nodes;"
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_edges;"
```

## Expected Output

After successful extraction, you should see:
- **Neo4j**: Hundreds or thousands of nodes (tables, views, columns, Control-M jobs)
- **Postgres**: Same nodes in `glean_nodes` table, edges in `glean_edges` table
- **Files**: `sgmi_view_lineage.json` and `sgmi_view_summary.json` in the store directory

## Troubleshooting

### Graph Service Not Starting
```bash
docker logs graph-server
# Check for errors, missing dependencies, etc.
```

### Extract Service Can't Connect to Neo4j
```bash
docker exec extract-service env | grep NEO4J
# Should show: NEO4J_URI=bolt://neo4j:7687
```

### Postgres Connection Fails
```bash
docker exec extract-service env | grep POSTGRES
# Should show: POSTGRES_CATALOG_DSN=postgresql://...
```

### No Data After Extraction
1. Check extract service logs: `docker logs extract-service`
2. Check graph service logs: `docker logs graph-server`
3. Verify SGMI data files exist in `data/training/sgmi/`

## Next Steps After Loading

Once SGMI data is loaded:

1. **Explore in Neo4j Browser** - Use queries from `docs/NEO4J_QUERIES.md`
2. **Query Postgres** - Use SQL to analyze the data
3. **Run Training** - Use the loaded data for Relational Transformer training
4. **Generate Training Data** - Extract training examples from the graph

