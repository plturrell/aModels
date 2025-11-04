# Next Steps After Loading SGMI Data

## Current Status: ❌ SGMI Data NOT Loaded

- **Neo4j**: 0 nodes (empty)
- **Postgres**: `glean_nodes` table doesn't exist yet
- **Graph Service**: Not running (needs to be started)

## Immediate Action Required

### Quick Load (Automated)

```bash
./scripts/load_sgmi_data.sh
```

This script will:
1. ✅ Start required services (graph, postgres, neo4j)
2. ✅ Configure Postgres connection
3. ✅ Check SGMI data files exist
4. ✅ Run SGMI extraction
5. ✅ Verify data loaded

### Manual Load (Step-by-Step)

#### Step 1: Start Graph Service
```bash
cd infrastructure/docker/brev
docker compose up -d graph
```

#### Step 2: Restart Extract Service (to pick up Postgres config)
```bash
docker compose restart extract
```

#### Step 3: Run SGMI Extraction
```bash
cd services/extract/scripts
./run_sgmi_full_graph.sh http://graph-server:19080/graph
```

#### Step 4: Verify
```bash
# Check Neo4j
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);"

# Check Postgres
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_nodes;"
```

## After Loading: What's Next?

### 1. Explore the Data

**In Neo4j Browser** (http://54.196.0.75:7474):
```cypher
// Count nodes
MATCH (n:Node) RETURN count(n)

// See node types
MATCH (n:Node) RETURN n.type, count(n) ORDER BY count(n) DESC

// View sample graph
MATCH (n:Node)-[r:RELATIONSHIP]->(m:Node) RETURN n, r, m LIMIT 50
```

**In Postgres**:
```sql
SELECT kind, COUNT(*) FROM glean_nodes GROUP BY kind;
SELECT label, COUNT(*) FROM glean_edges GROUP BY label;
```

### 2. Generate Training Data

Once data is loaded, you can generate training examples:

```bash
# Use extract service training generation endpoint
curl -X POST http://54.196.0.75:8082/generate/training \
  -H "Content-Type: application/json" \
  -d '{"project_id": "sgmi-full"}'
```

### 3. Train Relational Transformer

Use the loaded graph data for training:

```bash
cd data/training
python3 scripts/train_relational_transformer.py \
  --config configs/rt_sgmi.yaml \
  --data-dir data/training/extracts
```

### 4. Query and Analyze

**Lineage Analysis**:
```cypher
// Find all upstream dependencies
MATCH path = (source:Node)-[*1..5]->(target:Node)
WHERE target.id = 'your_table_name'
RETURN path LIMIT 30
```

**View Dependencies**:
```cypher
// Find views that depend on tables
MATCH (table:Node)-[*1..3]->(view:Node)
WHERE toLower(table.type) CONTAINS 'table'
  AND toLower(view.type) CONTAINS 'view'
RETURN DISTINCT table.id, view.id
```

### 5. Use with AgentFlow

The loaded data can be used by AgentFlow workflows:
- Access via `/sgmi/view-lineage` endpoint
- Use in LangFlow workflows
- Query via Postgres queries

## Data Flow After Loading

```
SGMI Data (Loaded)
    ↓
Neo4j (Graph queries, lineage)
    ↓
Postgres (SQL queries, analytics)
    ↓
Training Data Generation
    ↓
Relational Transformer Training
    ↓
Model Inference
```

## Troubleshooting

### "Graph service not responding"
- Check if graph service is running: `docker ps | grep graph`
- Check logs: `docker logs graph-server`
- Verify port 19080 is accessible

### "No data in Neo4j"
- Check extract service logs: `docker logs extract-service`
- Verify Neo4j connection: `docker exec extract-service env | grep NEO4J`
- Run extraction again: `./run_sgmi_full_graph.sh`

### "Postgres table doesn't exist"
- Check if Postgres DSN is configured: `docker exec extract-service env | grep POSTGRES`
- Restart extract service: `docker compose restart extract`
- Check Postgres connection: `docker exec postgres psql -U postgres -d amodels -c "\dt"`

## Summary

**Current State**: SGMI data is NOT loaded - databases are empty

**Action Needed**: Run `./scripts/load_sgmi_data.sh` to load the data

**After Loading**: Data will be in both Neo4j and Postgres, ready for:
- Graph exploration
- Training data generation
- Relational Transformer training
- Lineage analysis

