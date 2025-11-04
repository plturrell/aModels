# Training Setup Guide

This guide explains how to use the loaded SGMI data for Relational Transformer training.

## Data Status

✅ **Postgres**: 31,979 nodes loaded (tables, columns, views, Control-M jobs)
✅ **Neo4j**: 31,979 nodes loaded (after fix)
✅ **Training Data**: Can be generated from Postgres

## Step 1: Verify Data is Loaded

### Check Postgres

```bash
docker exec postgres psql -U postgres -d amodels -c "SELECT COUNT(*) FROM glean_nodes;"
docker exec postgres psql -U postgres -d amodels -c "SELECT kind, COUNT(*) FROM glean_nodes GROUP BY kind;"
```

### Check Neo4j

```bash
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);"
docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN n.type, count(n) ORDER BY count(n) DESC LIMIT 10;"
```

## Step 2: Generate Training Data

### Automatic Generation

```bash
./scripts/generate_training_from_postgres.sh
```

This script will:
1. Generate table extracts (schema information)
2. Generate document extracts (text descriptions)
3. Save to `data/training/extracts/`

### Manual Generation

#### Via Extract Service API

```bash
# Generate table extracts
curl -X POST http://54.196.0.75:8082/generate/training \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "table",
    "table_options": {
      "project_id": "sgmi-full"
    }
  }'

# Generate document extracts
curl -X POST http://54.196.0.75:8082/generate/training \
  -H "Content-Type: application/json" \
  -d '{
    "mode": "document",
    "document_options": {
      "project_id": "sgmi-full"
    }
  }'
```

#### Direct from Postgres

```bash
# Export table-column relationships
docker exec postgres psql -U postgres -d amodels -c "
COPY (
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
  ORDER BY t.id, c.label
) TO '/tmp/table_columns.csv' CSV HEADER;
"

# Copy from container to host
docker cp postgres:/tmp/table_columns.csv ./data/training/extracts/
```

## Step 3: Prepare Training Configuration

### Create Training Config

Create `configs/rt_sgmi.yaml`:

```yaml
model:
  name: relational_transformer
  hidden_size: 512
  num_layers: 6
  num_heads: 8

data:
  source: postgres
  connection: "postgresql://postgres:postgres@postgres:5432/amodels?sslmode=disable"
  tables:
    - glean_nodes
    - glean_edges
  
training:
  batch_size: 32
  learning_rate: 0.0001
  num_epochs: 10
  checkpoint_dir: ./models/checkpoints/sgmi
```

## Step 4: Run Training

### Using Trainer Container

```bash
# Enter training container
docker exec -it training-shell bash

# Inside container
cd /workspace
python3 scripts/train_relational_transformer.py \
  --config configs/rt_sgmi.yaml \
  --data-dir data/training/extracts
```

### Using Direct Python

```bash
cd data/training
python3 scripts/train_relational_transformer.py \
  --config ../../configs/rt_sgmi.yaml \
  --data-dir extracts
```

## Step 5: Query Training Data

### From Postgres

See `docs/POSTGRES_QUERIES.md` for comprehensive query examples.

#### Quick Examples

```sql
-- Get all table-column pairs
SELECT 
  t.label as table_name,
  c.label as column_name,
  c.properties_json->>'type' as data_type
FROM glean_nodes t
JOIN glean_edges e ON e.source_id = t.id
JOIN glean_nodes c ON e.target_id = c.id
WHERE t.kind = 'table' AND c.kind = 'column'
LIMIT 100;

-- Get view dependencies
SELECT 
  v.label as view_name,
  t.label as table_name
FROM glean_nodes v
JOIN glean_edges e ON e.source_id = v.id
JOIN glean_nodes t ON e.target_id = t.id
WHERE v.kind = 'view' AND t.kind = 'table';
```

### From Neo4j

```cypher
// Get table-column relationships
MATCH (table:Node {type: 'table'})-[r:RELATIONSHIP]->(column:Node {type: 'column'})
RETURN table.label, column.label, r.label
LIMIT 100;

// Get view dependencies
MATCH (view:Node {type: 'view'})-[*1..3]->(table:Node {type: 'table'})
RETURN DISTINCT view.label, table.label;
```

## Training Data Format

### Expected Structure

```
data/training/extracts/
├── tables/
│   └── YYYYMMDD-HHMMSS/
│       ├── manifest.json
│       ├── table_1.json
│       ├── table_2.json
│       └── ...
└── documents/
    └── YYYYMMDD-HHMMSS/
        ├── manifest.json
        ├── doc_1.json
        └── ...
```

### Table Extract Format

```json
{
  "table_id": "sgmisit.sgmi_all_f",
  "table_name": "sgmi_all_f",
  "columns": [
    {
      "column_id": "sgmisit.sgmi_all_f.contract_ref_no",
      "column_name": "contract_ref_no",
      "data_type": "string",
      "nullable": true
    }
  ],
  "relationships": [
    {
      "target_table": "sgmisit.sgmi_all_txn_f",
      "relationship_type": "references"
    }
  ]
}
```

## Next Steps

1. **Review Generated Data**: Check `data/training/extracts/` for generated files
2. **Configure Training**: Update training configs for your specific needs
3. **Run Training**: Start Relational Transformer training
4. **Evaluate**: Use evaluation scripts to test model performance
5. **Inference**: Use trained model for predictions on new data

## Troubleshooting

### No Data in Postgres/Neo4j

```bash
# Re-run SGMI extraction
cd services/extract/scripts
./run_sgmi_full_graph.sh http://54.196.0.75:8082/graph
```

### Training Data Generation Fails

```bash
# Check extract service logs
docker logs extract-service --tail 50

# Check if extract service is running
docker ps | grep extract-service

# Test extract service
curl http://54.196.0.75:8082/healthz
```

### Training Script Errors

```bash
# Check Python dependencies
docker exec training-shell pip list | grep -E "torch|transformers|pandas"

# Verify data directory
docker exec training-shell ls -la /workspace/data/training/extracts/
```

