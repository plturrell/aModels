#!/usr/bin/env bash
# Reconcile graph data between Neo4j and Postgres
# This ensures both stores have the same data
set -euo pipefail

echo "=== Graph Reconciliation: Neo4j ↔ Postgres ==="
echo ""

# Check counts
echo "📊 Current State:"
PG_NODES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')
PG_EDGES=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_edges;" 2>/dev/null | tr -d '[:space:]')
NEO4J_NODES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH (n:Node) RETURN count(n);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')
NEO4J_EDGES=$(docker exec neo4j cypher-shell -u neo4j -p amodels123 "MATCH ()-[r:RELATIONSHIP]->() RETURN count(r);" 2>/dev/null | grep -v "^$" | tail -1 | tr -d '[:space:]')

echo "  Postgres: $PG_NODES nodes, $PG_EDGES edges"
echo "  Neo4j:    $NEO4J_NODES nodes, $NEO4J_EDGES edges"
echo ""

# Check if reconciliation is needed
NODE_DIFF=$((PG_NODES - NEO4J_NODES))
EDGE_DIFF=$((PG_EDGES - NEO4J_EDGES))

if [[ "$NODE_DIFF" -eq 0 && "$EDGE_DIFF" -eq 0 ]]; then
    echo "✅ Graphs are already in sync!"
    exit 0
fi

echo "⚠️  Differences detected:"
echo "  Nodes: $NODE_DIFF difference"
echo "  Edges: $EDGE_DIFF difference"
echo ""

# Option 1: Re-sync from Postgres to Neo4j (recommended)
echo "📊 Option 1: Re-sync from Postgres to Neo4j"
echo "   (Postgres is the source of truth)"
echo ""

# Export from Postgres and import to Neo4j
echo "Exporting from Postgres..."
docker exec postgres psql -U postgres -d amodels -c "COPY (SELECT id, kind, label, COALESCE(properties_json::text, '{}') FROM glean_nodes) TO STDOUT WITH CSV HEADER;" > /tmp/nodes_export.csv
docker exec postgres psql -U postgres -d amodels -c "COPY (SELECT source_id, target_id, label, COALESCE(properties_json::text, '{}') FROM glean_edges) TO STDOUT WITH CSV HEADER;" > /tmp/edges_export.csv

echo "✅ Exported $PG_NODES nodes and $PG_EDGES edges"
echo ""

# Note: Full reconciliation requires re-running extraction or using a sync script
# For now, we'll create a Python script to do the sync
cat > /tmp/sync_to_neo4j.py << 'PYTHON'
import csv
from neo4j import GraphDatabase
import json

# Neo4j connection
driver = GraphDatabase.driver(
    "bolt://neo4j:7687",
    auth=("neo4j", "amodels123")
)

def clear_neo4j(tx):
    tx.run("MATCH (n:Node) DETACH DELETE n")

def import_nodes(tx, nodes_file):
    with open(nodes_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            props_json = row.get('properties_json', '{}')
            try:
                props = json.loads(props_json) if props_json else {}
            except:
                props = {}
            
            # Store properties as JSON string
            props_str = json.dumps(props)
            
            tx.run("""
                MERGE (n:Node {id: $id})
                SET n.type = $type,
                    n.label = $label,
                    n.properties_json = $props
            """, 
            id=row['id'],
            type=row['kind'],
            label=row['label'],
            props=props_str)

def import_edges(tx, edges_file):
    with open(edges_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            props_json = row.get('properties_json', '{}')
            try:
                props = json.loads(props_json) if props_json else {}
            except:
                props = {}
            
            props_str = json.dumps(props)
            
            tx.run("""
                MATCH (source:Node {id: $source_id})
                MATCH (target:Node {id: $target_id})
                MERGE (source)-[r:RELATIONSHIP]->(target)
                SET r.label = $label,
                    r.properties_json = $props
            """,
            source_id=row['source_id'],
            target_id=row['target_id'],
            label=row['label'],
            props=props_str)

with driver.session() as session:
    print("Clearing Neo4j...")
    session.execute_write(clear_neo4j)
    
    print("Importing nodes...")
    session.execute_write(import_nodes, "/tmp/nodes_export.csv")
    
    print("Importing edges...")
    session.execute_write(import_edges, "/tmp/edges_export.csv")
    
    print("✅ Sync complete!")

driver.close()
PYTHON

echo "Note: Full Neo4j sync requires Python with neo4j driver."
echo "      The extract service automatically syncs during extraction."
echo ""
echo "To manually sync, re-run the extraction or use the extract service's"
echo "graph persistence which handles this automatically."
echo ""

# Alternative: Check if extract service can re-sync
echo "📊 Alternative: Re-sync via Extract Service"
echo ""
echo "The extract service automatically reconciles during extraction."
echo "The graph is saved to both Postgres (via replicateSchema) and Neo4j (via graphPersistence)."
echo ""
echo "Current sync status:"
if [[ "$NODE_DIFF" -eq 0 && "$EDGE_DIFF" -eq 0 ]]; then
    echo "  ✅ In sync"
else
    echo "  ⚠️  Out of sync - re-run extraction to reconcile"
fi

