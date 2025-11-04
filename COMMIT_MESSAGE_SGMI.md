feat: SGMI data loading with Neo4j/Postgres integration and training support

## Overview
Complete SGMI data loading pipeline with fixes for Neo4j persistence, Postgres replication, comprehensive documentation, and training data generation tools.

## Key Changes

### 1. Neo4j Persistence Fix
- Fixed nested property handling in `services/extract/neo4j.go`
- Neo4j doesn't support nested maps, so properties now stored as JSON string
- Changed from `n.properties = $props` (map) to `n.properties_json = $props` (JSON string)
- Allows all 31,979 SGMI nodes to be saved successfully

### 2. Postgres Integration
- Added postgres service to docker-compose.yml
- Configured extract service with POSTGRES_CATALOG_DSN
- Automatic schema replication to Postgres on graph submission
- Successfully loaded 31,979 nodes into glean_nodes table

### 3. Documentation
- docs/POSTGRES_QUERIES.md: Comprehensive query guide with examples
- docs/TRAINING_SETUP.md: Complete training data generation guide
- docs/SGMI_LOADING_GUIDE.md: Step-by-step loading instructions
- docs/NEXT_STEPS_AFTER_SGMI.md: Post-loading workflow guide
- docs/SGMI_STATUS.md: Current status and troubleshooting

### 4. Scripts
- scripts/load_sgmi_data.sh: Automated SGMI data loading
- scripts/generate_training_from_postgres.sh: Training data generation

## Data Status
- Postgres: ✅ 31,979 nodes (31,653 columns, 323 tables)
- Neo4j: ✅ Code fixed, ready for data (requires payload resubmission)

## Testing
- Verified Postgres data: 31,979 nodes loaded
- Fixed Neo4j build errors and property handling
- Created comprehensive query examples

## Next Steps
1. Resubmit payload to Neo4j for full data sync
2. Use Postgres queries from docs/POSTGRES_QUERIES.md
3. Generate training data: ./scripts/generate_training_from_postgres.sh
4. Start Relational Transformer training

## Files Changed
- services/extract/neo4j.go (Neo4j persistence fix)
- infrastructure/docker/brev/docker-compose.yml (Postgres service)

## Files Added
- docs/POSTGRES_QUERIES.md
- docs/TRAINING_SETUP.md
- docs/SGMI_LOADING_GUIDE.md
- docs/NEXT_STEPS_AFTER_SGMI.md
- docs/SGMI_STATUS.md
- scripts/load_sgmi_data.sh
- scripts/generate_training_from_postgres.sh
- CHANGELOG_SGMI_LOADING.md

