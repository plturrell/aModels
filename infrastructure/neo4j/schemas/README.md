# Neo4j Schema Organization

This directory contains all Neo4j schema definitions, organized by purpose and domain.

## Directory Structure

- **`base/`** - Core/base schemas that must be applied first
  - Base Node and RELATIONSHIP constraints
  - Base indexes
  - Composite indexes for common query patterns

- **`domain/`** - Domain-specific schemas organized by service/domain
  - **`catalog/`** - Catalog service schemas (RDF, triplestore, execution tracking, data quality, performance metrics)
  - **`regulatory/`** - Regulatory compliance schemas (BCBS239)
  - **`graph/`** - Graph service specific schemas (fulltext indexes, agent/domain tracking)

- **`relationships/`** - Relationship type definitions and documentation
  - Data lineage relationships (DATA_FLOW, HAS_COLUMN, etc.)
  - Workflow relationships (SCHEDULES, BLOCKS, RELEASES)
  - Compliance relationships (BCBS239 relationships)

- **`optimizations/`** - Performance optimization queries
  - Additional indexes and constraints
  - Query optimization utilities
  - Database statistics and analysis queries

## Schema Execution Order

Schemas should be executed in the following order:

1. **Base schemas** (from `base/` directory, in numerical order)
2. **Domain schemas** (from `domain/` subdirectories, in numerical order within each domain)
3. **Optimizations** (from `optimizations/` directory)

See `MIGRATION_MANIFEST.md` for the complete execution order and dependencies.

## Usage

### Applying All Schemas

```bash
# Apply base schemas
for file in base/*.cypher; do
  cypher-shell -u neo4j -p <password> < "$file"
done

# Apply domain schemas
for file in domain/*/*.cypher; do
  cypher-shell -u neo4j -p <password> < "$file"
done

# Apply optimizations (optional)
for file in optimizations/*.cypher; do
  cypher-shell -u neo4j -p <password> < "$file"
done
```

### Using with Go Services

Services can load and execute these schema files programmatically. See service-specific migration code for examples.

## Relationship Types

See `relationships/README.md` for complete documentation of all relationship types used in the knowledge graph.

## Notes

- All schema files use `IF NOT EXISTS` to allow idempotent execution
- Schema files include both `Up` and `Down` migrations where applicable
- Some schemas may have dependencies on others - check `MIGRATION_MANIFEST.md`

