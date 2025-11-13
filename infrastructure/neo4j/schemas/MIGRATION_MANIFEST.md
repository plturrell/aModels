# Neo4j Schema Migration Manifest

This document lists all Neo4j schema files in the order they should be executed, along with their dependencies.

## Execution Order

Schemas must be executed in the following order to satisfy dependencies:

### Phase 1: Base Schemas (Required First)

These schemas create the fundamental structure that all other schemas depend on.

1. **`base/001_base_node_constraints.cypher`**
   - Creates base Node constraints (unique ID, not null)
   - **Dependencies:** None
   - **Required by:** All other schemas

2. **`base/002_base_relationship_indexes.cypher`**
   - Creates base Node and RELATIONSHIP indexes
   - **Dependencies:** `base/001_base_node_constraints.cypher`
   - **Required by:** All other schemas

3. **`base/003_composite_indexes.cypher`**
   - Creates composite indexes for common query patterns
   - **Dependencies:** `base/002_base_relationship_indexes.cypher`
   - **Required by:** None (but improves performance for all queries)

### Phase 2: Domain Schemas (Can be executed in parallel within each domain)

Domain schemas can be executed in any order within their domain, but should be executed in numerical order.

#### Catalog Domain

4. **`domain/catalog/001_neo4j_schema.cypher`**
   - Creates Resource, DataElementConcept, Representation, DataElement, ValueDomain constraints
   - **Dependencies:** `base/001_base_node_constraints.cypher`
   - **Required by:** `domain/catalog/002_triplestore_schema.cypher`

5. **`domain/catalog/002_triplestore_schema.cypher`**
   - Creates Triple and Graph constraints for RDF triplestore
   - **Dependencies:** `domain/catalog/001_neo4j_schema.cypher`

6. **`domain/catalog/003_execution_tracking.cypher`**
   - Creates Execution, ExecutionMetrics, ProcessEvent constraints
   - **Dependencies:** `base/001_base_node_constraints.cypher`

7. **`domain/catalog/004_data_quality.cypher`**
   - Creates QualityIssue and QualityMetric constraints
   - **Dependencies:** `base/001_base_node_constraints.cypher`

8. **`domain/catalog/005_performance_metrics.cypher`**
   - Creates PerformanceMetric and QueryPerformance constraints
   - **Dependencies:** `base/001_base_node_constraints.cypher`, `domain/catalog/003_execution_tracking.cypher`

#### Regulatory Domain

9. **`domain/regulatory/001_bcbs239_schema.cypher`**
   - Creates BCBS239 compliance schema (principles, controls, calculations, data assets, processes)
   - **Dependencies:** `base/001_base_node_constraints.cypher`
   - **Note:** Some composite indexes are included here for domain completeness

#### Graph Domain

10. **`domain/graph/001_fulltext_indexes.cypher`**
    - Creates full-text search indexes
    - **Dependencies:** `base/002_base_relationship_indexes.cypher`, `domain/regulatory/001_bcbs239_schema.cypher` (for BCBS239 fulltext index)

11. **`domain/graph/002_agent_domain_tracking.cypher`**
    - Creates indexes for agent_id and domain tracking
    - **Dependencies:** `base/002_base_relationship_indexes.cypher`

### Phase 3: Optimizations (Optional, can be executed after all schemas)

12. **`optimizations/indexes_and_constraints.cypher`**
    - Additional optimization queries and utilities
    - **Dependencies:** All base and domain schemas
    - **Note:** Contains utility queries (warm-up, statistics, analysis) that don't create schema objects

## Relationship Definitions

Relationship type definitions are documented in the `relationships/` directory but do not create schema objects. They are for reference only:

- `relationships/data_lineage.cypher` - DATA_FLOW, HAS_COLUMN, CONTAINS, REFERENCES
- `relationships/workflow.cypher` - SCHEDULES, BLOCKS, RELEASES, HAS_PETRI_NET
- `relationships/compliance.cypher` - BCBS239 compliance relationships

## Quick Reference

### Minimum Required Execution Order

```
1. base/001_base_node_constraints.cypher
2. base/002_base_relationship_indexes.cypher
3. base/003_composite_indexes.cypher
4. domain/catalog/001_neo4j_schema.cypher
5. domain/catalog/002_triplestore_schema.cypher
6. domain/catalog/003_execution_tracking.cypher
7. domain/catalog/004_data_quality.cypher
8. domain/catalog/005_performance_metrics.cypher
9. domain/regulatory/001_bcbs239_schema.cypher
10. domain/graph/001_fulltext_indexes.cypher
11. domain/graph/002_agent_domain_tracking.cypher
12. optimizations/indexes_and_constraints.cypher (optional)
```

### Execution Script

```bash
#!/bin/bash
# Execute all schemas in order

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

SCHEMAS_DIR="infrastructure/neo4j/schemas"

# Phase 1: Base schemas
for file in $SCHEMAS_DIR/base/*.cypher; do
  echo "Executing: $file"
  cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" < "$file"
done

# Phase 2: Domain schemas
for file in $SCHEMAS_DIR/domain/catalog/*.cypher; do
  echo "Executing: $file"
  cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" < "$file"
done

for file in $SCHEMAS_DIR/domain/regulatory/*.cypher; do
  echo "Executing: $file"
  cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" < "$file"
done

for file in $SCHEMAS_DIR/domain/graph/*.cypher; do
  echo "Executing: $file"
  cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" < "$file"
done

# Phase 3: Optimizations (optional)
for file in $SCHEMAS_DIR/optimizations/*.cypher; do
  echo "Executing: $file"
  cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASSWORD" -a "$NEO4J_URI" < "$file"
done
```

## Notes

- All schema files use `IF NOT EXISTS` to allow idempotent execution
- Schema files can be executed multiple times safely
- Some indexes may take time to build on large datasets
- Check index status with `SHOW INDEXES` after execution
- Relationship definitions in `relationships/` are documentation only and don't need execution

## Service Integration

- **Graph Service**: Uses migrations from `services/graph/pkg/migrations/registry.go` (references centralized schemas)
- **Regulatory Service**: Uses `services/regulatory/bcbs239_graph_schema.go` (references centralized schemas)
- **Catalog Service**: Uses goose migrations from `services/catalog/migrations/` (references centralized schemas)

All services maintain backward compatibility but reference the centralized schema files.

