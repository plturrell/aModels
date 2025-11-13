-- Composite Indexes for Performance
-- This schema creates composite indexes for common query patterns
-- Must be applied after base indexes

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Composite index for table lookups (type + label)
CREATE INDEX node_table_lookup IF NOT EXISTS FOR (n:Node) ON (n.type, n.label);

-- Composite index for temporal queries with type filtering
CREATE INDEX node_type_temporal IF NOT EXISTS FOR (n:Node) ON (n.type, n.updated_at);

-- Composite index for relationship label and temporal queries
CREATE INDEX relationship_label_temporal IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.label, r.updated_at);

-- ============================================================================
-- DOWN MIGRATION
-- ============================================================================
-- Execute these statements in reverse order to rollback this migration
--
-- DROP INDEX relationship_label_temporal IF EXISTS;
-- DROP INDEX node_type_temporal IF EXISTS;
-- DROP INDEX node_table_lookup IF EXISTS;
