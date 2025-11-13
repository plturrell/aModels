-- Graph Service: Full-text Indexes
-- This schema adds full-text search indexes for labels and descriptions

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Full-text index on node labels for search
CREATE FULLTEXT INDEX node_label_fulltext IF NOT EXISTS
FOR (n:Node) ON EACH [n.label, n.id];

-- Full-text index on BCBS239 principles
CREATE FULLTEXT INDEX bcbs239_principle_fulltext IF NOT EXISTS
FOR (p:BCBS239Principle) ON EACH [p.principle_name, p.description];

-- ============================================================================
-- DOWN MIGRATION
-- ============================================================================
-- Execute these statements in reverse order to rollback this migration
--
-- DROP INDEX bcbs239_principle_fulltext IF EXISTS;
-- DROP INDEX node_label_fulltext IF EXISTS;
