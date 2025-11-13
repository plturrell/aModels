-- Graph Service: Agent and Domain Tracking
-- This schema adds indexes for agent_id and domain tracking

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Index for agent tracking
CREATE INDEX node_agent_id IF NOT EXISTS FOR (n:Node) ON (n.agent_id);
CREATE INDEX rel_agent_id IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.agent_id);

-- Index for domain filtering
CREATE INDEX node_domain IF NOT EXISTS FOR (n:Node) ON (n.domain);
CREATE INDEX rel_domain IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.domain);

-- ============================================================================
-- DOWN MIGRATION
-- ============================================================================
-- Execute these statements in reverse order to rollback this migration
--
-- DROP INDEX rel_domain IF EXISTS;
-- DROP INDEX node_domain IF EXISTS;
-- DROP INDEX rel_agent_id IF EXISTS;
-- DROP INDEX node_agent_id IF EXISTS;
