-- Neo4j Performance Optimization Queries
-- Execute these to improve query performance across common access patterns
-- Note: Many of these may already be created by base and domain schemas
-- This file contains additional optimizations and utility queries

-- ============================================================================
-- ADDITIONAL INDEXES (if not already created by domain schemas)
-- ============================================================================

-- Index for Control-M job lookups (if not in base schemas)
CREATE INDEX controlm_job_lookup IF NOT EXISTS 
FOR (n:Node) ON (n.type, n.properties_json);

-- ============================================================================
-- QUERY OPTIMIZATION - Warm up indexes
-- ============================================================================
-- Run these queries to warm up indexes after creation

-- Warm up node type index
MATCH (n:Node) WHERE n.type IS NOT NULL RETURN count(n);

-- Warm up relationship label index
MATCH ()-[r:RELATIONSHIP]->() WHERE r.label IS NOT NULL RETURN count(r);

-- Warm up BCBS239 indexes
MATCH (p:BCBS239Principle) RETURN count(p);

-- ============================================================================
-- DATABASE STATISTICS
-- ============================================================================
-- Use these queries to inspect database state

-- View all indexes
SHOW INDEXES;

-- View all constraints
SHOW CONSTRAINTS;

-- Database statistics
CALL db.stats.retrieve('GRAPH COUNTS');

-- ============================================================================
-- PERFORMANCE ANALYSIS QUERIES
-- ============================================================================
-- Use these queries to analyze database performance and identify issues

-- Find nodes without proper type
MATCH (n:Node) WHERE n.type IS NULL RETURN count(n) as untyped_nodes;

-- Find relationships without proper label
MATCH ()-[r:RELATIONSHIP]->() WHERE r.label IS NULL RETURN count(r) as unlabeled_relationships;

-- Count nodes by type (for capacity planning)
MATCH (n:Node) 
RETURN n.type as type, count(n) as count 
ORDER BY count DESC;

-- Count relationships by label (for capacity planning)
MATCH ()-[r:RELATIONSHIP]->() 
RETURN r.label as label, count(r) as count 
ORDER BY count DESC;

-- Identify high-degree nodes (potential performance bottlenecks)
MATCH (n:Node)
WITH n, size((n)--()) as degree
WHERE degree > 100
RETURN n.id, n.type, n.label, degree
ORDER BY degree DESC
LIMIT 50;

-- ============================================================================
-- CLEANUP QUERIES (run periodically with caution)
-- ============================================================================

-- Find duplicate nodes (same ID)
MATCH (n:Node)
WITH n.id as node_id, collect(n) as nodes
WHERE size(nodes) > 1
RETURN node_id, size(nodes) as duplicates;

-- Remove orphaned nodes (nodes with no relationships)
-- CAUTION: Only run if this is expected behavior
-- Uncomment and use with care:
-- MATCH (n:Node) WHERE NOT (n)--() DELETE n;

-- ============================================================================
-- NOTES
-- ============================================================================
-- 1. Run this script after initial data load or schema changes
-- 2. Indexes are created asynchronously - check SHOW INDEXES for status
-- 3. Full-text indexes require Neo4j 4.1+
-- 4. Composite indexes significantly improve multi-property queries
-- 5. Monitor query performance with PROFILE/EXPLAIN before and after indexing
-- 6. Many indexes and constraints are already created by base and domain schemas
-- 7. This file focuses on additional optimizations and utility queries

