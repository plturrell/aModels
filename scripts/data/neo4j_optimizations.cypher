// Neo4j Performance Optimization Queries
// Execute these to improve query performance across common access patterns

// ============================================================================
// COMPOSITE INDEXES
// ============================================================================

// Index for common table lookups (type + label)
CREATE INDEX node_table_lookup IF NOT EXISTS 
FOR (n:Node) ON (n.type, n.label);

// Index for temporal queries with type filtering
CREATE INDEX node_type_temporal IF NOT EXISTS 
FOR (n:Node) ON (n.type, n.updated_at);

// Index for relationship label and temporal queries
CREATE INDEX relationship_label_temporal IF NOT EXISTS 
FOR ()-[r:RELATIONSHIP]-() ON (r.label, r.updated_at);

// Index for BCBS239 principle area and priority
CREATE INDEX bcbs_principle_area_priority IF NOT EXISTS 
FOR (p:BCBS239Principle) ON (p.compliance_area, p.priority);

// Index for regulatory calculation date and framework
CREATE INDEX reg_calc_date_framework IF NOT EXISTS 
FOR (c:RegulatoryCalculation) ON (c.regulatory_framework, c.calculation_date);

// Index for regulatory calculation status
CREATE INDEX reg_calc_status IF NOT EXISTS 
FOR (c:RegulatoryCalculation) ON (c.status, c.calculation_date);

// Index for data asset type
CREATE INDEX data_asset_type_id IF NOT EXISTS 
FOR (d:DataAsset) ON (d.asset_type, d.asset_id);

// Index for Control-M job lookups
CREATE INDEX controlm_job_lookup IF NOT EXISTS 
FOR (n:Node) ON (n.type, n.properties_json);

// ============================================================================
// FULL-TEXT INDEXES (for search capabilities)
// ============================================================================

// Full-text index on node labels for search
CREATE FULLTEXT INDEX node_label_fulltext IF NOT EXISTS
FOR (n:Node) ON EACH [n.label, n.id];

// Full-text index on BCBS239 principles
CREATE FULLTEXT INDEX bcbs_principle_fulltext IF NOT EXISTS
FOR (p:BCBS239Principle) ON EACH [p.principle_name, p.description];

// ============================================================================
// CONSTRAINTS (ensure data integrity)
// ============================================================================

// Ensure node IDs are unique and not null
CREATE CONSTRAINT node_id_unique IF NOT EXISTS 
FOR (n:Node) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT node_id_not_null IF NOT EXISTS 
FOR (n:Node) REQUIRE n.id IS NOT NULL;

// Ensure BCBS239 principle IDs are unique
CREATE CONSTRAINT bcbs239_principle_id IF NOT EXISTS 
FOR (p:BCBS239Principle) REQUIRE p.principle_id IS UNIQUE;

// Ensure control IDs are unique
CREATE CONSTRAINT bcbs239_control_id IF NOT EXISTS 
FOR (c:BCBS239Control) REQUIRE c.control_id IS UNIQUE;

// Ensure calculation IDs are unique
CREATE CONSTRAINT bcbs239_calculation_id IF NOT EXISTS 
FOR (c:RegulatoryCalculation) REQUIRE c.calculation_id IS UNIQUE;

// Ensure data asset IDs are unique
CREATE CONSTRAINT bcbs239_data_asset_id IF NOT EXISTS 
FOR (d:DataAsset) REQUIRE d.asset_id IS UNIQUE;

// ============================================================================
// QUERY OPTIMIZATION - Warm up indexes
// ============================================================================

// Warm up node type index
MATCH (n:Node) WHERE n.type IS NOT NULL RETURN count(n);

// Warm up relationship label index
MATCH ()-[r:RELATIONSHIP]->() WHERE r.label IS NOT NULL RETURN count(r);

// Warm up BCBS239 indexes
MATCH (p:BCBS239Principle) RETURN count(p);

// ============================================================================
// DATABASE STATISTICS
// ============================================================================

// View all indexes
SHOW INDEXES;

// View all constraints
SHOW CONSTRAINTS;

// Database statistics
CALL db.stats.retrieve('GRAPH COUNTS');

// ============================================================================
// PERFORMANCE ANALYSIS QUERIES
// ============================================================================

// Find nodes without proper type
MATCH (n:Node) WHERE n.type IS NULL RETURN count(n) as untyped_nodes;

// Find relationships without proper label
MATCH ()-[r:RELATIONSHIP]->() WHERE r.label IS NULL RETURN count(r) as unlabeled_relationships;

// Count nodes by type (for capacity planning)
MATCH (n:Node) 
RETURN n.type as type, count(n) as count 
ORDER BY count DESC;

// Count relationships by label (for capacity planning)
MATCH ()-[r:RELATIONSHIP]->() 
RETURN r.label as label, count(r) as count 
ORDER BY count DESC;

// Identify high-degree nodes (potential performance bottlenecks)
MATCH (n:Node)
WITH n, size((n)--()) as degree
WHERE degree > 100
RETURN n.id, n.type, n.label, degree
ORDER BY degree DESC
LIMIT 50;

// ============================================================================
// CLEANUP QUERIES (run periodically)
// ============================================================================

// Remove orphaned nodes (nodes with no relationships)
// CAUTION: Only run if this is expected behavior
// MATCH (n:Node) WHERE NOT (n)--() DELETE n;

// Find duplicate nodes (same ID)
MATCH (n:Node)
WITH n.id as node_id, collect(n) as nodes
WHERE size(nodes) > 1
RETURN node_id, size(nodes) as duplicates;

// ============================================================================
// NOTES
// ============================================================================

// 1. Run this script after initial data load or schema changes
// 2. Indexes are created asynchronously - check SHOW INDEXES for status
// 3. Full-text indexes require Neo4j 4.1+
// 4. Composite indexes significantly improve multi-property queries
// 5. Monitor query performance with PROFILE/EXPLAIN before and after indexing
