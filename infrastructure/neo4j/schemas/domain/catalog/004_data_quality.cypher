-- Catalog Service: Data Quality Schema
-- This schema enables tracking of data quality issues and metrics

-- QualityIssue nodes represent data quality problems
CREATE CONSTRAINT quality_issue_id IF NOT EXISTS
FOR (q:QualityIssue) REQUIRE q.id IS UNIQUE;

-- QualityMetric nodes store aggregated quality metrics for entities
CREATE CONSTRAINT quality_metric_id IF NOT EXISTS
FOR (q:QualityMetric) REQUIRE q.id IS UNIQUE;

-- Create indexes for common queries
CREATE INDEX quality_issue_type IF NOT EXISTS
FOR (q:QualityIssue) ON (q.issue_type);

CREATE INDEX quality_issue_severity IF NOT EXISTS
FOR (q:QualityIssue) ON (q.severity);

CREATE INDEX quality_issue_entity_id IF NOT EXISTS
FOR (q:QualityIssue) ON (q.entity_id);

CREATE INDEX quality_issue_created_at IF NOT EXISTS
FOR (q:QualityIssue) ON (q.created_at);

CREATE INDEX quality_metric_entity_id IF NOT EXISTS
FOR (q:QualityMetric) ON (q.entity_id);

CREATE INDEX quality_metric_metric_type IF NOT EXISTS
FOR (q:QualityMetric) ON (q.metric_type);

-- Create relationships
-- QualityIssue -> Node (AFFECTS) - links to table/column/view with quality issue
-- QualityMetric -> Node (MEASURES) - links to entity being measured
-- QualityIssue -> QualityMetric (RELATES_TO) - links issues to aggregated metrics

