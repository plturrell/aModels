-- +goose Up
-- Create Performance Metrics Schema for Neo4j
-- This schema enables tracking of performance metrics for queries, jobs, and processes

-- PerformanceMetric nodes store performance measurements
CREATE CONSTRAINT performance_metric_id IF NOT EXISTS
FOR (p:PerformanceMetric) REQUIRE p.id IS UNIQUE;

-- QueryPerformance nodes store detailed query performance data
CREATE CONSTRAINT query_performance_id IF NOT EXISTS
FOR (q:QueryPerformance) REQUIRE q.id IS UNIQUE;

-- Create indexes for common queries
CREATE INDEX performance_metric_entity_id IF NOT EXISTS
FOR (p:PerformanceMetric) ON (p.entity_id);

CREATE INDEX performance_metric_metric_type IF NOT EXISTS
FOR (p:PerformanceMetric) ON (p.metric_type);

CREATE INDEX performance_metric_timestamp IF NOT EXISTS
FOR (p:PerformanceMetric) ON (p.timestamp);

CREATE INDEX query_performance_query_id IF NOT EXISTS
FOR (q:QueryPerformance) ON (q.query_id);

CREATE INDEX query_performance_execution_id IF NOT EXISTS
FOR (q:QueryPerformance) ON (q.execution_id);

CREATE INDEX query_performance_timestamp IF NOT EXISTS
FOR (q:QueryPerformance) ON (q.timestamp);

-- Create relationships
-- PerformanceMetric -> Node (MEASURES) - links to entity being measured (table, query, job)
-- QueryPerformance -> Execution (PART_OF) - links query performance to execution
-- QueryPerformance -> Node (MEASURES) - links to SQL query node

-- +goose Down
-- Rollback Performance Metrics Schema
DROP INDEX query_performance_timestamp IF EXISTS;
DROP INDEX query_performance_execution_id IF EXISTS;
DROP INDEX query_performance_query_id IF EXISTS;
DROP INDEX performance_metric_timestamp IF EXISTS;
DROP INDEX performance_metric_metric_type IF EXISTS;
DROP INDEX performance_metric_entity_id IF EXISTS;
DROP CONSTRAINT query_performance_id IF EXISTS;
DROP CONSTRAINT performance_metric_id IF EXISTS;

