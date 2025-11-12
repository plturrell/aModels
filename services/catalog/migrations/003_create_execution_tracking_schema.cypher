-- +goose Up
-- Create Execution Tracking Schema for Neo4j
-- This schema enables tracking of job executions, SQL queries, and ETL processes

-- Execution nodes represent individual execution instances (jobs, queries, processes)
CREATE CONSTRAINT execution_id IF NOT EXISTS
FOR (e:Execution) REQUIRE e.id IS UNIQUE;

-- ExecutionMetrics nodes store detailed metrics for executions
CREATE CONSTRAINT execution_metrics_id IF NOT EXISTS
FOR (m:ExecutionMetrics) REQUIRE m.id IS UNIQUE;

-- ProcessEvent nodes represent individual steps/events within an execution
CREATE CONSTRAINT process_event_id IF NOT EXISTS
FOR (p:ProcessEvent) REQUIRE p.id IS UNIQUE;

-- Create indexes for common queries
CREATE INDEX execution_status IF NOT EXISTS
FOR (e:Execution) ON (e.status);

CREATE INDEX execution_started_at IF NOT EXISTS
FOR (e:Execution) ON (e.started_at);

CREATE INDEX execution_type IF NOT EXISTS
FOR (e:Execution) ON (e.execution_type);

CREATE INDEX execution_entity_id IF NOT EXISTS
FOR (e:Execution) ON (e.entity_id);

CREATE INDEX process_event_event_type IF NOT EXISTS
FOR (p:ProcessEvent) ON (p.event_type);

CREATE INDEX process_event_timestamp IF NOT EXISTS
FOR (p:ProcessEvent) ON (p.timestamp);

-- Create relationships
-- Execution -> ExecutionMetrics (HAS_METRICS)
-- Execution -> ProcessEvent (HAS_EVENT)
-- Execution -> Node (EXECUTES) - links to control-m-job, sql-query, etc.
-- Execution -> Node (AFFECTS) - links to tables/views affected by execution

-- +goose Down
-- Rollback Execution Tracking Schema
DROP INDEX process_event_timestamp IF EXISTS;
DROP INDEX process_event_event_type IF EXISTS;
DROP INDEX execution_entity_id IF EXISTS;
DROP INDEX execution_type IF EXISTS;
DROP INDEX execution_started_at IF EXISTS;
DROP INDEX execution_status IF EXISTS;
DROP CONSTRAINT process_event_id IF EXISTS;
DROP CONSTRAINT execution_metrics_id IF EXISTS;
DROP CONSTRAINT execution_id IF EXISTS;

