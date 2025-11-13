-- Regulatory Service: BCBS239 Compliance Schema
-- This schema models BCBS 239 compliance with principles, controls, and calculations

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- BCBS239 Principle constraints
CREATE CONSTRAINT bcbs239_principle_id IF NOT EXISTS FOR (p:BCBS239Principle) REQUIRE p.principle_id IS UNIQUE;

-- BCBS239 Control constraints
CREATE CONSTRAINT bcbs239_control_id IF NOT EXISTS FOR (c:BCBS239Control) REQUIRE c.control_id IS UNIQUE;

-- Regulatory Calculation constraints
CREATE CONSTRAINT bcbs239_calculation_id IF NOT EXISTS FOR (c:RegulatoryCalculation) REQUIRE c.calculation_id IS UNIQUE;

-- Data Asset constraints
CREATE CONSTRAINT bcbs239_data_asset_id IF NOT EXISTS FOR (d:DataAsset) REQUIRE d.asset_id IS UNIQUE;

-- Process constraints
CREATE CONSTRAINT bcbs239_process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE;

-- Indexes for efficient queries
CREATE INDEX bcbs239_principle_area IF NOT EXISTS FOR (p:BCBS239Principle) ON (p.compliance_area);
CREATE INDEX bcbs239_control_type IF NOT EXISTS FOR (c:BCBS239Control) ON (c.control_type);
CREATE INDEX bcbs239_calculation_framework IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.regulatory_framework);
CREATE INDEX bcbs239_calculation_date IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.calculation_date);
CREATE INDEX bcbs239_data_asset_type IF NOT EXISTS FOR (d:DataAsset) ON (d.asset_type);

-- Composite indexes for BCBS239 (included here for domain completeness)
CREATE INDEX bcbs239_principle_area_priority IF NOT EXISTS FOR (p:BCBS239Principle) ON (p.compliance_area, p.priority);
CREATE INDEX bcbs239_calculation_date_framework IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.regulatory_framework, c.calculation_date);
CREATE INDEX bcbs239_calculation_status IF NOT EXISTS FOR (c:RegulatoryCalculation) ON (c.status, c.calculation_date);
CREATE INDEX bcbs239_data_asset_type_id IF NOT EXISTS FOR (d:DataAsset) ON (d.asset_type, d.asset_id);

-- ============================================================================
-- DOWN MIGRATION
-- ============================================================================
-- Execute these statements in reverse order to rollback this migration
--
-- DROP INDEX bcbs239_data_asset_type_id IF EXISTS;
-- DROP INDEX bcbs239_calculation_status IF EXISTS;
-- DROP INDEX bcbs239_calculation_date_framework IF EXISTS;
-- DROP INDEX bcbs239_principle_area_priority IF EXISTS;
-- DROP INDEX bcbs239_data_asset_type IF EXISTS;
-- DROP INDEX bcbs239_calculation_date IF EXISTS;
-- DROP INDEX bcbs239_calculation_framework IF EXISTS;
-- DROP INDEX bcbs239_control_type IF EXISTS;
-- DROP INDEX bcbs239_principle_area IF EXISTS;
-- DROP CONSTRAINT bcbs239_process_id IF EXISTS;
-- DROP CONSTRAINT bcbs239_data_asset_id IF EXISTS;
-- DROP CONSTRAINT bcbs239_calculation_id IF EXISTS;
-- DROP CONSTRAINT bcbs239_control_id IF EXISTS;
-- DROP CONSTRAINT bcbs239_principle_id IF EXISTS;
