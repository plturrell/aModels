-- Catalog Service: Base Neo4j Schema
-- This creates the basic structure for storing RDF triples and catalog metadata

-- ============================================================================
-- UP MIGRATION
-- ============================================================================

-- Create Resource nodes (for RDF subjects and objects)
CREATE CONSTRAINT resource_uri IF NOT EXISTS
FOR (r:Resource) REQUIRE r.uri IS UNIQUE;

-- Create DataElementConcept nodes
CREATE CONSTRAINT data_element_concept_id IF NOT EXISTS
FOR (d:DataElementConcept) REQUIRE d.id IS UNIQUE;

-- Create Representation nodes
CREATE CONSTRAINT representation_id IF NOT EXISTS
FOR (r:Representation) REQUIRE r.id IS UNIQUE;

-- Create DataElement nodes
CREATE CONSTRAINT data_element_id IF NOT EXISTS
FOR (d:DataElement) REQUIRE d.id IS UNIQUE;

-- Create ValueDomain nodes
CREATE CONSTRAINT value_domain_id IF NOT EXISTS
FOR (v:ValueDomain) REQUIRE v.id IS UNIQUE;

-- Create indexes for common queries
CREATE INDEX resource_updated_at IF NOT EXISTS
FOR (r:Resource) ON (r.updated_at);

CREATE INDEX data_element_name IF NOT EXISTS
FOR (d:DataElement) ON (d.name);

CREATE INDEX data_element_concept_name IF NOT EXISTS
FOR (d:DataElementConcept) ON (d.name);

-- Create relationships
-- DataElementConcept -> Representation -> DataElement
-- DataElement -> ValueDomain
-- Resource -> Resource (for RDF triples)

-- ============================================================================
-- DOWN MIGRATION
-- ============================================================================
-- Execute these statements in reverse order to rollback this migration
--
-- DROP INDEX data_element_concept_name IF EXISTS;
-- DROP INDEX data_element_name IF EXISTS;
-- DROP INDEX resource_updated_at IF EXISTS;
-- DROP CONSTRAINT value_domain_id IF EXISTS;
-- DROP CONSTRAINT data_element_id IF EXISTS;
-- DROP CONSTRAINT representation_id IF EXISTS;
-- DROP CONSTRAINT data_element_concept_id IF EXISTS;
-- DROP CONSTRAINT resource_uri IF EXISTS;
