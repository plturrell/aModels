-- +goose Up
-- Create ISO 11179 structures in Neo4j
-- This creates the semantic metadata structures for the catalog

-- Create namespace prefixes for ISO 11179
MERGE (n:Namespace {prefix: 'iso11179', uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179'})
MERGE (n2:Namespace {prefix: 'rdf', uri: 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'})
MERGE (n3:Namespace {prefix: 'rdfs', uri: 'http://www.w3.org/2000/01/rdf-schema#'})
MERGE (n4:Namespace {prefix: 'owl', uri: 'http://www.w3.org/2002/07/owl#'});

-- Create base ontology classes
MERGE (c:Class {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#DataElementConcept'})
MERGE (c2:Class {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#Representation'})
MERGE (c3:Class {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#DataElement'})
MERGE (c4:Class {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#ValueDomain'});

-- Create properties
MERGE (p:Property {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#hasDataElementConcept'})
MERGE (p2:Property {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#hasRepresentation'})
MERGE (p3:Property {uri: 'http://metadata.dod.mil/mdr/ns/ISO/IEC11179#hasValueDomain'});

-- +goose Down
-- Rollback ISO 11179 structures
MATCH (p:Property) WHERE p.uri CONTAINS 'ISO/IEC11179' DELETE p;
MATCH (c:Class) WHERE c.uri CONTAINS 'ISO/IEC11179' DELETE c;
MATCH (n:Namespace) WHERE n.prefix IN ['iso11179', 'rdf', 'rdfs', 'owl'] DELETE n;

