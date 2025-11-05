-- +goose Up
-- Create triplestore schema for RDF triple storage
-- This extends the Neo4j schema with triplestore-specific structures

-- Create Triple nodes for explicit triple storage
CREATE CONSTRAINT triple_id IF NOT EXISTS
FOR (t:Triple) REQUIRE t.id IS UNIQUE;

-- Create index for triple lookups
CREATE INDEX triple_subject IF NOT EXISTS
FOR (t:Triple) ON (t.subject);

CREATE INDEX triple_predicate IF NOT EXISTS
FOR (t:Triple) ON (t.predicate);

CREATE INDEX triple_object IF NOT EXISTS
FOR (t:Triple) ON (t.object);

-- Create Graph nodes for named graphs (RDF contexts)
CREATE CONSTRAINT graph_uri IF NOT EXISTS
FOR (g:Graph) REQUIRE g.uri IS UNIQUE;

-- Create relationships for triples
-- Triple -> Resource (subject)
-- Triple -> Resource (object)
-- Triple -> Graph (context)

-- +goose Down
-- Rollback triplestore schema
DROP CONSTRAINT graph_uri IF EXISTS;
DROP INDEX triple_object IF EXISTS;
DROP INDEX triple_predicate IF EXISTS;
DROP INDEX triple_subject IF EXISTS;
DROP CONSTRAINT triple_id IF EXISTS;

