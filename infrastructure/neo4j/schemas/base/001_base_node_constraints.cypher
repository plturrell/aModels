-- Base Node Constraints
-- This schema creates the fundamental constraints for Node labels
-- Must be applied before any other schemas that use Node labels

-- Node ID must be unique
CREATE CONSTRAINT node_id_unique IF NOT EXISTS 
FOR (n:Node) REQUIRE n.id IS UNIQUE;

-- Node ID must not be null
CREATE CONSTRAINT node_id_not_null IF NOT EXISTS 
FOR (n:Node) REQUIRE n.id IS NOT NULL;

