-- Base Node and Relationship Indexes
-- This schema creates fundamental indexes for Node and RELATIONSHIP types
-- Must be applied after base constraints

-- Node indexes for common queries
CREATE INDEX node_id_index IF NOT EXISTS FOR (n:Node) ON (n.id);
CREATE INDEX node_type_index IF NOT EXISTS FOR (n:Node) ON (n.type);
CREATE INDEX node_label_index IF NOT EXISTS FOR (n:Node) ON (n.label);
CREATE INDEX node_updated_at_index IF NOT EXISTS FOR (n:Node) ON (n.updated_at);

-- Relationship indexes
CREATE INDEX rel_label_index IF NOT EXISTS FOR ()-[r:RELATIONSHIP]-() ON (r.label);

