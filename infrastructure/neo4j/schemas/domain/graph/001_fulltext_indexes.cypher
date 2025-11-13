-- Graph Service: Full-text Indexes
-- This schema adds full-text search indexes for labels and descriptions

-- Full-text index on node labels for search
CREATE FULLTEXT INDEX node_label_fulltext IF NOT EXISTS
FOR (n:Node) ON EACH [n.label, n.id];

-- Full-text index on BCBS239 principles
CREATE FULLTEXT INDEX bcbs_principle_fulltext IF NOT EXISTS
FOR (p:BCBS239Principle) ON EACH [p.principle_name, p.description];

