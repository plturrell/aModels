-- +goose Up
CREATE TABLE IF NOT EXISTS glean_nodes (
    id TEXT PRIMARY KEY,
    kind TEXT,
    label TEXT,
    properties_json JSONB,
    updated_at_utc TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS glean_edges (
    source_id TEXT,
    target_id TEXT,
    label TEXT,
    properties_json JSONB,
    updated_at_utc TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (source_id, target_id, label)
);

-- +goose Down
DROP TABLE IF EXISTS glean_edges;
DROP TABLE IF EXISTS glean_nodes;
