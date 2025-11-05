-- +goose Up
CREATE TABLE IF NOT EXISTS extract_entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT,
    entity_class TEXT,
    entity_text TEXT,
    attributes_json TEXT,
    source TEXT,
    extracted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS checkpoints (
    key TEXT PRIMARY KEY,
    payload BLOB
);

-- +goose Down
DROP TABLE IF EXISTS checkpoints;
DROP TABLE IF EXISTS extract_entities;
