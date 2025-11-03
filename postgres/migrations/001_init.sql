-- +goose Up
CREATE TABLE IF NOT EXISTS lang_operations (
    id UUID PRIMARY KEY,
    library_type TEXT NOT NULL,
    operation TEXT NOT NULL,
    input JSONB NOT NULL DEFAULT '{}'::jsonb,
    output JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL,
    error TEXT,
    latency_ms BIGINT,
    session_id TEXT,
    user_id_hash TEXT,
    privacy_level TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_lang_operations_library_type ON lang_operations(library_type);
CREATE INDEX IF NOT EXISTS idx_lang_operations_session_id ON lang_operations(session_id);
CREATE INDEX IF NOT EXISTS idx_lang_operations_status ON lang_operations(status);
CREATE INDEX IF NOT EXISTS idx_lang_operations_created_at ON lang_operations(created_at);

CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    state JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS privacy_budget (
    user_id_hash TEXT PRIMARY KEY,
    privacy_level TEXT NOT NULL,
    remaining_tokens BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- +goose Down
DROP TABLE IF EXISTS privacy_budget;
DROP TABLE IF EXISTS session_state;
DROP TABLE IF EXISTS lang_operations;
