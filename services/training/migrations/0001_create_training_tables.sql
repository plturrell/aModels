-- +goose Up
CREATE TABLE IF NOT EXISTS routing_weights (
    domain_id VARCHAR(255) PRIMARY KEY,
    weight FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ab_tests (
    test_id VARCHAR(255) PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    test_config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    start_date TIMESTAMP DEFAULT NOW(),
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ab_tests_domain_status
    ON ab_tests(domain_id, status);

CREATE TABLE IF NOT EXISTS rollback_events (
    id SERIAL PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    reason TEXT,
    from_version VARCHAR(255),
    to_version VARCHAR(255),
    current_metrics JSONB,
    baseline_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rollback_events_domain
    ON rollback_events(domain_id, created_at);

-- +goose Down
DROP TABLE IF EXISTS rollback_events;
DROP TABLE IF EXISTS ab_tests;
DROP TABLE IF EXISTS routing_weights;
