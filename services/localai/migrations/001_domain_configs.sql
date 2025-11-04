-- Migration: Create domain_configs table for storing domain configurations
-- This table links domain configs to training runs and model versions

CREATE TABLE IF NOT EXISTS domain_configs (
    id SERIAL PRIMARY KEY,
    domain_name VARCHAR(255) UNIQUE NOT NULL,
    config_json JSONB NOT NULL,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    -- Link to training process
    training_run_id VARCHAR(255),
    model_version VARCHAR(255),
    performance_metrics JSONB,
    -- Metadata
    description TEXT,
    tags TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_domain_configs_enabled ON domain_configs(enabled);
CREATE INDEX IF NOT EXISTS idx_domain_configs_training_run ON domain_configs(training_run_id);
CREATE INDEX IF NOT EXISTS idx_domain_configs_model_version ON domain_configs(model_version);
CREATE INDEX IF NOT EXISTS idx_domain_configs_updated_at ON domain_configs(updated_at);

COMMENT ON TABLE domain_configs IS 'Stores LocalAI domain configurations, linked to training runs';
COMMENT ON COLUMN domain_configs.config_json IS 'Full domain configuration as JSON (matches DomainConfig struct)';
COMMENT ON COLUMN domain_configs.training_run_id IS 'Links to training run that produced this model version';
COMMENT ON COLUMN domain_configs.model_version IS 'Model version identifier from training';
COMMENT ON COLUMN domain_configs.performance_metrics IS 'Performance metrics from training/evaluation';

