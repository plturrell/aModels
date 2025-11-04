-- Initialize domain_configs table with data from domains.json
-- This can be run after training to populate the database

-- Example: Insert a domain config from training
INSERT INTO domain_configs (domain_name, config_json, training_run_id, model_version, performance_metrics)
VALUES (
    'general',
    '{
        "name": "General Assistant",
        "layer": "general",
        "team": "General",
        "backend_type": "hf-transformers",
        "model_name": "phi-3.5-mini",
        "transformers_config": {
            "endpoint": "http://transformers-service:9090/v1/chat/completions",
            "model_name": "phi-3.5-mini",
            "timeout_seconds": 120
        },
        "max_tokens": 1024,
        "temperature": 0.7,
        "tags": ["general", "conversation", "qa"]
    }'::jsonb,
    'training_run_001',
    'phi-3.5-mini-v1',
    '{
        "accuracy": 0.85,
        "latency_ms": 120,
        "tokens_per_second": 45.2
    }'::jsonb
)
ON CONFLICT (domain_name) 
DO UPDATE SET 
    config_json = EXCLUDED.config_json,
    training_run_id = EXCLUDED.training_run_id,
    model_version = EXCLUDED.model_version,
    performance_metrics = EXCLUDED.performance_metrics,
    updated_at = NOW(),
    version = domain_configs.version + 1;

