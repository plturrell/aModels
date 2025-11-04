#!/usr/bin/env python3
"""
Load domain configurations from training results into PostgreSQL.
This script should be run after training to update domain configs.
"""

import os
import json
import sys
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any

def load_domain_config_from_training(
    postgres_dsn: str,
    domain_name: str,
    config: Dict[str, Any],
    training_run_id: str,
    model_version: str,
    performance_metrics: Dict[str, Any]
):
    """Load a domain configuration from training results into PostgreSQL."""
    
    conn = psycopg2.connect(postgres_dsn)
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO domain_configs (
                domain_name, 
                config_json, 
                training_run_id, 
                model_version, 
                performance_metrics,
                updated_at
            )
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (domain_name) 
            DO UPDATE SET 
                config_json = EXCLUDED.config_json,
                training_run_id = EXCLUDED.training_run_id,
                model_version = EXCLUDED.model_version,
                performance_metrics = EXCLUDED.performance_metrics,
                updated_at = NOW(),
                version = domain_configs.version + 1
        """
        
        cursor.execute(
            query,
            (
                domain_name,
                Json(config),
                training_run_id,
                model_version,
                Json(performance_metrics)
            )
        )
        
        conn.commit()
        print(f"✅ Loaded domain config for '{domain_name}' from training run '{training_run_id}'")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error loading domain config: {e}", file=sys.stderr)
        raise
    finally:
        cursor.close()
        conn.close()

def main():
    postgres_dsn = os.getenv(
        "POSTGRES_DSN",
        "postgres://postgres:postgres@localhost:5432/amodels?sslmode=disable"
    )
    
    # Example: Load a domain config from training
    # This would be called after training completes
    domain_config = {
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
    }
    
    performance_metrics = {
        "accuracy": 0.85,
        "latency_ms": 120,
        "tokens_per_second": 45.2,
        "training_loss": 0.023,
        "validation_loss": 0.028
    }
    
    load_domain_config_from_training(
        postgres_dsn=postgres_dsn,
        domain_name="general",
        config=domain_config,
        training_run_id=os.getenv("TRAINING_RUN_ID", "training_run_001"),
        model_version=os.getenv("MODEL_VERSION", "phi-3.5-mini-v1"),
        performance_metrics=performance_metrics
    )

if __name__ == "__main__":
    main()

