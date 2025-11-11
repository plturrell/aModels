#!/usr/bin/env python3
"""
Load domain configurations from training results into PostgreSQL with differential privacy.
This script should be run after training to update domain configs.
"""

import os
import json
import sys
import psycopg2
from psycopg2.extras import Json
from typing import Dict, Any
import numpy as np

def add_differential_privacy_noise(
    value: float,
    epsilon: float = 1.0,
    sensitivity: float = 1.0
) -> float:
    """Add Laplacian noise for differential privacy.
    
    Args:
        value: Original value
        epsilon: Privacy budget (ε)
        sensitivity: Sensitivity of the query
    
    Returns:
        Value with added noise
    """
    if epsilon <= 0:
        return value
    
    # Laplacian noise: Lap(sensitivity / epsilon)
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale)
    return value + noise


def apply_privacy_to_metrics(metrics: Dict[str, Any], epsilon: float = 1.0) -> Dict[str, Any]:
    """Apply differential privacy to performance metrics.
    
    Args:
        metrics: Performance metrics dictionary
        epsilon: Privacy budget (ε)
    
    Returns:
        Metrics with added noise
    """
    private_metrics = {}
    
    # Sensitivity for different metric types
    sensitivities = {
        "accuracy": 0.01,  # 1% sensitivity for accuracy
        "latency_ms": 10.0,  # 10ms sensitivity for latency
        "tokens_per_second": 1.0,  # 1 token/s sensitivity
        "training_loss": 0.001,  # 0.1% sensitivity for loss
        "validation_loss": 0.001,
    }
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            sensitivity = sensitivities.get(key, 1.0)
            private_metrics[key] = max(0.0, add_differential_privacy_noise(
                float(value), epsilon, sensitivity
            ))
        else:
            private_metrics[key] = value
    
    return private_metrics


def load_domain_config_from_training(
    postgres_dsn: str,
    domain_name: str,
    config: Dict[str, Any],
    training_run_id: str,
    model_version: str,
    performance_metrics: Dict[str, Any],
    apply_privacy: bool = True,
    epsilon: float = 1.0
):
    """Load a domain configuration from training results into PostgreSQL with differential privacy.
    
    Args:
        postgres_dsn: PostgreSQL connection string
        domain_name: Domain name
        config: Domain configuration
        training_run_id: Training run ID
        model_version: Model version
        performance_metrics: Performance metrics
        apply_privacy: Whether to apply differential privacy
        epsilon: Privacy budget (ε)
    """
    
    # Apply differential privacy to metrics if enabled
    if apply_privacy:
        performance_metrics = apply_privacy_to_metrics(performance_metrics, epsilon)
        print(f"🔒 Applied differential privacy (ε={epsilon}) to performance metrics")
    
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
        privacy_status = "with differential privacy" if apply_privacy else "without privacy"
        print(f"✅ Loaded domain config for '{domain_name}' from training run '{training_run_id}' {privacy_status}")
        
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
    
    # Get privacy configuration from environment
    apply_privacy = os.getenv("APPLY_DIFFERENTIAL_PRIVACY", "true").lower() == "true"
    epsilon = float(os.getenv("PRIVACY_EPSILON", "1.0"))
    
    load_domain_config_from_training(
        postgres_dsn=postgres_dsn,
        domain_name="general",
        config=domain_config,
        training_run_id=os.getenv("TRAINING_RUN_ID", "training_run_001"),
        model_version=os.getenv("MODEL_VERSION", "phi-3.5-mini-v1"),
        performance_metrics=performance_metrics,
        apply_privacy=apply_privacy,
        epsilon=epsilon
    )

if __name__ == "__main__":
    main()

