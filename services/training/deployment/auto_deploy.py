"""Automatic deployment trigger for domain models.

This module provides automatic deployment when training results meet thresholds.
"""

import os
import json
import logging
import httpx
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class AutoDeploymentTrigger:
    """Trigger automatic deployment when training thresholds are met."""
    
    def __init__(
        self,
        localai_url: Optional[str] = None,
        postgres_dsn: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        """Initialize auto-deployment trigger.
        
        Args:
            localai_url: LocalAI service URL
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis connection string for config sync
        """
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_DSN")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
    
    def check_and_deploy(
        self,
        domain_id: str,
        training_run_id: str,
        metrics: Dict[str, Any],
        model_path: str
    ) -> Dict[str, Any]:
        """Check if deployment threshold is met and deploy if so.
        
        Args:
            domain_id: Domain identifier
            training_run_id: Training run ID
            metrics: Performance metrics
            model_path: Path to trained model
        
        Returns:
            Deployment result
        """
        logger.info(f"Checking deployment threshold for domain {domain_id}")
        
        # Check thresholds
        thresholds = {
            "accuracy": float(os.getenv("DEPLOY_ACCURACY_THRESHOLD", "0.85")),
            "latency_ms": float(os.getenv("DEPLOY_LATENCY_THRESHOLD", "500")),
            "training_loss": float(os.getenv("DEPLOY_LOSS_THRESHOLD", "0.3")),
            "validation_loss": float(os.getenv("DEPLOY_VAL_LOSS_THRESHOLD", "0.35")),
        }
        
        # Check if all thresholds are met
        meets_threshold = True
        failures = []
        
        if metrics.get("accuracy", 0) < thresholds["accuracy"]:
            meets_threshold = False
            failures.append(f"accuracy ({metrics.get('accuracy', 0):.3f} < {thresholds['accuracy']})")
        
        if metrics.get("latency_ms", float("inf")) > thresholds["latency_ms"]:
            meets_threshold = False
            failures.append(f"latency ({metrics.get('latency_ms', float('inf'))} > {thresholds['latency_ms']})")
        
        if metrics.get("training_loss", float("inf")) > thresholds["training_loss"]:
            meets_threshold = False
            failures.append(f"training_loss ({metrics.get('training_loss', float('inf'))} > {thresholds['training_loss']})")
        
        if metrics.get("validation_loss", float("inf")) > thresholds["validation_loss"]:
            meets_threshold = False
            failures.append(f"validation_loss ({metrics.get('validation_loss', float('inf'))} > {thresholds['validation_loss']})")
        
        if not meets_threshold:
            logger.info(f"⚠️  Deployment threshold not met for {domain_id}: {', '.join(failures)}")
            return {
                "deployed": False,
                "reason": "threshold_not_met",
                "failures": failures,
                "thresholds": thresholds,
                "metrics": metrics,
            }
        
        # Deploy model
        logger.info(f"✅ Deployment threshold met for {domain_id}, deploying model")
        
        try:
            # Generate model version
            model_version = f"{domain_id}-v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Update domain config
            deployment_result = self._update_domain_config(
                domain_id=domain_id,
                model_path=model_path,
                model_version=model_version,
                training_run_id=training_run_id,
                metrics=metrics
            )
            
            # Sync to Redis
            if self.redis_url:
                self._sync_to_redis(domain_id)
            
            # Trigger LocalAI reload (if API available)
            self._trigger_localai_reload(domain_id)
            
            return {
                "deployed": True,
                "model_version": model_version,
                "deployment_result": deployment_result,
                "deployed_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"❌ Deployment failed for {domain_id}: {e}")
            return {
                "deployed": False,
                "reason": "deployment_error",
                "error": str(e),
            }
    
    def _update_domain_config(
        self,
        domain_id: str,
        model_path: str,
        model_version: str,
        training_run_id: str,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update domain configuration."""
        try:
            import psycopg2
            from psycopg2.extras import Json
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Get current config
            cursor.execute(
                "SELECT config_json FROM domain_configs WHERE domain_name = %s",
                (domain_id,)
            )
            row = cursor.fetchone()
            
            if row:
                config = row[0]
            else:
                # Get from LocalAI
                config = self._get_domain_config_from_localai(domain_id)
            
            # Update model path
            config["model_path"] = model_path
            config["model_version"] = model_version
            
            # Save to PostgreSQL
            query = """
                INSERT INTO domain_configs (
                    domain_name, config_json, training_run_id, model_version,
                    performance_metrics, updated_at
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
                (domain_id, Json(config), training_run_id, model_version, Json(metrics))
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return {"status": "updated", "domain_id": domain_id, "model_version": model_version}
            
        except Exception as e:
            logger.error(f"Failed to update domain config: {e}")
            raise
    
    def _get_domain_config_from_localai(self, domain_id: str) -> Dict[str, Any]:
        """Get domain config from LocalAI."""
        try:
            client = httpx.Client(timeout=10.0)
            response = client.get(f"{self.localai_url}/v1/domains")
            response.raise_for_status()
            
            data = response.json()
            for domain_info in data.get("data", []):
                if domain_info.get("id") == domain_id:
                    return domain_info.get("config", {})
            
            return {}
        except Exception as e:
            logger.warning(f"Failed to get domain config: {e}")
            return {}
    
    def _sync_to_redis(self, domain_id: str):
        """Sync domain config to Redis."""
        logger.info(f"Syncing domain {domain_id} to Redis")
        # Implementation would call sync service or update Redis directly
    
    def _trigger_localai_reload(self, domain_id: str):
        """Trigger LocalAI to reload domain configuration."""
        logger.info(f"Triggering LocalAI reload for domain {domain_id}")
        # Implementation would call LocalAI reload endpoint if available

