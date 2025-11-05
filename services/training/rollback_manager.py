"""Automatic rollback manager for domain models.

This module provides automatic rollback capabilities when model
performance degrades after deployment.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import httpx

logger = logging.getLogger(__name__)


class RollbackManager:
    """Manage automatic rollbacks when performance degrades."""
    
    def __init__(
        self,
        postgres_dsn: Optional[str] = None,
        redis_url: Optional[str] = None,
        localai_url: Optional[str] = None
    ):
        """Initialize rollback manager.
        
        Args:
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis connection string
            localai_url: LocalAI service URL
        """
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_DSN")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        
        # Rollback thresholds
        self.rollback_thresholds = {
            "accuracy_degradation": 0.05,  # 5% drop triggers rollback
            "latency_increase": 1.5,  # 1.5x increase triggers rollback
            "error_rate_increase": 0.1,  # 10% error rate triggers rollback
            "min_samples": 50,  # Minimum samples before rollback
        }
    
    def check_and_rollback(
        self,
        domain_id: str,
        current_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if rollback is needed and perform if so.
        
        Args:
            domain_id: Domain identifier
            current_metrics: Current performance metrics
        
        Returns:
            Rollback result
        """
        logger.info(f"Checking rollback conditions for domain {domain_id}")
        
        # Get baseline metrics (previous version)
        baseline_metrics = self._get_baseline_metrics(domain_id)
        
        if not baseline_metrics:
            logger.info(f"No baseline found for {domain_id}, skipping rollback check")
            return {
                "rollback_triggered": False,
                "reason": "no_baseline",
            }
        
        # Check if rollback is needed
        rollback_needed, reason = self._check_rollback_conditions(
            current_metrics,
            baseline_metrics
        )
        
        if not rollback_needed:
            return {
                "rollback_triggered": False,
                "reason": "metrics_acceptable",
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
            }
        
        # Perform rollback
        logger.warning(f"⚠️  Rollback triggered for {domain_id}: {reason}")
        
        rollback_result = self._perform_rollback(
            domain_id=domain_id,
            reason=reason,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics
        )
        
        return {
            "rollback_triggered": True,
            "reason": reason,
            "rollback_result": rollback_result,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
        }
    
    def _get_baseline_metrics(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Get baseline metrics from previous model version."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Get previous version (second most recent)
            cursor.execute(
                """
                SELECT model_version, performance_metrics, updated_at
                FROM domain_configs
                WHERE domain_name = %s
                ORDER BY updated_at DESC
                LIMIT 2
                """,
                (domain_id,)
            )
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if len(rows) >= 2:
                # Return second most recent (previous version)
                _, metrics_json, _ = rows[1]
                return metrics_json
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get baseline metrics: {e}")
            return None
    
    def _check_rollback_conditions(
        self,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if rollback conditions are met.
        
        Returns:
            Tuple of (should_rollback, reason)
        """
        # Check accuracy degradation
        current_accuracy = current_metrics.get("accuracy", 1.0)
        baseline_accuracy = baseline_metrics.get("accuracy", 1.0)
        
        if baseline_accuracy > 0:
            accuracy_drop = baseline_accuracy - current_accuracy
            accuracy_drop_pct = accuracy_drop / baseline_accuracy
            
            if accuracy_drop_pct >= self.rollback_thresholds["accuracy_degradation"]:
                return True, f"accuracy_degradation ({accuracy_drop_pct:.2%} drop)"
        
        # Check latency increase
        current_latency = current_metrics.get("latency_ms", 0)
        baseline_latency = baseline_metrics.get("latency_ms", 1)
        
        if baseline_latency > 0:
            latency_increase = current_latency / baseline_latency
            
            if latency_increase >= self.rollback_thresholds["latency_increase"]:
                return True, f"latency_increase ({latency_increase:.2f}x increase)"
        
        # Check error rate
        current_error_rate = current_metrics.get("error_rate", 0)
        
        if current_error_rate >= self.rollback_thresholds["error_rate_increase"]:
            return True, f"error_rate_increase ({current_error_rate:.2%})"
        
        return False, "metrics_acceptable"
    
    def _perform_rollback(
        self,
        domain_id: str,
        reason: str,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform rollback to previous version."""
        logger.info(f"Rolling back domain {domain_id} to previous version")
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Get previous version config
            cursor.execute(
                """
                SELECT model_version, config_json, performance_metrics
                FROM domain_configs
                WHERE domain_name = %s
                ORDER BY updated_at DESC
                LIMIT 2
                """,
                (domain_id,)
            )
            
            rows = cursor.fetchall()
            
            if len(rows) < 2:
                cursor.close()
                conn.close()
                return {"error": "No previous version found"}
            
            _, prev_config, prev_metrics = rows[1]
            
            # Rollback: restore previous version
            rollback_version = f"{prev_config.get('model_version', 'unknown')}_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute(
                """
                INSERT INTO domain_configs (
                    domain_name, config_json, model_version, performance_metrics,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (domain_name)
                DO UPDATE SET
                    config_json = EXCLUDED.config_json,
                    model_version = EXCLUDED.model_version,
                    performance_metrics = EXCLUDED.performance_metrics,
                    updated_at = NOW(),
                    version = domain_configs.version + 1
                """,
                (
                    domain_id,
                    json.dumps(prev_config),
                    rollback_version,
                    json.dumps(prev_metrics),
                )
            )
            
            conn.commit()
            
            # Log rollback event
            self._log_rollback_event(
                domain_id=domain_id,
                reason=reason,
                from_version=current_metrics.get("model_version", "current"),
                to_version=rollback_version,
                current_metrics=current_metrics,
                baseline_metrics=baseline_metrics
            )
            
            # Sync to Redis
            if self.redis_url:
                self._sync_to_redis(domain_id)
            
            # Trigger LocalAI reload
            self._trigger_localai_reload(domain_id)
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Rolled back {domain_id} to version {rollback_version}")
            
            return {
                "status": "rolled_back",
                "domain_id": domain_id,
                "rollback_version": rollback_version,
                "reason": reason,
                "rolled_back_at": datetime.now().isoformat(),
            }
            
        except Exception as e:
            logger.error(f"❌ Rollback failed: {e}")
            return {"error": str(e)}
    
    def _log_rollback_event(
        self,
        domain_id: str,
        reason: str,
        from_version: str,
        to_version: str,
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any]
    ):
        """Log rollback event for audit."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Create rollback events table if not exists
            cursor.execute("""
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
            """)
            
            # Insert rollback event
            cursor.execute(
                """
                INSERT INTO rollback_events (
                    domain_id, reason, from_version, to_version,
                    current_metrics, baseline_metrics
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    domain_id,
                    reason,
                    from_version,
                    to_version,
                    json.dumps(current_metrics),
                    json.dumps(baseline_metrics),
                )
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to log rollback event: {e}")
    
    def _sync_to_redis(self, domain_id: str):
        """Sync domain config to Redis after rollback."""
        logger.info(f"Syncing rolled back domain {domain_id} to Redis")
        # Implementation would sync to Redis
    
    def _trigger_localai_reload(self, domain_id: str):
        """Trigger LocalAI to reload domain configuration."""
        logger.info(f"Triggering LocalAI reload for domain {domain_id}")
        # Implementation would call LocalAI reload endpoint
    
    def get_rollback_history(
        self,
        domain_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get rollback history for a domain.
        
        Args:
            domain_id: Domain identifier
            limit: Maximum number of events to return
        
        Returns:
            List of rollback events
        """
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT reason, from_version, to_version, current_metrics, 
                       baseline_metrics, created_at
                FROM rollback_events
                WHERE domain_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (domain_id, limit)
            )
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            events = []
            for row in rows:
                reason, from_version, to_version, current_metrics, baseline_metrics, created_at = row
                events.append({
                    "reason": reason,
                    "from_version": from_version,
                    "to_version": to_version,
                    "current_metrics": current_metrics,
                    "baseline_metrics": baseline_metrics,
                    "created_at": created_at.isoformat() if created_at else None,
                })
            
            return events
            
        except Exception as e:
            logger.warning(f"Failed to get rollback history: {e}")
            return []

