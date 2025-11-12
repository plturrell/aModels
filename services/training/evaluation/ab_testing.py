"""A/B testing for domain models.

This module provides A/B testing capabilities to compare multiple
model versions for a domain.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import httpx

logger = logging.getLogger(__name__)


class ABTestManager:
    """Manage A/B testing for domain models."""
    
    def __init__(
        self,
        postgres_dsn: Optional[str] = None,
        redis_url: Optional[str] = None
    ):
        """Initialize A/B test manager.
        
        Args:
            postgres_dsn: PostgreSQL connection string
            redis_url: Redis connection string for traffic splitting
        """
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_DSN")
        self.redis_url = redis_url or os.getenv("REDIS_URL")
    
    def create_ab_test(
        self,
        domain_id: str,
        variant_a: Dict[str, Any],
        variant_b: Dict[str, Any],
        traffic_split: float = 0.5,
        duration_days: int = 7
    ) -> Dict[str, Any]:
        """Create an A/B test for a domain.
        
        Args:
            domain_id: Domain identifier
            variant_a: Configuration for variant A (control)
            variant_b: Configuration for variant B (treatment)
            traffic_split: Percentage of traffic to variant B (0.0-1.0)
            duration_days: Test duration in days
        
        Returns:
            A/B test configuration
        """
        test_id = f"ab_test_{domain_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        ab_test = {
            "test_id": test_id,
            "domain_id": domain_id,
            "variant_a": variant_a,
            "variant_b": variant_b,
            "traffic_split": traffic_split,
            "start_date": datetime.now().isoformat(),
            "end_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "status": "active",
            "metrics": {
                "variant_a": defaultdict(int),
                "variant_b": defaultdict(int),
            },
        }
        
        # Save to PostgreSQL
        if self.postgres_dsn:
            self._save_ab_test(ab_test)
        
        logger.info(f"✅ Created A/B test {test_id} for domain {domain_id}")
        
        return ab_test
    
    def route_request(
        self,
        domain_id: str,
        request_id: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Route a request to A or B variant.
        
        Args:
            domain_id: Domain identifier
            request_id: Unique request identifier
        
        Returns:
            Tuple of (variant, variant_config)
        """
        # Get active A/B test for domain
        ab_test = self._get_active_ab_test(domain_id)
        
        if not ab_test:
            # No active test, use default
            return "default", {}
        
        # Use consistent hashing based on request_id
        hash_value = hash(request_id) % 100
        threshold = int(ab_test["traffic_split"] * 100)
        
        if hash_value < threshold:
            variant = "variant_b"
            variant_config = ab_test["variant_b"]
        else:
            variant = "variant_a"
            variant_config = ab_test["variant_a"]
        
        # Track routing decision
        self._track_routing(ab_test["test_id"], variant, request_id)
        
        return variant, variant_config
    
    def record_metric(
        self,
        test_id: str,
        variant: str,
        metric_name: str,
        value: float
    ):
        """Record a metric for an A/B test variant.
        
        Args:
            test_id: A/B test identifier
            variant: Variant name (variant_a or variant_b)
            metric_name: Metric name (accuracy, latency_ms, etc.)
            value: Metric value
        """
        ab_test = self._get_ab_test(test_id)
        if not ab_test:
            return
        
        if variant not in ab_test["metrics"]:
            ab_test["metrics"][variant] = defaultdict(int)
        
        metrics = ab_test["metrics"][variant]
        
        # Track count and sum for averaging
        count_key = f"{metric_name}_count"
        sum_key = f"{metric_name}_sum"
        
        metrics[count_key] += 1
        metrics[sum_key] += value
        
        # Save updated metrics
        if self.postgres_dsn:
            self._update_ab_test_metrics(ab_test)
        
        logger.debug(f"Recorded {metric_name}={value} for {variant} in test {test_id}")
    
    def get_ab_test_results(
        self,
        test_id: str
    ) -> Dict[str, Any]:
        """Get results for an A/B test.
        
        Args:
            test_id: A/B test identifier
        
        Returns:
            Test results with statistical analysis
        """
        ab_test = self._get_ab_test(test_id)
        if not ab_test:
            return {}
        
        results = {
            "test_id": test_id,
            "domain_id": ab_test["domain_id"],
            "status": ab_test["status"],
            "start_date": ab_test["start_date"],
            "end_date": ab_test["end_date"],
            "variants": {},
            "winner": None,
            "statistical_significance": None,
        }
        
        # Calculate metrics for each variant
        for variant in ["variant_a", "variant_b"]:
            metrics = ab_test["metrics"].get(variant, {})
            
            variant_results = {}
            for metric_name in ["accuracy", "latency_ms", "training_loss", "validation_loss"]:
                count_key = f"{metric_name}_count"
                sum_key = f"{metric_name}_sum"
                
                count = metrics.get(count_key, 0)
                sum_value = metrics.get(sum_key, 0)
                
                if count > 0:
                    variant_results[metric_name] = {
                        "average": sum_value / count,
                        "count": count,
                        "total": sum_value,
                    }
            
            results["variants"][variant] = variant_results
        
        # Determine winner based on primary metric (accuracy)
        variant_a_accuracy = results["variants"]["variant_a"].get("accuracy", {}).get("average", 0)
        variant_b_accuracy = results["variants"]["variant_b"].get("accuracy", {}).get("average", 0)
        
        if variant_b_accuracy > variant_a_accuracy:
            results["winner"] = "variant_b"
            results["improvement"] = ((variant_b_accuracy - variant_a_accuracy) / variant_a_accuracy) * 100
        elif variant_a_accuracy > variant_b_accuracy:
            results["winner"] = "variant_a"
            results["improvement"] = 0
        else:
            results["winner"] = "tie"
            results["improvement"] = 0
        
        # Calculate statistical significance (simplified)
        results["statistical_significance"] = self._calculate_significance(
            results["variants"]["variant_a"],
            results["variants"]["variant_b"]
        )
        
        return results
    
    def conclude_ab_test(
        self,
        test_id: str,
        deploy_winner: bool = True
    ) -> Dict[str, Any]:
        """Conclude an A/B test and optionally deploy winner.
        
        Args:
            test_id: A/B test identifier
            deploy_winner: Whether to deploy the winning variant
        
        Returns:
            Conclusion results
        """
        results = self.get_ab_test_results(test_id)
        
        if results["status"] != "active":
            return {"error": "Test is not active"}
        
        # Mark test as concluded
        if self.postgres_dsn:
            self._update_ab_test_status(test_id, "concluded")
        
        conclusion = {
            "test_id": test_id,
            "concluded_at": datetime.now().isoformat(),
            "winner": results["winner"],
            "improvement": results.get("improvement", 0),
            "statistical_significance": results["statistical_significance"],
            "deployed": False,
        }
        
        # Deploy winner if requested
        if deploy_winner and results["winner"] and results["winner"] != "tie":
            variant_config = results["variants"][results["winner"]]
            
            # Deploy winning variant
            deployment_result = self._deploy_variant(
                results["domain_id"],
                results["winner"],
                variant_config
            )
            
            conclusion["deployed"] = True
            conclusion["deployment"] = deployment_result
        
        logger.info(f"✅ Concluded A/B test {test_id}, winner: {results['winner']}")
        
        return conclusion
    
    def _get_active_ab_test(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Get active A/B test for a domain."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute(
                """
                SELECT test_config FROM ab_tests
                WHERE domain_id = %s AND status = 'active'
                AND end_date > NOW()
                ORDER BY start_date DESC
                LIMIT 1
                """,
                (domain_id,)
            )
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                return row[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get active A/B test: {e}")
            return None
    
    def _get_ab_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get A/B test by ID."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT test_config FROM ab_tests WHERE test_id = %s",
                (test_id,)
            )
            
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if row:
                return row[0]
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get A/B test: {e}")
            return None
    
    def _save_ab_test(self, ab_test: Dict[str, Any]):
        """Save A/B test to PostgreSQL."""
        try:
            import psycopg2
            from psycopg2.extras import Json
            
            # Create table if not exists
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute("""
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
            """)
            
            # Insert test
            cursor.execute(
                """
                INSERT INTO ab_tests (test_id, domain_id, test_config, start_date, end_date)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    ab_test["test_id"],
                    ab_test["domain_id"],
                    Json(ab_test),
                    ab_test["start_date"],
                    ab_test["end_date"],
                )
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save A/B test: {e}")
            raise
    
    def _update_ab_test_metrics(self, ab_test: Dict[str, Any]):
        """Update A/B test metrics in PostgreSQL."""
        try:
            import psycopg2
            from psycopg2.extras import Json
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE ab_tests SET test_config = %s WHERE test_id = %s",
                (Json(ab_test), ab_test["test_id"])
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update A/B test metrics: {e}")
    
    def _update_ab_test_status(self, test_id: str, status: str):
        """Update A/B test status."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE ab_tests SET status = %s WHERE test_id = %s",
                (status, test_id)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update A/B test status: {e}")
    
    def _track_routing(self, test_id: str, variant: str, request_id: str):
        """Track routing decision for A/B test."""
        # Store in Redis for fast access
        # Implementation would go here
        pass
    
    def _calculate_significance(
        self,
        variant_a_metrics: Dict[str, Any],
        variant_b_metrics: Dict[str, Any]
    ) -> float:
        """Calculate statistical significance (simplified t-test)."""
        # Simplified statistical significance calculation
        # In production, would use proper statistical tests
        
        a_accuracy = variant_a_metrics.get("accuracy", {}).get("average", 0)
        b_accuracy = variant_b_metrics.get("accuracy", {}).get("average", 0)
        
        a_count = variant_a_metrics.get("accuracy", {}).get("count", 0)
        b_count = variant_b_metrics.get("accuracy", {}).get("count", 0)
        
        if a_count < 30 or b_count < 30:
            return 0.0  # Not enough samples
        
        # Simplified: if difference is > 5% and both have > 30 samples, consider significant
        diff = abs(b_accuracy - a_accuracy)
        if diff > 0.05:
            return 0.95  # High confidence
        elif diff > 0.02:
            return 0.80  # Medium confidence
        else:
            return 0.50  # Low confidence
    
    def _deploy_variant(
        self,
        domain_id: str,
        variant: str,
        variant_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deploy a variant as the production model."""
        from .auto_deploy import AutoDeploymentTrigger
        
        trigger = AutoDeploymentTrigger(
            postgres_dsn=self.postgres_dsn,
            redis_url=self.redis_url
        )
        
        # Extract metrics from variant config
        metrics = {
            "accuracy": variant_config.get("accuracy", {}).get("average", 0),
            "latency_ms": variant_config.get("latency_ms", {}).get("average", 0),
            "training_loss": variant_config.get("training_loss", {}).get("average", 0),
            "validation_loss": variant_config.get("validation_loss", {}).get("average", 0),
        }
        
        # Deploy
        result = trigger.check_and_deploy(
            domain_id=domain_id,
            training_run_id=f"ab_test_{variant}",
            metrics=metrics,
            model_path=variant_config.get("model_path", "")
        )
        
        return result

