"""Domain-specific performance metrics collection and analytics.

This module provides comprehensive metrics tracking for domain models,
including performance, usage, and quality metrics.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
import httpx

logger = logging.getLogger(__name__)


class DomainMetricsCollector:
    """Collect and analyze domain-specific performance metrics."""
    
    def __init__(
        self,
        localai_url: Optional[str] = None,
        postgres_dsn: Optional[str] = None
    ):
        """Initialize metrics collector.
        
        Args:
            localai_url: LocalAI service URL
            postgres_dsn: PostgreSQL connection string
        """
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_DSN")
    
    def collect_domain_metrics(
        self,
        domain_id: str,
        time_window_days: int = 7
    ) -> Dict[str, Any]:
        """Collect comprehensive metrics for a domain.
        
        Args:
            domain_id: Domain identifier
            time_window_days: Number of days to look back
        
        Returns:
            Dictionary with aggregated metrics
        """
        logger.info(f"Collecting metrics for domain {domain_id} (last {time_window_days} days)")
        
        metrics = {
            "domain_id": domain_id,
            "time_window_days": time_window_days,
            "collected_at": datetime.now().isoformat(),
            "performance": {},
            "usage": {},
            "quality": {},
            "trends": {},
        }
        
        # Collect from PostgreSQL
        if self.postgres_dsn:
            pg_metrics = self._collect_from_postgres(domain_id, time_window_days)
            metrics.update(pg_metrics)
        
        # Collect from LocalAI (if available)
        localai_metrics = self._collect_from_localai(domain_id)
        metrics["localai"] = localai_metrics
        
        # Calculate trends
        metrics["trends"] = self._calculate_trends(domain_id, time_window_days)
        
        return metrics
    
    def _collect_from_postgres(
        self,
        domain_id: str,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Collect metrics from PostgreSQL."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Get performance metrics from domain_configs
            cursor.execute(
                """
                SELECT 
                    model_version,
                    performance_metrics,
                    updated_at,
                    version
                FROM domain_configs
                WHERE domain_name = %s
                ORDER BY updated_at DESC
                LIMIT 10
                """,
                (domain_id,)
            )
            
            rows = cursor.fetchall()
            
            performance_history = []
            for row in rows:
                version, metrics_json, updated_at, version_num = row
                if metrics_json:
                    performance_history.append({
                        "version": version,
                        "version_num": version_num,
                        "metrics": metrics_json,
                        "updated_at": updated_at.isoformat() if updated_at else None,
                    })
            
            # Aggregate performance metrics
            if performance_history:
                latest = performance_history[0]
                avg_metrics = self._aggregate_metrics(performance_history)
                
                cursor.close()
                conn.close()
                
                return {
                    "performance": {
                        "latest": latest["metrics"],
                        "average": avg_metrics,
                        "history": performance_history,
                    }
                }
            
            cursor.close()
            conn.close()
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to collect from PostgreSQL: {e}")
            return {}
    
    def _collect_from_localai(self, domain_id: str) -> Dict[str, Any]:
        """Collect metrics from LocalAI service."""
        try:
            client = httpx.Client(timeout=10.0)
            
            # Get domain info
            response = client.get(f"{self.localai_url}/v1/domains")
            response.raise_for_status()
            
            data = response.json()
            for domain_info in data.get("data", []):
                if domain_info.get("id") == domain_id:
                    return {
                        "loaded": domain_info.get("loaded", False),
                        "name": domain_info.get("name"),
                        "config": domain_info.get("config", {}),
                    }
            
            return {}
            
        except Exception as e:
            logger.warning(f"Failed to collect from LocalAI: {e}")
            return {}
    
    def _aggregate_metrics(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from history."""
        if not history:
            return {}
        
        metrics_sum = defaultdict(float)
        metrics_count = defaultdict(int)
        
        for entry in history:
            metrics = entry.get("metrics", {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_sum[key] += value
                    metrics_count[key] += 1
        
        aggregated = {}
        for key in metrics_sum:
            if metrics_count[key] > 0:
                aggregated[key] = metrics_sum[key] / metrics_count[key]
        
        return aggregated
    
    def _calculate_trends(
        self,
        domain_id: str,
        time_window_days: int
    ) -> Dict[str, Any]:
        """Calculate trends over time."""
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Get metrics over time
            since = datetime.now() - timedelta(days=time_window_days)
            
            cursor.execute(
                """
                SELECT 
                    updated_at,
                    performance_metrics
                FROM domain_configs
                WHERE domain_id = %s AND updated_at >= %s
                ORDER BY updated_at ASC
                """,
                (domain_id, since)
            )
            
            rows = cursor.fetchall()
            
            trends = {
                "accuracy": [],
                "latency_ms": [],
                "training_loss": [],
                "validation_loss": [],
            }
            
            for updated_at, metrics_json in rows:
                if metrics_json:
                    timestamp = updated_at.isoformat() if updated_at else None
                    for metric_name in trends:
                        if metric_name in metrics_json:
                            trends[metric_name].append({
                                "timestamp": timestamp,
                                "value": metrics_json[metric_name],
                            })
            
            # Calculate trend direction
            trend_direction = {}
            for metric_name, values in trends.items():
                if len(values) >= 2:
                    first = values[0]["value"]
                    last = values[-1]["value"]
                    if last > first:
                        trend_direction[metric_name] = "improving"
                    elif last < first:
                        trend_direction[metric_name] = "degrading"
                    else:
                        trend_direction[metric_name] = "stable"
            
            trends["direction"] = trend_direction
            
            cursor.close()
            conn.close()
            
            return trends
            
        except Exception as e:
            logger.warning(f"Failed to calculate trends: {e}")
            return {}
    
    def get_domain_comparison(
        self,
        domain_ids: List[str]
    ) -> Dict[str, Any]:
        """Compare metrics across multiple domains.
        
        Args:
            domain_ids: List of domain identifiers to compare
        
        Returns:
            Comparison metrics
        """
        logger.info(f"Comparing metrics for {len(domain_ids)} domains")
        
        comparison = {
            "domains": {},
            "rankings": {},
            "comparison_at": datetime.now().isoformat(),
        }
        
        all_metrics = {}
        for domain_id in domain_ids:
            metrics = self.collect_domain_metrics(domain_id, time_window_days=7)
            comparison["domains"][domain_id] = metrics
            all_metrics[domain_id] = metrics.get("performance", {}).get("latest", {})
        
        # Rank domains by various metrics
        if all_metrics:
            # Rank by accuracy
            accuracy_ranking = sorted(
                all_metrics.items(),
                key=lambda x: x[1].get("accuracy", 0),
                reverse=True
            )
            comparison["rankings"]["accuracy"] = [
                {"domain_id": domain_id, "value": metrics.get("accuracy", 0)}
                for domain_id, metrics in accuracy_ranking
            ]
            
            # Rank by latency (lower is better)
            latency_ranking = sorted(
                all_metrics.items(),
                key=lambda x: x[1].get("latency_ms", float("inf")),
            )
            comparison["rankings"]["latency"] = [
                {"domain_id": domain_id, "value": metrics.get("latency_ms", float("inf"))}
                for domain_id, metrics in latency_ranking
            ]
        
        return comparison
    
    def export_metrics_dashboard(
        self,
        output_path: str,
        domain_ids: Optional[List[str]] = None
    ):
        """Export metrics for dashboard visualization.
        
        Args:
            output_path: Path to save dashboard data
            domain_ids: Optional list of domains (if None, exports all)
        """
        logger.info(f"Exporting metrics dashboard to {output_path}")
        
        if domain_ids is None:
            # Get all domains from LocalAI
            domain_ids = self._get_all_domain_ids()
        
        dashboard_data = {
            "generated_at": datetime.now().isoformat(),
            "domains": {},
        }
        
        for domain_id in domain_ids:
            metrics = self.collect_domain_metrics(domain_id, time_window_days=30)
            dashboard_data["domains"][domain_id] = metrics
        
        # Add comparison
        if len(domain_ids) > 1:
            dashboard_data["comparison"] = self.get_domain_comparison(domain_ids)
        
        # Save to file
        with open(output_path, "w") as f:
            json.dump(dashboard_data, f, indent=2)
        
        logger.info(f"✅ Exported dashboard data for {len(domain_ids)} domains")
    
    def _get_all_domain_ids(self) -> List[str]:
        """Get all domain IDs from LocalAI."""
        try:
            client = httpx.Client(timeout=10.0)
            response = client.get(f"{self.localai_url}/v1/domains")
            response.raise_for_status()
            
            data = response.json()
            return [domain_info.get("id") for domain_info in data.get("data", [])]
        except Exception as e:
            logger.warning(f"Failed to get domain IDs: {e}")
            return []

