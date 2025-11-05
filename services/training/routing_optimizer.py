"""Advanced routing optimization with learning.

This module provides learning-based routing optimization that improves
domain selection over time based on performance feedback.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class RoutingOptimizer:
    """Optimize domain routing based on performance feedback."""
    
    def __init__(
        self,
        postgres_dsn: Optional[str] = None,
        learning_rate: float = 0.1
    ):
        """Initialize routing optimizer.
        
        Args:
            postgres_dsn: PostgreSQL connection string
            learning_rate: Learning rate for optimization (0.0-1.0)
        """
        self.postgres_dsn = postgres_dsn or os.getenv("POSTGRES_DSN")
        self.learning_rate = learning_rate
        
        # Routing weights (domain -> score)
        self.routing_weights: Dict[str, float] = {}
        
        # Performance history
        self.performance_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Load existing weights
        self._load_routing_weights()
    
    def record_routing_decision(
        self,
        domain_id: str,
        query: str,
        decision_confidence: float,
        actual_metrics: Dict[str, Any]
    ):
        """Record a routing decision and its outcome.
        
        Args:
            domain_id: Domain that was selected
            query: Query text
            decision_confidence: Confidence in routing decision
            actual_metrics: Actual performance metrics (latency, accuracy, etc.)
        """
        # Store performance history
        self.performance_history[domain_id].append({
            "query": query,
            "confidence": decision_confidence,
            "metrics": actual_metrics,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Keep only last 1000 records per domain
        if len(self.performance_history[domain_id]) > 1000:
            self.performance_history[domain_id] = self.performance_history[domain_id][-1000:]
        
        # Update routing weights based on performance
        self._update_routing_weights(domain_id, actual_metrics)
        
        # Save weights periodically
        self._save_routing_weights()
    
    def optimize_routing_weights(
        self,
        domain_ids: List[str]
    ) -> Dict[str, float]:
        """Optimize routing weights for domains based on performance history.
        
        Args:
            domain_ids: List of domain identifiers
        
        Returns:
            Optimized routing weights
        """
        for domain_id in domain_ids:
            if domain_id not in self.performance_history:
                continue
            
            history = self.performance_history[domain_id]
            if len(history) < 10:
                continue  # Need at least 10 samples
            
            # Calculate average performance metrics
            avg_accuracy = np.mean([h["metrics"].get("accuracy", 0) for h in history])
            avg_latency = np.mean([h["metrics"].get("latency_ms", 0) for h in history])
            
            # Normalize metrics (0-1 scale)
            normalized_accuracy = avg_accuracy  # Already 0-1
            normalized_latency = 1.0 / (1.0 + avg_latency / 1000.0)  # Inverse latency
            
            # Combined score (weighted)
            performance_score = (normalized_accuracy * 0.7) + (normalized_latency * 0.3)
            
            # Update routing weight
            current_weight = self.routing_weights.get(domain_id, 0.5)
            new_weight = current_weight + self.learning_rate * (performance_score - current_weight)
            
            self.routing_weights[domain_id] = max(0.0, min(1.0, new_weight))
        
        logger.info(f"✅ Optimized routing weights for {len(domain_ids)} domains")
        
        return self.routing_weights.copy()
    
    def get_optimal_domain(
        self,
        candidate_domains: List[str],
        query: str,
        base_scores: Dict[str, float]
    ) -> str:
        """Get optimal domain using learned routing weights.
        
        Args:
            candidate_domains: List of candidate domain IDs
            query: Query text
            base_scores: Base routing scores from intelligent router
        
        Returns:
            Optimal domain ID
        """
        if not candidate_domains:
            return ""
        
        # Combine base scores with learned weights
        combined_scores = {}
        for domain_id in candidate_domains:
            base_score = base_scores.get(domain_id, 0.5)
            learned_weight = self.routing_weights.get(domain_id, 0.5)
            
            # Weighted combination
            combined_score = (base_score * 0.6) + (learned_weight * 0.4)
            combined_scores[domain_id] = combined_score
        
        # Select domain with highest combined score
        optimal_domain = max(combined_scores.items(), key=lambda x: x[1])[0]
        
        return optimal_domain
    
    def _update_routing_weights(
        self,
        domain_id: str,
        metrics: Dict[str, Any]
    ):
        """Update routing weights based on performance."""
        if domain_id not in self.routing_weights:
            self.routing_weights[domain_id] = 0.5  # Initial weight
        
        # Calculate performance score
        accuracy = metrics.get("accuracy", 0.5)
        latency_ms = metrics.get("latency_ms", 500)
        
        # Normalize
        normalized_accuracy = accuracy
        normalized_latency = 1.0 / (1.0 + latency_ms / 1000.0)
        
        performance_score = (normalized_accuracy * 0.7) + (normalized_latency * 0.3)
        
        # Update weight with learning rate
        current_weight = self.routing_weights[domain_id]
        new_weight = current_weight + self.learning_rate * (performance_score - current_weight)
        
        self.routing_weights[domain_id] = max(0.0, min(1.0, new_weight))
    
    def _load_routing_weights(self):
        """Load routing weights from PostgreSQL."""
        if not self.postgres_dsn:
            return
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS routing_weights (
                    domain_id VARCHAR(255) PRIMARY KEY,
                    weight FLOAT NOT NULL,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
            
            # Load weights
            cursor.execute("SELECT domain_id, weight FROM routing_weights")
            rows = cursor.fetchall()
            
            for domain_id, weight in rows:
                self.routing_weights[domain_id] = weight
            
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Loaded {len(self.routing_weights)} routing weights")
            
        except Exception as e:
            logger.warning(f"Failed to load routing weights: {e}")
    
    def _save_routing_weights(self):
        """Save routing weights to PostgreSQL."""
        if not self.postgres_dsn:
            return
        
        try:
            import psycopg2
            
            conn = psycopg2.connect(self.postgres_dsn)
            cursor = conn.cursor()
            
            # Save/update weights
            for domain_id, weight in self.routing_weights.items():
                cursor.execute("""
                    INSERT INTO routing_weights (domain_id, weight, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (domain_id)
                    DO UPDATE SET weight = EXCLUDED.weight, updated_at = NOW()
                """, (domain_id, weight))
            
            conn.commit()
            cursor.close()
            conn.close()
            
        except Exception as e:
            logger.warning(f"Failed to save routing weights: {e}")
    
    def get_routing_analytics(
        self,
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get routing analytics.
        
        Args:
            domain_id: Optional domain ID (if None, returns all domains)
        
        Returns:
            Routing analytics
        """
        analytics = {
            "domains": {},
            "overall": {
                "total_decisions": 0,
                "average_confidence": 0.0,
            },
        }
        
        if domain_id:
            domains = [domain_id]
        else:
            domains = list(self.performance_history.keys())
        
        total_confidence = 0.0
        total_count = 0
        
        for did in domains:
            history = self.performance_history.get(did, [])
            weight = self.routing_weights.get(did, 0.5)
            
            if history:
                avg_accuracy = np.mean([h["metrics"].get("accuracy", 0) for h in history])
                avg_latency = np.mean([h["metrics"].get("latency_ms", 0) for h in history])
                avg_confidence = np.mean([h["confidence"] for h in history])
                
                analytics["domains"][did] = {
                    "routing_weight": weight,
                    "decision_count": len(history),
                    "average_accuracy": float(avg_accuracy),
                    "average_latency": float(avg_latency),
                    "average_confidence": float(avg_confidence),
                }
                
                total_confidence += avg_confidence * len(history)
                total_count += len(history)
        
        if total_count > 0:
            analytics["overall"]["average_confidence"] = total_confidence / total_count
            analytics["overall"]["total_decisions"] = total_count
        
        return analytics

