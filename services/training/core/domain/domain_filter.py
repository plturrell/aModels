"""Domain-specific training data filtering with differential privacy.

This module provides domain-aware filtering of training data with
differential privacy protection for sensitive information.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import httpx
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PrivacyConfig:
    """Differential privacy configuration."""
    epsilon: float = 1.0  # Privacy budget (ε)
    delta: float = 1e-5   # Privacy parameter (δ)
    noise_scale: float = 0.1  # Noise scale for Laplacian mechanism
    sensitivity: float = 1.0  # Sensitivity of the query
    max_queries: int = 100  # Maximum number of queries per domain
    privacy_level: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        """Set privacy parameters based on level."""
        if self.privacy_level == "low":
            self.epsilon = 2.0
            self.delta = 1e-4
            self.noise_scale = 0.05
        elif self.privacy_level == "high":
            self.epsilon = 0.5
            self.delta = 1e-6
            self.noise_scale = 0.2
        else:  # medium
            self.epsilon = 1.0
            self.delta = 1e-5
            self.noise_scale = 0.1


class DomainFilter:
    """Filter training data by domain with differential privacy."""
    
    def __init__(
        self,
        localai_url: Optional[str] = None,
        privacy_config: Optional[PrivacyConfig] = None
    ):
        """Initialize domain filter.
        
        Args:
            localai_url: URL of LocalAI service for domain configs
            privacy_config: Differential privacy configuration
        """
        self.localai_url = localai_url or os.getenv(
            "LOCALAI_URL", "http://localai:8080"
        )
        self.privacy_config = privacy_config or PrivacyConfig()
        self.domain_configs: Dict[str, Dict[str, Any]] = {}
        self.query_count: Dict[str, int] = {}  # Track queries per domain
        self._load_domains()
    
    def _load_domains(self):
        """Load domain configurations from LocalAI."""
        try:
            client = httpx.Client(timeout=10.0)
            response = client.get(f"{self.localai_url}/v1/domains")
            response.raise_for_status()
            
            data = response.json()
            for domain_info in data.get("data", []):
                domain_id = domain_info.get("id")
                if domain_id and domain_info.get("config"):
                    self.domain_configs[domain_id] = {
                        "agent_id": domain_info.get("agent_id", ""),
                        "keywords": domain_info.get("keywords", []),
                        "tags": domain_info.get("tags", []),
                        "name": domain_info.get("name", domain_id),
                    }
                    self.query_count[domain_id] = 0
            
            logger.info(f"✅ Loaded {len(self.domain_configs)} domains for filtering")
        except Exception as e:
            logger.warning(f"⚠️  Failed to load domains: {e}")
            self.domain_configs = {}
    
    def _check_privacy_budget(self, domain_id: str) -> bool:
        """Check if privacy budget allows another query."""
        if domain_id not in self.query_count:
            return False
        
        cost = 1.0 / self.privacy_config.max_queries
        used_budget = (self.query_count[domain_id] * cost) / self.privacy_config.epsilon
        
        return used_budget < 1.0
    
    def _consume_privacy_budget(self, domain_id: str):
        """Consume privacy budget for a query."""
        if domain_id in self.query_count:
            self.query_count[domain_id] += 1
    
    def _add_noise(self, value: float) -> float:
        """Add Laplacian noise for differential privacy."""
        # Laplacian noise: Lap(sensitivity / epsilon)
        scale = self.privacy_config.sensitivity / self.privacy_config.epsilon
        noise = np.random.laplace(0, scale)
        return value + noise
    
    def _add_noise_to_distribution(self, distribution: Dict[str, float]) -> Dict[str, float]:
        """Add noise to a distribution while preserving sum = 1.0."""
        noisy_dist = {}
        total = 0.0
        
        # Add noise to each value
        for key, value in distribution.items():
            noisy_value = max(0.0, self._add_noise(value))
            noisy_dist[key] = noisy_value
            total += noisy_value
        
        # Normalize to preserve probability distribution
        if total > 0:
            for key in noisy_dist:
                noisy_dist[key] /= total
        else:
            # If all values are negative, use uniform distribution
            n = len(noisy_dist)
            for key in noisy_dist:
                noisy_dist[key] = 1.0 / n
        
        return noisy_dist
    
    def filter_by_domain(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        domain_id: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter nodes and edges by domain with differential privacy.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            domain_id: Optional domain ID to filter by (if None, auto-detect)
        
        Returns:
            Tuple of (filtered_nodes, filtered_edges) with privacy protection
        """
        if not domain_id:
            # Auto-detect domain from nodes
            domain_id = self._detect_domain_from_nodes(nodes)
        
        if not domain_id or domain_id not in self.domain_configs:
            logger.warning(f"⚠️  Domain '{domain_id}' not found, returning empty results")
            return [], []
        
        # Check privacy budget
        if not self._check_privacy_budget(domain_id):
            logger.warning(
                f"⚠️  Privacy budget exhausted for domain '{domain_id}', "
                "returning empty results"
            )
            return [], []
        
        # Filter nodes by domain
        filtered_nodes = []
        for node in nodes:
            node_domain = node.get("properties", {}).get("domain")
            node_agent_id = node.get("properties", {}).get("agent_id")
            
            domain_config = self.domain_configs[domain_id]
            if node_domain == domain_id or node_agent_id == domain_config.get("agent_id"):
                # Apply differential privacy to node properties
                private_node = self._apply_privacy_to_node(node, domain_id)
                filtered_nodes.append(private_node)
        
        # Filter edges by domain
        filtered_edges = []
        for edge in edges:
            edge_domain = edge.get("properties", {}).get("domain")
            edge_agent_id = edge.get("properties", {}).get("agent_id")
            
            domain_config = self.domain_configs[domain_id]
            if edge_domain == domain_id or edge_agent_id == domain_config.get("agent_id"):
                # Apply differential privacy to edge properties
                private_edge = self._apply_privacy_to_edge(edge, domain_id)
                filtered_edges.append(private_edge)
        
        # Consume privacy budget
        self._consume_privacy_budget(domain_id)
        
        logger.info(
            f"✅ Filtered {len(filtered_nodes)} nodes, {len(filtered_edges)} edges "
            f"for domain '{domain_id}' (privacy budget: {self.query_count[domain_id]}/{self.privacy_config.max_queries})"
        )
        
        return filtered_nodes, filtered_edges
    
    def _detect_domain_from_nodes(self, nodes: List[Dict[str, Any]]) -> Optional[str]:
        """Detect the most common domain from nodes."""
        domain_counts = {}
        
        for node in nodes:
            node_domain = node.get("properties", {}).get("domain")
            if node_domain:
                domain_counts[node_domain] = domain_counts.get(node_domain, 0) + 1
        
        if not domain_counts:
            return None
        
        # Return most common domain
        return max(domain_counts.items(), key=lambda x: x[1])[0]
    
    def _apply_privacy_to_node(self, node: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        """Apply differential privacy to node properties."""
        private_node = node.copy()
        
        # Preserve structural information
        if "properties" in private_node:
            props = private_node["properties"].copy()
            
            # Add noise to numeric properties
            for key, value in props.items():
                if isinstance(value, (int, float)):
                    props[key] = self._add_noise(float(value))
                elif isinstance(value, dict) and "distribution" in str(key).lower():
                    # Add noise to distributions
                    props[key] = self._add_noise_to_distribution(value)
            
            private_node["properties"] = props
        
        return private_node
    
    def _apply_privacy_to_edge(self, edge: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        """Apply differential privacy to edge properties."""
        private_edge = edge.copy()
        
        # Preserve structural information
        if "properties" in private_edge:
            props = private_edge["properties"].copy()
            
            # Add noise to numeric properties
            for key, value in props.items():
                if isinstance(value, (int, float)):
                    props[key] = self._add_noise(float(value))
            
            private_edge["properties"] = props
        
        return private_edge
    
    def filter_features_by_domain(
        self,
        features: List[Dict[str, Any]],
        domain_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Filter training features by domain with differential privacy.
        
        Args:
            features: List of training features
            domain_id: Optional domain ID to filter by
        
        Returns:
            Filtered features with privacy protection
        """
        if not domain_id:
            # Try to detect from features
            domain_id = self._detect_domain_from_features(features)
        
        if not domain_id or domain_id not in self.domain_configs:
            logger.warning(f"⚠️  Domain '{domain_id}' not found")
            return []
        
        # Check privacy budget
        if not self._check_privacy_budget(domain_id):
            logger.warning(f"⚠️  Privacy budget exhausted for domain '{domain_id}'")
            return []
        
        filtered_features = []
        domain_config = self.domain_configs[domain_id]
        keywords = set(kw.lower() for kw in domain_config.get("keywords", []))
        tags = set(tag.lower() for tag in domain_config.get("tags", []))
        
        for feature in features:
            feature_type = feature.get("type", "").lower()
            feature_data = feature.get("data", {})
            
            # Check if feature matches domain
            matches = False
            if any(kw in feature_type for kw in keywords):
                matches = True
            elif any(tag in feature_type for tag in tags):
                matches = True
            elif isinstance(feature_data, dict):
                # Check feature data for domain keywords
                data_str = json.dumps(feature_data).lower()
                if any(kw in data_str for kw in keywords):
                    matches = True
            
            if matches:
                # Apply differential privacy to feature data
                private_feature = self._apply_privacy_to_feature(feature, domain_id)
                filtered_features.append(private_feature)
        
        # Consume privacy budget
        self._consume_privacy_budget(domain_id)
        
        logger.info(
            f"✅ Filtered {len(filtered_features)} features for domain '{domain_id}' "
            f"(privacy budget: {self.query_count[domain_id]}/{self.privacy_config.max_queries})"
        )
        
        return filtered_features
    
    def _detect_domain_from_features(self, features: List[Dict[str, Any]]) -> Optional[str]:
        """Detect domain from feature keywords."""
        feature_text = json.dumps(features).lower()
        
        best_domain = None
        best_score = 0
        
        for domain_id, config in self.domain_configs.items():
            keywords = set(kw.lower() for kw in config.get("keywords", []))
            tags = set(tag.lower() for tag in config.get("tags", []))
            
            score = sum(1 for kw in keywords if kw in feature_text)
            score += sum(1 for tag in tags if tag in feature_text)
            
            if score > best_score:
                best_score = score
                best_domain = domain_id
        
        return best_domain
    
    def _apply_privacy_to_feature(self, feature: Dict[str, Any], domain_id: str) -> Dict[str, Any]:
        """Apply differential privacy to feature data."""
        private_feature = feature.copy()
        
        if "data" in private_feature:
            data = private_feature["data"]
            
            if isinstance(data, dict):
                # Add noise to numeric values in dictionaries
                private_data = {}
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        private_data[key] = self._add_noise(float(value))
                    elif isinstance(value, dict):
                        # Recursively apply privacy
                        private_data[key] = self._apply_privacy_to_feature(
                            {"data": value}, domain_id
                        )["data"]
                    else:
                        private_data[key] = value
                private_feature["data"] = private_data
            elif isinstance(data, (list, tuple)):
                # Add noise to numeric lists
                private_data = [
                    self._add_noise(float(v)) if isinstance(v, (int, float)) else v
                    for v in data
                ]
                private_feature["data"] = private_data
        
        return private_feature
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy budget statistics."""
        return {
            "domains": {
                domain_id: {
                    "queries_used": count,
                    "queries_remaining": self.privacy_config.max_queries - count,
                    "budget_utilization": (count / self.privacy_config.max_queries) * 100,
                }
                for domain_id, count in self.query_count.items()
            },
            "privacy_config": {
                "epsilon": self.privacy_config.epsilon,
                "delta": self.privacy_config.delta,
                "noise_scale": self.privacy_config.noise_scale,
                "privacy_level": self.privacy_config.privacy_level,
            }
        }

