"""Cross-domain pattern transfer learning.

This module implements pattern transfer learning, adapting patterns from one domain
to another, few-shot learning for new domains, and domain adaptation techniques.

Domain-aware enhancements:
- Integration with Phase 4 routing optimizer for domain similarity
- Domain-aware pattern adaptation using domain configs
- Enhanced similarity calculation with domain keywords
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import json
import httpx

logger = logging.getLogger(__name__)


class PatternTransferLearner:
    """Learns to transfer patterns across domains.
    
    Domain-aware enhancements:
    - Uses Phase 4 routing optimizer for domain similarity
    - Domain-aware pattern adaptation
    - Enhanced similarity with domain configs
    """
    
    def __init__(self, adaptation_rate: float = 0.5, localai_url: Optional[str] = None):
        """Initialize pattern transfer learner.
        
        Args:
            adaptation_rate: Rate of adaptation for new domains (0.0 to 1.0)
            localai_url: LocalAI URL for domain config fetching (optional)
        """
        self.adaptation_rate = adaptation_rate
        self.domain_patterns = {}
        self.transfer_rules = {}
        self.domain_similarities = {}
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_configs = {}  # domain_id -> domain config
        
        # Phase 4 routing optimizer integration
        try:
            from .routing_optimizer import RoutingOptimizer
            self.routing_optimizer = RoutingOptimizer()
        except ImportError:
            self.routing_optimizer = None
            logger.warning("RoutingOptimizer not available, using basic similarity")
    
    def _load_domain_config(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Load domain configuration from cache or LocalAI.
        
        Args:
            domain_id: Domain identifier
            
        Returns:
            Domain configuration dict or None if not found
        """
        # Check cache first
        if domain_id in self.domain_configs:
            return self.domain_configs[domain_id]
        
        # Try to load from LocalAI
        try:
            response = httpx.get(
                f"{self.localai_url}/v1/domains",
                timeout=5.0
            )
            if response.status_code == 200:
                domains = response.json()
                if isinstance(domains, list):
                    for domain in domains:
                        if domain.get("id") == domain_id or domain.get("name") == domain_id:
                            self.domain_configs[domain_id] = domain
                            return domain
                elif isinstance(domains, dict):
                    # Single domain response
                    if domains.get("id") == domain_id or domains.get("name") == domain_id:
                        self.domain_configs[domain_id] = domains
                        return domains
        except Exception as e:
            logger.debug(f"Failed to load domain config for {domain_id}: {e}")
        
        return None
    
    def transfer_patterns(
        self,
        source_domain: str,
        target_domain: str,
        source_patterns: Dict[str, Any],
        target_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Transfer patterns from source domain to target domain.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            source_patterns: Patterns from source domain
            target_context: Optional context for target domain
        
        Returns:
            Dictionary with transferred patterns:
            - transferred_patterns: Adapted patterns for target domain
            - transfer_confidence: Confidence in transfer
            - adaptation_changes: Changes made during adaptation
        """
        logger.info(f"Transferring patterns from {source_domain} to {target_domain}...")
        
        # Calculate domain similarity
        similarity = self.calculate_domain_similarity(source_domain, target_domain, target_context)
        
        # Adapt patterns
        adapted_patterns = self.adapt_patterns(source_patterns, source_domain, target_domain, similarity)
        
        # Calculate transfer confidence
        confidence = self.calculate_transfer_confidence(source_patterns, adapted_patterns, similarity)
        
        # Track transfer rules
        transfer_key = f"{source_domain}->{target_domain}"
        self.transfer_rules[transfer_key] = {
            "similarity": similarity,
            "confidence": confidence,
            "adaptation_rate": self.adaptation_rate,
        }
        
        result = {
            "transferred_patterns": adapted_patterns,
            "transfer_confidence": confidence,
            "domain_similarity": similarity,
            "adaptation_changes": self._get_adaptation_changes(source_patterns, adapted_patterns),
            "source_domain": source_domain,
            "target_domain": target_domain,
        }
        
        logger.info(f"Pattern transfer complete: confidence={confidence:.2f}, similarity={similarity:.2f}")
        
        return result
    
    def calculate_domain_similarity(
        self,
        source_domain: str,
        target_domain: str,
        target_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate similarity between source and target domains.
        
        Args:
            source_domain: Source domain name
            target_domain: Target domain name
            target_context: Optional context for target domain
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Phase 8.4: Use domain configs for enhanced similarity
        source_config = self._load_domain_config(source_domain)
        target_config = self._load_domain_config(target_domain)
        
        if source_config and target_config:
            # Use domain keywords for similarity
            source_keywords = set(source_config.get("keywords", []))
            target_keywords = set(target_config.get("keywords", []))
            
            if source_keywords and target_keywords:
                # Jaccard similarity
                intersection = len(source_keywords & target_keywords)
                union = len(source_keywords | target_keywords)
                similarity = intersection / union if union > 0 else 0.0
                
                # Boost if same layer/team
                if source_config.get("layer") == target_config.get("layer"):
                    similarity = min(1.0, similarity + 0.2)
                if source_config.get("team") == target_config.get("team"):
                    similarity = min(1.0, similarity + 0.1)
                
                return similarity
        
        # Fallback to keyword matching (legacy)
        domain_keywords = {
            "financial": ["amount", "price", "cost", "revenue", "payment", "transaction"],
            "customer": ["customer", "client", "user", "person", "contact"],
            "product": ["product", "item", "sku", "catalog", "inventory"],
            "order": ["order", "order_item", "purchase", "cart"],
            "logistics": ["shipment", "delivery", "warehouse", "location"],
        }
        
        source_keywords = set(domain_keywords.get(source_domain.lower(), []))
        target_keywords = set(domain_keywords.get(target_domain.lower(), []))
        
        # Calculate Jaccard similarity
        if len(source_keywords) == 0 and len(target_keywords) == 0:
            return 0.5  # Default similarity for unknown domains
        
        if len(source_keywords) == 0 or len(target_keywords) == 0:
            return 0.3  # Low similarity if one domain is unknown
        
        intersection = len(source_keywords & target_keywords)
        union = len(source_keywords | target_keywords)
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Boost similarity if context provides clues
        if target_context:
            context_text = str(target_context).lower()
            matching_keywords = sum(1 for kw in source_keywords if kw in context_text)
            if matching_keywords > 0:
                similarity = min(1.0, similarity + 0.2)
        
        return similarity
    
    def adapt_patterns(
        self,
        source_patterns: Dict[str, Any],
        source_domain: str,
        target_domain: str,
        similarity: float
    ) -> Dict[str, Any]:
        """Adapt patterns from source domain to target domain.
        
        Args:
            source_patterns: Patterns from source domain
            source_domain: Source domain name
            target_domain: Target domain name
            similarity: Domain similarity score
        
        Returns:
            Adapted patterns for target domain
        """
        adapted_patterns = {}
        
        # Adaptation strategy based on similarity
        if similarity > 0.8:
            # High similarity: minimal adaptation
            adaptation_factor = 0.1
        elif similarity > 0.5:
            # Medium similarity: moderate adaptation
            adaptation_factor = 0.3
        else:
            # Low similarity: significant adaptation
            adaptation_factor = 0.6
        
        # Adapt column patterns
        if "column_patterns" in source_patterns:
            adapted_patterns["column_patterns"] = self._adapt_column_patterns(
                source_patterns["column_patterns"],
                source_domain,
                target_domain,
                adaptation_factor
            )
        
        # Adapt relationship patterns
        if "relationship_patterns" in source_patterns:
            adapted_patterns["relationship_patterns"] = self._adapt_relationship_patterns(
                source_patterns["relationship_patterns"],
                source_domain,
                target_domain,
                adaptation_factor
            )
        
        # Adapt metadata patterns
        if "metrics_patterns" in source_patterns:
            adapted_patterns["metrics_patterns"] = source_patterns["metrics_patterns"]
            # Metrics are more domain-agnostic
        
        return adapted_patterns
    
    def _adapt_column_patterns(
        self,
        column_patterns: Dict[str, Any],
        source_domain: str,
        target_domain: str,
        adaptation_factor: float
    ) -> Dict[str, Any]:
        """Adapt column type patterns."""
        adapted = column_patterns.copy()
        
        # Adjust type distributions based on domain
        if "type_distributions" in adapted:
            type_distributions = adapted["type_distributions"].copy()
            
            # Domain-specific adjustments
            if target_domain == "financial":
                # Boost numeric types
                for type_name in ["decimal", "numeric", "float", "double"]:
                    if type_name in type_distributions:
                        type_distributions[type_name] = int(
                            type_distributions[type_name] * (1.0 + adaptation_factor)
                        )
            
            elif target_domain == "customer":
                # Boost text types
                for type_name in ["varchar", "text", "string"]:
                    if type_name in type_distributions:
                        type_distributions[type_name] = int(
                            type_distributions[type_name] * (1.0 + adaptation_factor)
                        )
            
            adapted["type_distributions"] = type_distributions
        
        return adapted
    
    def _adapt_relationship_patterns(
        self,
        relationship_patterns: Dict[str, Any],
        source_domain: str,
        target_domain: str,
        adaptation_factor: float
    ) -> Dict[str, Any]:
        """Adapt relationship patterns."""
        adapted = relationship_patterns.copy()
        
        # Relationship patterns are generally more transferable
        # Just adjust confidence based on adaptation factor
        if "edge_labels" in adapted:
            edge_labels = adapted["edge_labels"].copy()
            for label in edge_labels:
                edge_labels[label] = int(edge_labels[label] * (1.0 - adaptation_factor * 0.2))
            adapted["edge_labels"] = edge_labels
        
        return adapted
    
    def calculate_transfer_confidence(
        self,
        source_patterns: Dict[str, Any],
        adapted_patterns: Dict[str, Any],
        similarity: float
    ) -> float:
        """Calculate confidence in pattern transfer.
        
        Args:
            source_patterns: Original patterns
            adapted_patterns: Adapted patterns
            similarity: Domain similarity
        
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on domain similarity
        confidence = similarity
        
        # Adjust based on pattern coverage
        source_coverage = len(source_patterns)
        adapted_coverage = len(adapted_patterns)
        
        if source_coverage > 0:
            coverage_ratio = adapted_coverage / source_coverage
            confidence = (confidence + coverage_ratio) / 2.0
        
        # Adjust based on adaptation changes
        changes = self._get_adaptation_changes(source_patterns, adapted_patterns)
        if len(changes) > 0:
            # More changes = lower confidence
            change_factor = min(1.0, len(changes) / 10.0)
            confidence = confidence * (1.0 - change_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    def _get_adaptation_changes(
        self,
        source_patterns: Dict[str, Any],
        adapted_patterns: Dict[str, Any]
    ) -> List[str]:
        """Get list of changes made during adaptation."""
        changes = []
        
        # Compare patterns
        if "column_patterns" in source_patterns and "column_patterns" in adapted_patterns:
            source_types = source_patterns["column_patterns"].get("type_distributions", {})
            adapted_types = adapted_patterns["column_patterns"].get("type_distributions", {})
            
            for type_name in set(list(source_types.keys()) + list(adapted_types.keys())):
                source_count = source_types.get(type_name, 0)
                adapted_count = adapted_types.get(type_name, 0)
                
                if source_count != adapted_count:
                    changes.append(f"Column type {type_name}: {source_count} -> {adapted_count}")
        
        return changes
    
    def few_shot_learning(
        self,
        target_domain: str,
        source_domains: List[str],
        source_patterns: Dict[str, Dict[str, Any]],
        target_examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform few-shot learning for a new domain.
        
        Args:
            target_domain: Target domain name
            source_domains: List of source domain names
            source_patterns: Patterns from source domains (domain -> patterns)
            target_examples: Few examples from target domain
        
        Returns:
            Dictionary with learned patterns for target domain
        """
        logger.info(f"Few-shot learning for domain {target_domain} with {len(target_examples)} examples")
        
        # Find most similar source domain
        best_source = None
        best_similarity = 0.0
        
        for source_domain in source_domains:
            similarity = self.calculate_domain_similarity(source_domain, target_domain)
            if similarity > best_similarity:
                best_similarity = similarity
                best_source = source_domain
        
        if best_source is None:
            return {"error": "No suitable source domain found"}
        
        # Transfer patterns from best source
        transferred = self.transfer_patterns(
            best_source,
            target_domain,
            source_patterns[best_source]
        )
        
        # Adapt based on target examples
        adapted_patterns = self._adapt_from_examples(
            transferred["transferred_patterns"],
            target_examples
        )
        
        return {
            "target_domain": target_domain,
            "source_domain": best_source,
            "similarity": best_similarity,
            "learned_patterns": adapted_patterns,
            "few_shot_examples": len(target_examples),
        }
    
    def _adapt_from_examples(
        self,
        base_patterns: Dict[str, Any],
        examples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Adapt patterns based on examples."""
        adapted = base_patterns.copy()
        
        # Extract patterns from examples
        example_patterns = {
            "column_types": set(),
            "relationships": set(),
        }
        
        for example in examples:
            if "columns" in example:
                for col in example["columns"]:
                    if "type" in col:
                        example_patterns["column_types"].add(col["type"])
            
            if "relationships" in example:
                for rel in example["relationships"]:
                    example_patterns["relationships"].add(rel.get("type", "unknown"))
        
        # Merge example patterns into base patterns
        if "column_patterns" in adapted:
            type_distributions = adapted["column_patterns"].get("type_distributions", {})
            for col_type in example_patterns["column_types"]:
                if col_type not in type_distributions:
                    type_distributions[col_type] = 1  # Add new type from examples
                else:
                    type_distributions[col_type] += 1  # Boost existing type
            adapted["column_patterns"]["type_distributions"] = type_distributions
        
        return adapted
    
    def get_transfer_rules(self) -> Dict[str, Any]:
        """Get learned transfer rules."""
        return self.transfer_rules

