"""Meta-pattern learning: Learn patterns of patterns.

This module implements hierarchical pattern composition, pattern abstraction,
and cross-domain pattern transfer.

Domain-aware enhancements:
- Domain-aware pattern grouping by layer/team
- Integration with Phase 4 cross-domain learning
- Domain-specific meta-patterns
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import json
import httpx

logger = logging.getLogger(__name__)


class MetaPatternLearner:
    """Learns meta-patterns (patterns of patterns) from learned patterns.
    
    Domain-aware enhancements:
    - Groups patterns by domain layer/team
    - Uses Phase 4 routing optimizer for domain similarity
    - Learns domain-specific meta-patterns
    """
    
    def __init__(self, localai_url: Optional[str] = None):
        """Initialize meta-pattern learner.
        
        Args:
            localai_url: LocalAI URL for domain config fetching (optional)
        """
        self.pattern_taxonomy = {}
        self.pattern_hierarchy = {}
        self.abstract_patterns = {}
        self.cross_domain_patterns = {}
        self.layer_patterns = {}
        self.team_patterns = {}
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_configs = {}  # domain_id -> domain config
    
    def learn_meta_patterns(
        self,
        learned_patterns: Dict[str, Any],
        domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Learn meta-patterns from learned patterns.
        
        Args:
            learned_patterns: Dictionary of learned patterns from PatternLearningEngine
            domains: Optional list of domain names for cross-domain transfer
        
        Returns:
            Dictionary with meta-patterns:
            - pattern_taxonomy: Hierarchical organization of patterns
            - abstract_patterns: Abstracted/generalized patterns
            - cross_domain_patterns: Patterns that transfer across domains
            - pattern_composition: How patterns combine
            - layer_patterns: Patterns grouped by domain layer
            - team_patterns: Patterns grouped by domain team
        """
        logger.info("Learning meta-patterns from learned patterns...")
        
        # Build pattern taxonomy
        taxonomy = self._build_pattern_taxonomy(learned_patterns)
        
        # Learn abstract patterns
        abstract_patterns = self._learn_abstract_patterns(learned_patterns)
        
        # Learn pattern composition
        composition_patterns = self._learn_pattern_composition(learned_patterns)
        
        # NEW: Domain-aware pattern grouping
        layer_patterns = {}
        team_patterns = {}
        if domains:
            layer_patterns = self._learn_layer_meta_patterns(learned_patterns, domains)
            team_patterns = self._learn_team_meta_patterns(learned_patterns, domains)
        
        # Cross-domain pattern transfer (if domains provided)
        cross_domain_patterns = {}
        if domains and len(domains) > 1:
            # Enhanced cross-domain learning with Phase 4 integration
            cross_domain_patterns = self._learn_cross_domain_patterns_enhanced(
                learned_patterns, domains
            )
        
        self.pattern_taxonomy = taxonomy
        self.abstract_patterns = abstract_patterns
        self.cross_domain_patterns = cross_domain_patterns
        self.layer_patterns = layer_patterns
        self.team_patterns = team_patterns
        
        meta_patterns = {
            "pattern_taxonomy": taxonomy,
            "abstract_patterns": abstract_patterns,
            "pattern_composition": composition_patterns,
            "cross_domain_patterns": cross_domain_patterns,
            "layer_patterns": layer_patterns,
            "team_patterns": team_patterns,
            "taxonomy_levels": len(taxonomy.get("levels", [])),
            "abstract_pattern_count": len(abstract_patterns),
        }
        
        logger.info(
            f"Meta-pattern learning complete: {len(taxonomy.get('levels', []))} taxonomy levels, "
            f"{len(abstract_patterns)} abstract patterns, "
            f"{len(layer_patterns)} layer patterns, "
            f"{len(team_patterns)} team patterns"
        )
        
        return meta_patterns
    
    def _build_pattern_taxonomy(
        self,
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build hierarchical pattern taxonomy.
        
        Args:
            learned_patterns: Learned patterns from PatternLearningEngine
        
        Returns:
            Dictionary with pattern taxonomy structure
        """
        taxonomy = {
            "levels": [],
            "root_patterns": [],
            "leaf_patterns": [],
            "pattern_tree": {}
        }
        
        # Level 1: Pattern categories (column, relationship, metrics, sequence)
        level1 = {
            "level": 1,
            "name": "Pattern Categories",
            "patterns": []
        }
        
        for category in ["column_patterns", "relationship_patterns", "metrics_patterns", "sequence_patterns"]:
            if category in learned_patterns:
                level1["patterns"].append({
                    "id": category,
                    "name": category.replace("_", " ").title(),
                    "type": "category",
                    "children": []
                })
        
        taxonomy["levels"].append(level1)
        taxonomy["root_patterns"] = [p["id"] for p in level1["patterns"]]
        
        # Level 2: Pattern types within each category
        level2 = {
            "level": 2,
            "name": "Pattern Types",
            "patterns": []
        }
        
        for category_pattern in level1["patterns"]:
            category_key = category_pattern["id"]
            category_data = learned_patterns.get(category_key, {})
            
            # Extract pattern types
            if category_key == "column_patterns":
                pattern_types = category_data.get("type_distributions", {})
                for pattern_type, count in pattern_types.items():
                    level2["patterns"].append({
                        "id": f"{category_key}:{pattern_type}",
                        "name": pattern_type,
                        "type": "pattern_type",
                        "parent": category_key,
                        "frequency": count
                    })
            
            elif category_key == "relationship_patterns":
                edge_labels = category_data.get("edge_labels", {})
                for edge_label, count in edge_labels.items():
                    level2["patterns"].append({
                        "id": f"{category_key}:{edge_label}",
                        "name": edge_label,
                        "type": "pattern_type",
                        "parent": category_key,
                        "frequency": count
                    })
        
        taxonomy["levels"].append(level2)
        
        # Level 3: Specific pattern instances (leaf nodes)
        level3 = {
            "level": 3,
            "name": "Pattern Instances",
            "patterns": []
        }
        
        taxonomy["levels"].append(level3)
        taxonomy["leaf_patterns"] = [p["id"] for p in level3["patterns"]]
        
        # Build pattern tree
        taxonomy["pattern_tree"] = self._build_pattern_tree(taxonomy["levels"])
        
        return taxonomy
    
    def _build_pattern_tree(self, levels: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build tree structure from taxonomy levels.
        
        Args:
            levels: List of taxonomy levels
        
        Returns:
            Tree structure dictionary
        """
        tree = {}
        
        # Start from root patterns
        if levels:
            root_level = levels[0]
            for pattern in root_level.get("patterns", []):
                tree[pattern["id"]] = {
                    "pattern": pattern,
                    "children": []
                }
        
        # Add children for each level
        for i in range(1, len(levels)):
            current_level = levels[i]
            for pattern in current_level.get("patterns", []):
                parent_id = pattern.get("parent")
                if parent_id and parent_id in tree:
                    tree[parent_id]["children"].append(pattern)
        
        return tree
    
    def _learn_abstract_patterns(
        self,
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn abstract patterns by generalizing concrete patterns.
        
        Args:
            learned_patterns: Learned patterns
        
        Returns:
            Dictionary of abstract patterns
        """
        abstract_patterns = {}
        
        # Abstract column type patterns
        column_patterns = learned_patterns.get("column_patterns", {})
        type_distributions = column_patterns.get("type_distributions", {})
        
        # Group similar types (e.g., all numeric types)
        type_groups = {
            "numeric": ["int", "integer", "bigint", "smallint", "decimal", "numeric", "float", "double"],
            "text": ["string", "varchar", "text", "char"],
            "temporal": ["date", "timestamp", "datetime"],
            "boolean": ["boolean", "bool"]
        }
        
        abstract_column_patterns = {}
        for group_name, types in type_groups.items():
            total_count = sum(type_distributions.get(t, 0) for t in types)
            if total_count > 0:
                abstract_column_patterns[group_name] = {
                    "abstract_type": group_name,
                    "concrete_types": types,
                    "frequency": total_count,
                    "abstraction_level": "type_group"
                }
        
        abstract_patterns["column_type_groups"] = abstract_column_patterns
        
        # Abstract relationship patterns
        relationship_patterns = learned_patterns.get("relationship_patterns", {})
        edge_labels = relationship_patterns.get("edge_labels", {})
        
        # Group by relationship category
        relationship_groups = {
            "structural": ["HAS_COLUMN", "HAS_SQL", "HAS_PETRI_NET"],
            "data_flow": ["DATA_FLOW", "PROCESSES_BEFORE", "DEPENDS_ON"],
            "hierarchical": ["CONTAINS", "BELONGS_TO"]
        }
        
        abstract_relationship_patterns = {}
        for group_name, labels in relationship_groups.items():
            total_count = sum(edge_labels.get(l, 0) for l in labels)
            if total_count > 0:
                abstract_relationship_patterns[group_name] = {
                    "abstract_category": group_name,
                    "concrete_labels": labels,
                    "frequency": total_count,
                    "abstraction_level": "relationship_category"
                }
        
        abstract_patterns["relationship_categories"] = abstract_relationship_patterns
        
        return abstract_patterns
    
    def _learn_pattern_composition(
        self,
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn how patterns compose together.
        
        Args:
            learned_patterns: Learned patterns
        
        Returns:
            Dictionary of pattern composition rules
        """
        composition_patterns = {
            "composition_rules": [],
            "pattern_combinations": {},
            "composition_frequency": {}
        }
        
        # Learn common pattern combinations
        column_patterns = learned_patterns.get("column_patterns", {})
        relationship_patterns = learned_patterns.get("relationship_patterns", {})
        
        # Example: Tables with certain column type distributions tend to have certain relationships
        type_distributions = column_patterns.get("type_distributions", {})
        edge_labels = relationship_patterns.get("edge_labels", {})
        
        # Simple composition rule: If table has many numeric columns, likely has data flow relationships
        numeric_types = ["int", "integer", "bigint", "decimal", "numeric", "float", "double"]
        numeric_count = sum(type_distributions.get(t, 0) for t in numeric_types)
        
        if numeric_count > 0:
            composition_patterns["composition_rules"].append({
                "rule_id": "numeric_columns_data_flow",
                "condition": "high_numeric_column_count",
                "implication": "likely_data_flow_relationships",
                "confidence": min(1.0, numeric_count / 100.0)
            })
        
        # Pattern combinations
        composition_patterns["pattern_combinations"] = {
            "column_type_relationship": {
                "description": "Column type patterns combined with relationship patterns",
                "examples": [
                    {
                        "column_pattern": "numeric_dominant",
                        "relationship_pattern": "data_flow_heavy",
                        "frequency": numeric_count
                    }
                ]
            }
        }
        
        return composition_patterns
    
    def _load_domain_config(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Load domain configuration from LocalAI.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Domain configuration or None if not found
        """
        if domain_id in self.domain_configs:
            return self.domain_configs[domain_id]
        
        try:
            response = httpx.get(
                f"{self.localai_url}/v1/domains",
                timeout=5.0
            )
            if response.status_code == 200:
                domains_data = response.json()
                domains = domains_data.get("domains", {})
                
                if domain_id in domains:
                    domain_info = domains[domain_id]
                    config = domain_info.get("config", domain_info)
                    self.domain_configs[domain_id] = config
                    return config
        except Exception as e:
            logger.warning(f"Failed to load domain config for {domain_id}: {e}")
        
        return None
    
    def _group_patterns_by_domain_layer(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Group patterns by domain layer.
        
        Args:
            learned_patterns: Learned patterns
            domains: List of domain identifiers
        
        Returns:
            Dictionary mapping layer -> patterns
        """
        layer_groups = defaultdict(lambda: {"patterns": [], "domains": []})
        
        for domain_id in domains:
            domain_config = self._load_domain_config(domain_id)
            if domain_config:
                layer = domain_config.get("layer", "unknown")
                layer_groups[layer]["domains"].append(domain_id)
                layer_groups[layer]["patterns"].append({
                    "domain_id": domain_id,
                    "domain_name": domain_config.get("name", domain_id),
                    "patterns": learned_patterns  # Would filter by domain in real implementation
                })
        
        return dict(layer_groups)
    
    def _learn_layer_meta_patterns(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Learn meta-patterns specific to domain layers.
        
        Args:
            learned_patterns: Learned patterns
            domains: List of domain identifiers
        
        Returns:
            Dictionary of layer-specific meta-patterns
        """
        layer_groups = self._group_patterns_by_domain_layer(learned_patterns, domains)
        
        layer_patterns = {}
        for layer, group_data in layer_groups.items():
            # Extract common patterns across domains in this layer
            layer_patterns[layer] = {
                "layer": layer,
                "domains": group_data["domains"],
                "common_patterns": self._extract_common_patterns(group_data["patterns"]),
                "layer_specific_abstractions": self._learn_abstract_patterns(learned_patterns)
            }
        
        return layer_patterns
    
    def _learn_team_meta_patterns(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Learn meta-patterns specific to domain teams.
        
        Args:
            learned_patterns: Learned patterns
            domains: List of domain identifiers
        
        Returns:
            Dictionary of team-specific meta-patterns
        """
        team_groups = defaultdict(lambda: {"patterns": [], "domains": []})
        
        for domain_id in domains:
            domain_config = self._load_domain_config(domain_id)
            if domain_config:
                team = domain_config.get("team", "unknown")
                team_groups[team]["domains"].append(domain_id)
                team_groups[team]["patterns"].append({
                    "domain_id": domain_id,
                    "domain_name": domain_config.get("name", domain_id),
                    "patterns": learned_patterns
                })
        
        team_patterns = {}
        for team, group_data in team_groups.items():
            team_patterns[team] = {
                "team": team,
                "domains": group_data["domains"],
                "common_patterns": self._extract_common_patterns(group_data["patterns"]),
                "team_specific_abstractions": self._learn_abstract_patterns(learned_patterns)
            }
        
        return dict(team_patterns)
    
    def _extract_common_patterns(self, pattern_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common patterns across a list of pattern sets.
        
        Args:
            pattern_list: List of pattern dictionaries
        
        Returns:
            Dictionary of common patterns
        """
        # Simplified: return a structure indicating common patterns
        # In real implementation, would compare patterns across domains
        return {
            "common_column_types": [],
            "common_relationships": [],
            "common_metrics": []
        }
    
    def _learn_cross_domain_patterns_enhanced(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Learn patterns that transfer across domains using Phase 4 routing optimizer.
        
        Args:
            learned_patterns: Learned patterns (could be from multiple domains)
            domains: List of domain names
        
        Returns:
            Dictionary of cross-domain patterns with similarity scores
        """
        cross_domain_patterns = {
            "universal_patterns": {},
            "domain_specific_patterns": {},
            "transfer_rules": [],
            "similar_domains": {}
        }
        
        # Use Phase 4 routing optimizer for domain similarity
        try:
            from .routing_optimizer import RoutingOptimizer
            optimizer = RoutingOptimizer()
            
            # Find similar domains for each domain
            for source_domain in domains:
                # Get similar domains (would need to implement find_similar_domains)
                # For now, use basic similarity
                similar_domains = []
                for target_domain in domains:
                    if target_domain != source_domain:
                        # Calculate similarity based on domain configs
                        similarity = self._calculate_domain_similarity(source_domain, target_domain)
                        if similarity > 0.7:
                            similar_domains.append({
                                "domain_id": target_domain,
                                "similarity": similarity
                            })
                
                cross_domain_patterns["similar_domains"][source_domain] = similar_domains
                
                # Create transfer rules for similar domains
                for similar in similar_domains:
                    cross_domain_patterns["transfer_rules"].append({
                        "source_domain": source_domain,
                        "target_domain": similar["domain_id"],
                        "transferable_patterns": ["column_type_patterns", "relationship_patterns"],
                        "transfer_confidence": similar["similarity"]
                    })
        except ImportError:
            logger.warning("RoutingOptimizer not available, using basic cross-domain learning")
        
        # Identify universal patterns (appear across all domains)
        cross_domain_patterns["universal_patterns"] = {
            "description": "Patterns that appear across all domains",
            "examples": [
                {
                    "pattern_type": "id_column_pattern",
                    "description": "Most tables have an ID column",
                    "domains": domains,
                    "transferability": "high"
                }
            ]
        }
        
        return cross_domain_patterns
    
    def _calculate_domain_similarity(self, domain1: str, domain2: str) -> float:
        """Calculate similarity between two domains.
        
        Args:
            domain1: First domain identifier
            domain2: Second domain identifier
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        config1 = self._load_domain_config(domain1)
        config2 = self._load_domain_config(domain2)
        
        if not config1 or not config2:
            return 0.0
        
        # Compare keywords
        keywords1 = set(config1.get("keywords", []))
        keywords2 = set(config2.get("keywords", []))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        similarity = intersection / union if union > 0 else 0.0
        
        # Boost if same layer/team
        if config1.get("layer") == config2.get("layer"):
            similarity = min(1.0, similarity + 0.2)
        if config1.get("team") == config2.get("team"):
            similarity = min(1.0, similarity + 0.1)
        
        return similarity
    
    def _learn_cross_domain_patterns(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Learn patterns that transfer across domains (legacy method).
        
        Args:
            learned_patterns: Learned patterns (could be from multiple domains)
            domains: List of domain names
        
        Returns:
            Dictionary of cross-domain patterns
        """
        # Use enhanced version if available
        return self._learn_cross_domain_patterns_enhanced(learned_patterns, domains)
    
    def get_pattern_taxonomy(self) -> Dict[str, Any]:
        """Get the learned pattern taxonomy."""
        return self.pattern_taxonomy
    
    def get_abstract_patterns(self) -> Dict[str, Any]:
        """Get the learned abstract patterns."""
        return self.abstract_patterns
    
    def get_cross_domain_patterns(self) -> Dict[str, Any]:
        """Get the learned cross-domain patterns."""
        return self.cross_domain_patterns

