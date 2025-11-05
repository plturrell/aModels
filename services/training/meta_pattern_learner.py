"""Meta-pattern learning: Learn patterns of patterns.

This module implements hierarchical pattern composition, pattern abstraction,
and cross-domain pattern transfer.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


class MetaPatternLearner:
    """Learns meta-patterns (patterns of patterns) from learned patterns."""
    
    def __init__(self):
        self.pattern_taxonomy = {}
        self.pattern_hierarchy = {}
        self.abstract_patterns = {}
        self.cross_domain_patterns = {}
    
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
        """
        logger.info("Learning meta-patterns from learned patterns...")
        
        # Build pattern taxonomy
        taxonomy = self._build_pattern_taxonomy(learned_patterns)
        
        # Learn abstract patterns
        abstract_patterns = self._learn_abstract_patterns(learned_patterns)
        
        # Learn pattern composition
        composition_patterns = self._learn_pattern_composition(learned_patterns)
        
        # Cross-domain pattern transfer (if domains provided)
        cross_domain_patterns = {}
        if domains and len(domains) > 1:
            cross_domain_patterns = self._learn_cross_domain_patterns(learned_patterns, domains)
        
        self.pattern_taxonomy = taxonomy
        self.abstract_patterns = abstract_patterns
        self.cross_domain_patterns = cross_domain_patterns
        
        meta_patterns = {
            "pattern_taxonomy": taxonomy,
            "abstract_patterns": abstract_patterns,
            "pattern_composition": composition_patterns,
            "cross_domain_patterns": cross_domain_patterns,
            "taxonomy_levels": len(taxonomy.get("levels", [])),
            "abstract_pattern_count": len(abstract_patterns),
        }
        
        logger.info(
            f"Meta-pattern learning complete: {len(taxonomy.get('levels', []))} taxonomy levels, "
            f"{len(abstract_patterns)} abstract patterns"
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
    
    def _learn_cross_domain_patterns(
        self,
        learned_patterns: Dict[str, Any],
        domains: List[str]
    ) -> Dict[str, Any]:
        """Learn patterns that transfer across domains.
        
        Args:
            learned_patterns: Learned patterns (could be from multiple domains)
            domains: List of domain names
        
        Returns:
            Dictionary of cross-domain patterns
        """
        cross_domain_patterns = {
            "universal_patterns": {},
            "domain_specific_patterns": {},
            "transfer_rules": []
        }
        
        # Identify universal patterns (appear across all domains)
        # This would require patterns from multiple domains
        # For now, create a structure for it
        
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
        
        # Transfer rules
        cross_domain_patterns["transfer_rules"] = [
            {
                "source_domain": domains[0] if domains else "unknown",
                "target_domain": domains[1] if len(domains) > 1 else "unknown",
                "transferable_patterns": ["column_type_patterns", "relationship_patterns"],
                "transfer_confidence": 0.8
            }
        ]
        
        return cross_domain_patterns
    
    def get_pattern_taxonomy(self) -> Dict[str, Any]:
        """Get the learned pattern taxonomy."""
        return self.pattern_taxonomy
    
    def get_abstract_patterns(self) -> Dict[str, Any]:
        """Get the learned abstract patterns."""
        return self.abstract_patterns
    
    def get_cross_domain_patterns(self) -> Dict[str, Any]:
        """Get the learned cross-domain patterns."""
        return self.cross_domain_patterns

