"""Active learning for pattern discovery.

This module implements automated pattern discovery with active learning,
identifying rare patterns requiring human review, unsupervised pattern discovery,
and automated pattern taxonomy generation.

Domain-aware enhancements:
- Domain filtering for pattern discovery
- Domain keyword validation
- Domain-specific pattern taxonomy
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import json
import httpx

logger = logging.getLogger(__name__)


class ActivePatternLearner:
    """Active learning system for pattern discovery.
    
    Domain-aware enhancements:
    - Filters patterns by domain
    - Validates patterns with domain keywords
    - Generates domain-specific pattern taxonomy
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        rarity_threshold: float = 0.1,
        localai_url: Optional[str] = None
    ):
        """Initialize active pattern learner.
        
        Args:
            confidence_threshold: Minimum confidence for pattern acceptance
            rarity_threshold: Maximum frequency to consider a pattern "rare"
            localai_url: LocalAI URL for domain config fetching (optional)
        """
        self.confidence_threshold = confidence_threshold
        self.rarity_threshold = rarity_threshold
        self.discovered_patterns = []
        self.rare_patterns = []
        self.pattern_confidence = {}
        self.pattern_taxonomy = {}
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_configs = {}  # domain_id -> domain config
    
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
    
    def _filter_by_domain(
        self,
        items: List[Dict[str, Any]],
        domain_id: str
    ) -> List[Dict[str, Any]]:
        """Filter items by domain keywords.
        
        Args:
            items: List of items (nodes or edges)
            domain_id: Domain identifier
        
        Returns:
            Filtered list of items matching domain
        """
        domain_config = self._load_domain_config(domain_id)
        if not domain_config:
            return items  # Return all if domain not found
        
        domain_keywords = set(kw.lower() for kw in domain_config.get("keywords", []))
        if not domain_keywords:
            return items  # Return all if no keywords
        
        filtered = []
        for item in items:
            # Check if item matches domain keywords
            item_text = str(item).lower()
            if any(kw in item_text for kw in domain_keywords):
                filtered.append(item)
        
        return filtered
    
    def _validate_with_domain_keywords(
        self,
        patterns: List[Dict[str, Any]],
        domain_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """Validate patterns using domain keywords.
        
        Args:
            patterns: List of discovered patterns
            domain_keywords: Domain keywords for validation
        
        Returns:
            Validated patterns with domain relevance scores
        """
        validated = []
        keyword_set = set(kw.lower() for kw in domain_keywords)
        
        for pattern in patterns:
            pattern_text = str(pattern).lower()
            matches = sum(1 for kw in keyword_set if kw in pattern_text)
            relevance = matches / len(keyword_set) if keyword_set else 0.0
            
            validated_pattern = pattern.copy()
            validated_pattern["domain_relevance"] = relevance
            validated_pattern["domain_keyword_matches"] = matches
            
            # Only include if relevant
            if relevance > 0.0:
                validated.append(validated_pattern)
        
        return validated
    
    def discover_patterns(
        self,
        learned_patterns: Dict[str, Any],
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]],
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Discover patterns using active learning.
        
        Args:
            learned_patterns: Previously learned patterns
            graph_nodes: Current graph nodes
            graph_edges: Current graph edges
            domain_id: Optional domain identifier for domain-specific discovery
        
        Returns:
            Dictionary with discovered patterns:
            - discovered_patterns: New patterns found
            - rare_patterns: Rare patterns requiring review
            - pattern_confidence: Confidence scores for patterns
            - pattern_taxonomy: Automated taxonomy
            - domain_patterns: Domain-specific patterns (if domain_id provided)
        """
        logger.info("Starting active pattern discovery...")
        
        # NEW: Filter by domain if specified
        domain_nodes = graph_nodes
        domain_edges = graph_edges
        if domain_id:
            logger.info(f"Filtering patterns for domain: {domain_id}")
            domain_nodes = self._filter_by_domain(graph_nodes, domain_id)
            domain_edges = self._filter_by_domain(graph_edges, domain_id)
            logger.info(f"Filtered to {len(domain_nodes)} nodes, {len(domain_edges)} edges for domain {domain_id}")
        
        # Discover new patterns (from filtered data)
        new_patterns = self._discover_new_patterns(domain_nodes, domain_edges, learned_patterns)
        
        # NEW: Validate with domain keywords if domain_id provided
        if domain_id:
            domain_config = self._load_domain_config(domain_id)
            if domain_config:
                domain_keywords = domain_config.get("keywords", [])
                validated_patterns = self._validate_with_domain_keywords(new_patterns, domain_keywords)
                new_patterns = validated_patterns
        
        # Identify rare patterns
        rare_patterns = self._identify_rare_patterns(new_patterns, learned_patterns)
        
        # Calculate confidence scores
        pattern_confidence = self._calculate_pattern_confidence(new_patterns, learned_patterns)
        
        # Generate pattern taxonomy (domain-aware if domain_id provided)
        if domain_id:
            pattern_taxonomy = self._generate_domain_pattern_taxonomy(new_patterns, learned_patterns, domain_id)
        else:
            pattern_taxonomy = self._generate_pattern_taxonomy(new_patterns, learned_patterns)
        
        # Filter patterns requiring review
        patterns_for_review = self._filter_patterns_for_review(
            new_patterns, rare_patterns, pattern_confidence
        )
        
        self.discovered_patterns = new_patterns
        self.rare_patterns = rare_patterns
        self.pattern_confidence = pattern_confidence
        self.pattern_taxonomy = pattern_taxonomy
        
        result = {
            "discovered_patterns": new_patterns,
            "rare_patterns": rare_patterns,
            "pattern_confidence": pattern_confidence,
            "pattern_taxonomy": pattern_taxonomy,
            "patterns_for_review": patterns_for_review,
            "total_discovered": len(new_patterns),
            "total_rare": len(rare_patterns),
            "total_for_review": len(patterns_for_review),
        }
        
        # NEW: Add domain-specific results
        if domain_id:
            result["domain_id"] = domain_id
            result["domain_patterns"] = {
                "total": len(new_patterns),
                "rare": len([p for p in rare_patterns if p.get("domain_relevance", 0) > 0]),
                "validated": len([p for p in new_patterns if p.get("domain_relevance", 0) > 0])
            }
        
        logger.info(
            f"Active pattern discovery complete: {len(new_patterns)} patterns discovered, "
            f"{len(rare_patterns)} rare patterns, {len(patterns_for_review)} require review"
            + (f" (domain: {domain_id})" if domain_id else "")
        )
        
        return result
    
    def _generate_domain_pattern_taxonomy(
        self,
        new_patterns: List[Dict[str, Any]],
        learned_patterns: Dict[str, Any],
        domain_id: str
    ) -> Dict[str, Any]:
        """Generate domain-specific pattern taxonomy.
        
        Args:
            new_patterns: Newly discovered patterns
            learned_patterns: Previously learned patterns
            domain_id: Domain identifier
        
        Returns:
            Domain-specific pattern taxonomy
        """
        # Generate base taxonomy
        taxonomy = self._generate_pattern_taxonomy(new_patterns, learned_patterns)
        
        # Add domain-specific metadata
        domain_config = self._load_domain_config(domain_id)
        if domain_config:
            taxonomy["domain_id"] = domain_id
            taxonomy["domain_name"] = domain_config.get("name", domain_id)
            taxonomy["domain_layer"] = domain_config.get("layer", "unknown")
            taxonomy["domain_team"] = domain_config.get("team", "unknown")
            
            # Filter patterns by domain relevance
            domain_patterns = [
                p for p in new_patterns
                if p.get("domain_relevance", 0) > 0
            ]
            taxonomy["domain_specific_patterns"] = {
                "count": len(domain_patterns),
                "patterns": domain_patterns[:10]  # Top 10
            }
        
        return taxonomy
    
    def _discover_new_patterns(
        self,
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]],
        learned_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Discover new patterns not in learned patterns.
        
        Args:
            graph_nodes: Graph nodes
            graph_edges: Graph edges
            learned_patterns: Previously learned patterns
        
        Returns:
            List of newly discovered patterns
        """
        new_patterns = []
        
        # Extract column type patterns
        column_patterns = learned_patterns.get("column_patterns", {})
        known_types = set(column_patterns.get("type_distributions", {}).keys())
        
        # Find new column types
        current_types = set()
        for node in graph_nodes:
            props = node.get("properties", {})
            if isinstance(props, str):
                try:
                    props = json.loads(props)
                except:
                    props = {}
            
            node_type = node.get("type", "")
            if node_type == "column" and isinstance(props, dict):
                col_type = props.get("type", props.get("data_type", ""))
                if col_type and col_type not in known_types:
                    current_types.add(col_type)
        
        for col_type in current_types - known_types:
            new_patterns.append({
                "pattern_type": "column_type",
                "pattern_value": col_type,
                "pattern_category": "discovered",
                "confidence": 0.5,  # Low confidence for new patterns
                "requires_review": True,
            })
        
        # Extract relationship patterns
        relationship_patterns = learned_patterns.get("relationship_patterns", {})
        known_edge_labels = set(relationship_patterns.get("edge_labels", {}).keys())
        
        # Find new edge labels
        current_edge_labels = set()
        for edge in graph_edges:
            edge_type = edge.get("type", edge.get("label", ""))
            if edge_type and edge_type not in known_edge_labels:
                current_edge_labels.add(edge_type)
        
        for edge_label in current_edge_labels - known_edge_labels:
            new_patterns.append({
                "pattern_type": "relationship_type",
                "pattern_value": edge_label,
                "pattern_category": "discovered",
                "confidence": 0.5,
                "requires_review": True,
            })
        
        # Discover sequence patterns
        sequence_patterns = self._discover_sequence_patterns(graph_nodes, graph_edges)
        new_patterns.extend(sequence_patterns)
        
        return new_patterns
    
    def _discover_sequence_patterns(
        self,
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Discover sequence patterns (Control-M → SQL → Table).
        
        Args:
            graph_nodes: Graph nodes
            graph_edges: Graph edges
        
        Returns:
            List of discovered sequence patterns
        """
        sequences = []
        
        # Build adjacency map
        adjacency = defaultdict(list)
        for edge in graph_edges:
            source = edge.get("source_id", edge.get("source", ""))
            target = edge.get("target_id", edge.get("target", ""))
            if source and target:
                adjacency[source].append(target)
        
        # Find sequences of length >= 3
        for node in graph_nodes:
            node_id = node.get("id", node.get("key", {}).get("id", ""))
            node_type = node.get("type", "")
            
            if "control-m" in node_type.lower() or "job" in node_type.lower():
                # Start sequence from Control-M node
                sequence = self._build_sequence(node_id, adjacency, graph_nodes, max_length=3)
                if len(sequence) >= 3:
                    sequences.append({
                        "pattern_type": "sequence",
                        "pattern_value": sequence,
                        "pattern_category": "discovered",
                        "confidence": 0.6,
                        "requires_review": False,
                    })
        
        return sequences
    
    def _build_sequence(
        self,
        start_id: str,
        adjacency: Dict[str, List[str]],
        graph_nodes: List[Dict[str, Any]],
        max_length: int = 3,
        visited: Optional[set] = None
    ) -> List[str]:
        """Build a sequence starting from a node.
        
        Args:
            start_id: Starting node ID
            adjacency: Adjacency map
            graph_nodes: All graph nodes
            max_length: Maximum sequence length
            visited: Set of visited nodes (for cycle detection)
        
        Returns:
            List of node IDs in sequence
        """
        if visited is None:
            visited = set()
        
        if start_id in visited or len(visited) >= max_length:
            return []
        
        visited.add(start_id)
        sequence = [start_id]
        
        if start_id in adjacency:
            for next_id in adjacency[start_id]:
                if next_id not in visited:
                    next_sequence = self._build_sequence(
                        next_id, adjacency, graph_nodes, max_length, visited.copy()
                    )
                    if next_sequence:
                        sequence.extend(next_sequence)
                        break  # Take first valid path
        
        return sequence
    
    def _identify_rare_patterns(
        self,
        new_patterns: List[Dict[str, Any]],
        learned_patterns: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify rare patterns that require review.
        
        Args:
            new_patterns: Newly discovered patterns
            learned_patterns: Previously learned patterns
        
        Returns:
            List of rare patterns
        """
        rare_patterns = []
        
        # Calculate pattern frequencies
        pattern_frequencies = self._calculate_pattern_frequencies(learned_patterns)
        
        for pattern in new_patterns:
            pattern_type = pattern.get("pattern_type")
            pattern_value = pattern.get("pattern_value")
            
            # Calculate frequency
            frequency = pattern_frequencies.get(pattern_type, {}).get(pattern_value, 0.0)
            
            if frequency < self.rarity_threshold:
                pattern["frequency"] = frequency
                pattern["is_rare"] = True
                pattern["requires_review"] = True
                rare_patterns.append(pattern)
        
        return rare_patterns
    
    def _calculate_pattern_frequencies(
        self,
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate frequencies of patterns.
        
        Args:
            learned_patterns: Learned patterns
        
        Returns:
            Dictionary of pattern type -> pattern value -> frequency
        """
        frequencies = {}
        
        # Column type frequencies
        column_patterns = learned_patterns.get("column_patterns", {})
        type_distributions = column_patterns.get("type_distributions", {})
        total_columns = sum(type_distributions.values())
        
        if total_columns > 0:
            frequencies["column_type"] = {
                col_type: count / total_columns
                for col_type, count in type_distributions.items()
            }
        
        # Relationship type frequencies
        relationship_patterns = learned_patterns.get("relationship_patterns", {})
        edge_labels = relationship_patterns.get("edge_labels", {})
        total_edges = sum(edge_labels.values())
        
        if total_edges > 0:
            frequencies["relationship_type"] = {
                edge_label: count / total_edges
                for edge_label, count in edge_labels.items()
            }
        
        return frequencies
    
    def _calculate_pattern_confidence(
        self,
        new_patterns: List[Dict[str, Any]],
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate confidence scores for patterns.
        
        Args:
            new_patterns: Newly discovered patterns
            learned_patterns: Previously learned patterns
        
        Returns:
            Dictionary of pattern ID -> confidence score
        """
        confidence_scores = {}
        
        for i, pattern in enumerate(new_patterns):
            pattern_id = f"pattern_{i}"
            base_confidence = pattern.get("confidence", 0.5)
            
            # Adjust confidence based on pattern type
            pattern_type = pattern.get("pattern_type")
            if pattern_type == "column_type":
                # Higher confidence if similar types exist
                base_confidence *= 0.8
            elif pattern_type == "relationship_type":
                # Higher confidence if similar relationships exist
                base_confidence *= 0.8
            elif pattern_type == "sequence":
                # Lower confidence for sequences (more complex)
                base_confidence *= 0.7
            
            confidence_scores[pattern_id] = base_confidence
        
        return confidence_scores
    
    def _generate_pattern_taxonomy(
        self,
        new_patterns: List[Dict[str, Any]],
        learned_patterns: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate automated pattern taxonomy.
        
        Args:
            new_patterns: Newly discovered patterns
            learned_patterns: Previously learned patterns
        
        Returns:
            Pattern taxonomy dictionary
        """
        taxonomy = {
            "levels": [],
            "categories": {},
            "hierarchical_structure": {}
        }
        
        # Level 1: Pattern types
        pattern_types = set()
        for pattern in new_patterns:
            pattern_types.add(pattern.get("pattern_type"))
        
        taxonomy["levels"].append({
            "level": 1,
            "name": "Pattern Types",
            "categories": list(pattern_types)
        })
        
        # Level 2: Pattern categories
        categories = defaultdict(list)
        for pattern in new_patterns:
            pattern_type = pattern.get("pattern_type")
            category = pattern.get("pattern_category", "discovered")
            categories[pattern_type].append(category)
        
        taxonomy["categories"] = dict(categories)
        
        # Build hierarchical structure
        taxonomy["hierarchical_structure"] = {
            pattern_type: {
                "categories": list(set(cats)),
                "pattern_count": len([p for p in new_patterns if p.get("pattern_type") == pattern_type])
            }
            for pattern_type, cats in categories.items()
        }
        
        return taxonomy
    
    def _filter_patterns_for_review(
        self,
        new_patterns: List[Dict[str, Any]],
        rare_patterns: List[Dict[str, Any]],
        pattern_confidence: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Filter patterns that require human review.
        
        Args:
            new_patterns: Newly discovered patterns
            rare_patterns: Rare patterns
            pattern_confidence: Confidence scores
        
        Returns:
            List of patterns requiring review
        """
        patterns_for_review = []
        
        for i, pattern in enumerate(new_patterns):
            pattern_id = f"pattern_{i}"
            confidence = pattern_confidence.get(pattern_id, 0.5)
            
            # Require review if:
            # 1. Low confidence (< threshold)
            # 2. Pattern is rare
            # 3. Pattern explicitly requires review
            if (confidence < self.confidence_threshold or
                pattern in rare_patterns or
                pattern.get("requires_review", False)):
                
                pattern["review_reason"] = []
                if confidence < self.confidence_threshold:
                    pattern["review_reason"].append("low_confidence")
                if pattern in rare_patterns:
                    pattern["review_reason"].append("rare_pattern")
                if pattern.get("requires_review", False):
                    pattern["review_reason"].append("explicit_flag")
                
                patterns_for_review.append(pattern)
        
        return patterns_for_review
    
    def get_patterns_for_review(self) -> List[Dict[str, Any]]:
        """Get all patterns that require human review."""
        return [
            p for p in self.discovered_patterns
            if p.get("requires_review", False)
        ]
    
    def validate_pattern(self, pattern_id: str, is_valid: bool, feedback: Optional[str] = None):
        """Validate a pattern based on human feedback.
        
        Args:
            pattern_id: Pattern identifier
            is_valid: Whether the pattern is valid
            feedback: Optional feedback text
        """
        # Update pattern confidence based on validation
        for pattern in self.discovered_patterns:
            if pattern.get("id") == pattern_id:
                if is_valid:
                    pattern["confidence"] = min(1.0, pattern.get("confidence", 0.5) + 0.2)
                    pattern["validated"] = True
                else:
                    pattern["confidence"] = max(0.0, pattern.get("confidence", 0.5) - 0.2)
                    pattern["validated"] = False
                    pattern["requires_review"] = False  # Remove from review queue
                
                if feedback:
                    pattern["feedback"] = feedback
                
                logger.info(f"Pattern {pattern_id} validated: {is_valid}")
                break

