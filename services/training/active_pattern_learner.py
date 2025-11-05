"""Active learning for pattern discovery.

This module implements automated pattern discovery with active learning,
identifying rare patterns requiring human review, unsupervised pattern discovery,
and automated pattern taxonomy generation.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, Counter
import json

logger = logging.getLogger(__name__)


class ActivePatternLearner:
    """Active learning system for pattern discovery."""
    
    def __init__(self, confidence_threshold: float = 0.7, rarity_threshold: float = 0.1):
        """Initialize active pattern learner.
        
        Args:
            confidence_threshold: Minimum confidence for pattern acceptance
            rarity_threshold: Maximum frequency to consider a pattern "rare"
        """
        self.confidence_threshold = confidence_threshold
        self.rarity_threshold = rarity_threshold
        self.discovered_patterns = []
        self.rare_patterns = []
        self.pattern_confidence = {}
        self.pattern_taxonomy = {}
    
    def discover_patterns(
        self,
        learned_patterns: Dict[str, Any],
        graph_nodes: List[Dict[str, Any]],
        graph_edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Discover patterns using active learning.
        
        Args:
            learned_patterns: Previously learned patterns
            graph_nodes: Current graph nodes
            graph_edges: Current graph edges
        
        Returns:
            Dictionary with discovered patterns:
            - discovered_patterns: New patterns found
            - rare_patterns: Rare patterns requiring review
            - pattern_confidence: Confidence scores for patterns
            - pattern_taxonomy: Automated taxonomy
        """
        logger.info("Starting active pattern discovery...")
        
        # Discover new patterns
        new_patterns = self._discover_new_patterns(graph_nodes, graph_edges, learned_patterns)
        
        # Identify rare patterns
        rare_patterns = self._identify_rare_patterns(new_patterns, learned_patterns)
        
        # Calculate confidence scores
        pattern_confidence = self._calculate_pattern_confidence(new_patterns, learned_patterns)
        
        # Generate pattern taxonomy
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
        
        logger.info(
            f"Active pattern discovery complete: {len(new_patterns)} patterns discovered, "
            f"{len(rare_patterns)} rare patterns, {len(patterns_for_review)} require review"
        )
        
        return result
    
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

