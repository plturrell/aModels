"""Explanatory AI: Generate human-readable narrative explanations."""

import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

from .narrative_graph import NarrativeGraph
from .storyline import Storyline
from .narrative_node import NarrativeNode

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generates human-readable narrative explanations from graph dynamics."""
    
    def __init__(self, narrative_graph: Optional[NarrativeGraph] = None):
        """Initialize explanation generator.
        
        Args:
            narrative_graph: Optional narrative graph to work with
        """
        self.narrative_graph = narrative_graph
        logger.info("Initialized ExplanationGenerator")
    
    def generate_narrative_explanation(
        self,
        storyline: Storyline,
        narrative_graph: Optional[NarrativeGraph] = None,
        focus_node_id: Optional[str] = None
    ) -> str:
        """Generate human-readable explanation for a storyline.
        
        Args:
            storyline: Storyline to explain
            narrative_graph: Narrative graph (uses self.narrative_graph if None)
            focus_node_id: Optional specific node to focus explanation on
            
        Returns:
            Human-readable explanation string
        """
        graph = narrative_graph or self.narrative_graph
        if not graph:
            raise ValueError("Narrative graph required")
        
        # Identify key actors
        protagonists = self.identify_key_actors(storyline, graph)
        
        # Find turning points
        turning_points = graph.find_narrative_turning_points(storyline.storyline_id)
        
        # Build causal chain
        causal_chain = graph.build_causal_chain(
            storyline.storyline_id,
            start_node_id=focus_node_id
        )
        
        # Generate explanation
        explanation_parts = []
        
        # Introduction
        explanation_parts.append(
            f"In the story of '{storyline.theme}', "
        )
        
        # Key actors
        if protagonists:
            actor_names = [node.node_id for node, _ in protagonists[:3]]
            explanation_parts.append(
                f"{', '.join(actor_names)} "
            )
        
        # Causal chain
        if causal_chain:
            explanation_parts.append(
                self._build_causal_chain_text(causal_chain, graph)
            )
        
        # Turning points
        if turning_points:
            explanation_parts.append(
                self._describe_turning_points(turning_points)
            )
        
        explanation = "".join(explanation_parts)
        
        logger.debug(f"Generated explanation for storyline {storyline.storyline_id}")
        return explanation
    
    def identify_key_actors(
        self,
        storyline: Storyline,
        graph: NarrativeGraph
    ) -> List[Tuple[NarrativeNode, float]]:
        """Identify key actors in a storyline.
        
        Args:
            storyline: Storyline
            graph: Narrative graph
            
        Returns:
            List of (node, score) tuples
        """
        return graph.identify_key_actors(storyline.storyline_id, top_k=5)
    
    def _build_causal_chain_text(
        self,
        causal_chain: List[Tuple[str, str, str]],
        graph: NarrativeGraph
    ) -> str:
        """Build text description of causal chain.
        
        Args:
            causal_chain: List of (source, target, relation) tuples
            graph: Narrative graph
            
        Returns:
            Text description
        """
        if not causal_chain:
            return ""
        
        parts = []
        for i, (source_id, target_id, relation_type) in enumerate(causal_chain):
            source_node = graph.nodes.get(source_id)
            target_node = graph.nodes.get(target_id)
            
            source_name = source_node.node_id if source_node else source_id
            target_name = target_node.node_id if target_node else target_id
            
            if i == 0:
                parts.append(f"{source_name} {relation_type} {target_name}")
            else:
                parts.append(f", which {relation_type} {target_name}")
        
        return "".join(parts) + ". "
    
    def _describe_turning_points(self, turning_points: List[Dict[str, Any]]) -> str:
        """Describe turning points in narrative.
        
        Args:
            turning_points: List of turning point events
            
        Returns:
            Text description
        """
        if not turning_points:
            return ""
        
        descriptions = []
        for event in turning_points[:3]:  # Top 3 turning points
            desc = event.get("description", "a key event")
            time = event.get("time", 0.0)
            descriptions.append(f"At time {time:.1f}, {desc}")
        
        return "Key turning points: " + "; ".join(descriptions) + ". "
    
    def answer_what_if(
        self,
        narrative_graph: NarrativeGraph,
        counterfactual_condition: Dict[str, Any],
        storyline_id: str
    ) -> Dict[str, Any]:
        """Answer what-if questions using counterfactual reasoning.
        
        Args:
            narrative_graph: Narrative graph
            counterfactual_condition: Dict describing counterfactual (e.g., {"node_id": "X", "action": "remove"})
            storyline_id: Storyline to analyze
            
        Returns:
            Dict with original_outcome, counterfactual_outcome, and comparison
        """
        # Create modified graph
        modified_graph = self._apply_counterfactual(narrative_graph, counterfactual_condition)
        
        # Run narrative on original
        original_storyline = narrative_graph.get_storyline(storyline_id)
        original_outcome = self._run_narrative(narrative_graph, original_storyline)
        
        # Run narrative on modified
        modified_storyline = modified_graph.get_storyline(storyline_id)
        modified_outcome = self._run_narrative(modified_graph, modified_storyline)
        
        # Compare
        comparison = self._compare_narratives(original_outcome, modified_outcome)
        
        return {
            "original_outcome": original_outcome,
            "counterfactual_outcome": modified_outcome,
            "comparison": comparison
        }
    
    def _apply_counterfactual(
        self,
        graph: NarrativeGraph,
        condition: Dict[str, Any]
    ) -> NarrativeGraph:
        """Apply counterfactual condition to graph.
        
        Args:
            graph: Original graph
            condition: Counterfactual condition
            
        Returns:
            Modified graph copy
        """
        # Create a copy (simplified - in practice would deep copy)
        import copy
        modified_graph = copy.deepcopy(graph)
        
        action = condition.get("action")
        node_id = condition.get("node_id")
        
        if action == "remove" and node_id:
            # Remove node
            if node_id in modified_graph.nodes:
                del modified_graph.nodes[node_id]
            # Remove edges
            modified_graph.edges = [
                e for e in modified_graph.edges
                if e.source_id != node_id and e.target_id != node_id
            ]
        elif action == "modify" and node_id:
            # Modify node properties
            if node_id in modified_graph.nodes:
                node = modified_graph.nodes[node_id]
                modifications = condition.get("modifications", {})
                for key, value in modifications.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
        
        return modified_graph
    
    def _run_narrative(
        self,
        graph: NarrativeGraph,
        storyline: Optional[Storyline]
    ) -> Dict[str, Any]:
        """Run narrative to get outcome.
        
        Args:
            graph: Narrative graph
            storyline: Storyline to run
            
        Returns:
            Outcome dict
        """
        if not storyline:
            return {"outcome": "unknown", "coherence": 0.0}
        
        # Get key actors
        key_actors = graph.identify_key_actors(storyline.storyline_id)
        
        # Get causal chain
        causal_chain = graph.build_causal_chain(storyline.storyline_id)
        
        return {
            "outcome": storyline.theme,
            "coherence": storyline.get_overall_coherence(),
            "key_actors": [node.node_id for node, _ in key_actors],
            "causal_chain_length": len(causal_chain),
            "explanatory_quality": storyline.explanatory_quality
        }
    
    def _compare_narratives(
        self,
        original: Dict[str, Any],
        counterfactual: Dict[str, Any]
    ) -> str:
        """Compare original and counterfactual narratives.
        
        Args:
            original: Original outcome
            counterfactual: Counterfactual outcome
            
        Returns:
            Comparison text
        """
        coherence_diff = counterfactual.get("coherence", 0.0) - original.get("coherence", 0.0)
        
        if coherence_diff > 0.1:
            return f"Counterfactual scenario would improve narrative coherence by {coherence_diff:.2f}"
        elif coherence_diff < -0.1:
            return f"Counterfactual scenario would reduce narrative coherence by {abs(coherence_diff):.2f}"
        else:
            return "Counterfactual scenario has similar coherence to original"

