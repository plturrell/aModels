"""Narrative-aware temporal graph managing storylines and narrative intelligence."""

import logging
from typing import List, Dict, Optional, Any, Tuple

from ..data.temporal_graph import TemporalGraph
from .narrative_node import NarrativeNode
from .narrative_edge import NarrativeEdge
from .storyline import Storyline, NarrativeType

logger = logging.getLogger(__name__)


class NarrativeGraph(TemporalGraph):
    """Temporal graph with narrative intelligence capabilities.
    
    Extends TemporalGraph with storyline management and narrative-aware operations.
    """
    
    def __init__(
        self,
        nodes: Optional[List[NarrativeNode]] = None,
        edges: Optional[List[NarrativeEdge]] = None,
        storylines: Optional[Dict[str, Storyline]] = None,
        snapshots: Optional[Dict[float, Tuple[List, List]]] = None
    ):
        """Initialize narrative graph.
        
        Args:
            nodes: List of narrative nodes
            edges: List of narrative edges
            storylines: Dict mapping storyline_id -> Storyline
            snapshots: Pre-computed graph snapshots
        """
        # Convert to base TemporalGraph format
        base_nodes = nodes or []
        base_edges = edges or []
        
        super().__init__(
            nodes=base_nodes,
            edges=base_edges,
            snapshots=snapshots
        )
        
        # Store narrative-specific data
        self.storylines = storylines or {}
        
        logger.info(
            f"Created NarrativeGraph with {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges, {len(self.storylines)} storylines"
        )
    
    def add_storyline(self, storyline: Storyline):
        """Add a storyline to the graph.
        
        Args:
            storyline: Storyline to add
        """
        self.storylines[storyline.storyline_id] = storyline
        logger.debug(f"Added storyline {storyline.storyline_id} to narrative graph")
    
    def get_storyline(self, storyline_id: str) -> Optional[Storyline]:
        """Get storyline by ID.
        
        Args:
            storyline_id: Storyline identifier
            
        Returns:
            Storyline or None if not found
        """
        return self.storylines.get(storyline_id)
    
    def get_nodes_in_storyline(self, storyline_id: str) -> List[NarrativeNode]:
        """Get all nodes participating in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            
        Returns:
            List of NarrativeNodes in the storyline
        """
        nodes = []
        for node in self.nodes.values():
            if isinstance(node, NarrativeNode):
                if storyline_id in node.narrative_roles:
                    nodes.append(node)
        return nodes
    
    def get_edges_in_storyline(self, storyline_id: str) -> List[NarrativeEdge]:
        """Get all edges participating in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            
        Returns:
            List of NarrativeEdges in the storyline
        """
        edges = []
        for edge in self.edges:
            if isinstance(edge, NarrativeEdge):
                if storyline_id in edge.narrative_significance:
                    edges.append(edge)
        return edges
    
    def identify_key_actors(self, storyline_id: str, top_k: int = 5) -> List[Tuple[NarrativeNode, float]]:
        """Identify key actors (nodes) in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            top_k: Number of top actors to return
            
        Returns:
            List of (node, causal_influence) tuples, sorted by influence
        """
        nodes = self.get_nodes_in_storyline(storyline_id)
        
        # Score by causal influence and explanatory power
        scored_nodes = []
        for node in nodes:
            influence = node.get_causal_influence(storyline_id)
            score = influence * (1.0 + node.explanatory_power)
            scored_nodes.append((node, score))
        
        # Sort by score
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return scored_nodes[:top_k]
    
    def find_narrative_turning_points(self, storyline_id: str) -> List[Dict[str, Any]]:
        """Find turning points in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            
        Returns:
            List of turning point events
        """
        storyline = self.get_storyline(storyline_id)
        if not storyline:
            return []
        
        # Key events are potential turning points
        turning_points = []
        
        for event in storyline.key_events:
            # Check if event has high causal impact
            node_id = event.get("node_id")
            if node_id and node_id in self.nodes:
                node = self.nodes[node_id]
                if isinstance(node, NarrativeNode):
                    influence = node.get_causal_influence(storyline_id)
                    if influence > 0.5:  # High influence threshold
                        turning_points.append(event)
        
        return turning_points
    
    def build_causal_chain(
        self,
        storyline_id: str,
        start_node_id: Optional[str] = None,
        end_node_id: Optional[str] = None
    ) -> List[Tuple[str, str, str]]:
        """Build causal chain for a storyline.
        
        Args:
            storyline_id: Storyline identifier
            start_node_id: Optional starting node
            end_node_id: Optional ending node
            
        Returns:
            List of (source_id, target_id, relation_type) tuples
        """
        storyline = self.get_storyline(storyline_id)
        if not storyline:
            return []
        
        # Use storyline's causal links
        causal_chain = storyline.causal_links.copy()
        
        # If start/end specified, filter chain
        if start_node_id:
            # Find path from start_node
            filtered_chain = []
            current_node = start_node_id
            visited = set()
            
            while current_node and current_node not in visited:
                visited.add(current_node)
                # Find next link
                for source, target, rel_type in causal_chain:
                    if source == current_node:
                        filtered_chain.append((source, target, rel_type))
                        current_node = target
                        break
                else:
                    break
            
            causal_chain = filtered_chain
        
        if end_node_id:
            # Filter to only include links leading to end_node
            causal_chain = [
                (s, t, r) for s, t, r in causal_chain
                if t == end_node_id or s == end_node_id
            ]
        
        return causal_chain

