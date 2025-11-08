"""Narrative-aware node extending TemporalNode with storyline roles and causal influence."""

import logging
from typing import Dict, Optional, Any

from ..data.temporal_node import TemporalNode

logger = logging.getLogger(__name__)


class NarrativeNode(TemporalNode):
    """Temporal node with narrative intelligence capabilities.
    
    Extends TemporalNode with:
    - Narrative roles in different storylines
    - Causal influence on narrative outcomes
    - Explanatory power for generating explanations
    """
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        features: Optional[Dict[str, Any]] = None,
        static_embedding: Optional[Any] = None,
        lifespan: Optional[tuple] = None,
        state_history: Optional[list] = None,
        properties: Optional[Dict[str, Any]] = None,
        narrative_roles: Optional[Dict[str, Dict[str, Any]]] = None,
        causal_influence: Optional[Dict[str, float]] = None,
        explanatory_power: Optional[float] = None
    ):
        """Initialize narrative node.
        
        Args:
            node_id: Unique identifier
            node_type: Type of node
            features: Static node features
            static_embedding: Time-invariant semantic embedding
            lifespan: Tuple of (start_time, end_time)
            state_history: List of (time, state_vector) tuples
            properties: Additional node properties
            narrative_roles: Dict mapping storyline_id -> {"role": str, "arc_phase": str, ...}
            causal_influence: Dict mapping storyline_id -> influence_score (0-1)
            explanatory_power: Overall importance for generating explanations (0-1)
        """
        super().__init__(
            node_id=node_id,
            node_type=node_type,
            features=features,
            static_embedding=static_embedding,
            lifespan=lifespan,
            state_history=state_history,
            properties=properties
        )
        
        self.narrative_roles = narrative_roles or {}
        self.causal_influence = causal_influence or {}
        self.explanatory_power = explanatory_power or 0.0
        
        logger.debug(
            f"Created NarrativeNode {node_id} with {len(self.narrative_roles)} narrative roles"
        )
    
    def get_narrative_role(self, storyline_id: str) -> Optional[Dict[str, Any]]:
        """Get narrative role in a specific storyline.
        
        Args:
            storyline_id: Storyline identifier
            
        Returns:
            Role dict with "role", "arc_phase", etc., or None
        """
        return self.narrative_roles.get(storyline_id)
    
    def set_narrative_role(
        self,
        storyline_id: str,
        role: str,
        arc_phase: Optional[str] = None,
        additional_attrs: Optional[Dict[str, Any]] = None
    ):
        """Set narrative role in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            role: Role type (e.g., "protagonist", "antagonist", "catalyst", "observer")
            arc_phase: Current phase in narrative arc ("setup", "rising", "climax", "falling", "resolution")
            additional_attrs: Additional role attributes
        """
        if storyline_id not in self.narrative_roles:
            self.narrative_roles[storyline_id] = {}
        
        self.narrative_roles[storyline_id]["role"] = role
        if arc_phase:
            self.narrative_roles[storyline_id]["arc_phase"] = arc_phase
        if additional_attrs:
            self.narrative_roles[storyline_id].update(additional_attrs)
        
        logger.debug(
            f"Set narrative role for {self.node_id} in {storyline_id}: {role}"
        )
    
    def get_causal_influence(self, storyline_id: Optional[str] = None) -> float:
        """Get causal influence score.
        
        Args:
            storyline_id: Optional specific storyline, otherwise returns average
            
        Returns:
            Causal influence score (0-1)
        """
        if storyline_id:
            return self.causal_influence.get(storyline_id, 0.0)
        
        if not self.causal_influence:
            return 0.0
        
        return sum(self.causal_influence.values()) / len(self.causal_influence)
    
    def set_causal_influence(self, storyline_id: str, influence: float):
        """Set causal influence in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            influence: Influence score (0-1)
        """
        self.causal_influence[storyline_id] = max(0.0, min(1.0, influence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including narrative attributes."""
        base_dict = super().to_dict()
        base_dict.update({
            "narrative_roles": self.narrative_roles,
            "causal_influence": self.causal_influence,
            "explanatory_power": self.explanatory_power
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrativeNode":
        """Create NarrativeNode from dictionary."""
        # Extract narrative-specific fields
        narrative_roles = data.pop("narrative_roles", {})
        causal_influence = data.pop("causal_influence", {})
        explanatory_power = data.pop("explanatory_power", 0.0)
        
        # Create base TemporalNode
        node = super().from_dict(data)
        
        # Convert to NarrativeNode and add narrative attributes
        narrative_node = cls(
            node_id=node.node_id,
            node_type=node.node_type,
            features=node.features,
            static_embedding=node.static_embedding,
            lifespan=node.lifespan,
            state_history=node.state_history,
            properties=node.properties,
            narrative_roles=narrative_roles,
            causal_influence=causal_influence,
            explanatory_power=explanatory_power
        )
        
        return narrative_node

