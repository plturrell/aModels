"""Narrative-aware edge extending TemporalEdge with storyline significance and causal strength."""

import logging
from typing import Dict, Optional, Any, Tuple

from ..data.temporal_edge import TemporalEdge

logger = logging.getLogger(__name__)


class NarrativeEdge(TemporalEdge):
    """Temporal edge with narrative intelligence capabilities.
    
    Extends TemporalEdge with:
    - Narrative significance in different storylines
    - Causal strength quantification
    - Counterfactual importance for what-if reasoning
    """
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        relation_embedding: Optional[Any] = None,
        temporal_scope: Optional[Tuple[float, Optional[float]]] = None,
        weight_function: Optional[Any] = None,
        base_weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None,
        narrative_significance: Optional[Dict[str, float]] = None,
        causal_strength: Optional[float] = None,
        counterfactual_importance: Optional[float] = None
    ):
        """Initialize narrative edge.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation
            relation_embedding: Semantic embedding for relation
            temporal_scope: Tuple of (start_time, end_time) when edge is active
            weight_function: Callable for time-varying weight
            base_weight: Base weight
            properties: Additional edge properties
            narrative_significance: Dict mapping storyline_id -> importance_score (0-1)
            causal_strength: Quantified causal impact (0-1)
            counterfactual_importance: How much removal changes narratives (0-1)
        """
        super().__init__(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            relation_embedding=relation_embedding,
            temporal_scope=temporal_scope,
            weight_function=weight_function,
            base_weight=base_weight,
            properties=properties
        )
        
        self.narrative_significance = narrative_significance or {}
        self.causal_strength = causal_strength or 0.0
        self.counterfactual_importance = counterfactual_importance or 0.0
        
        logger.debug(
            f"Created NarrativeEdge {source_id}->{target_id} "
            f"with {len(self.narrative_significance)} narrative significances"
        )
    
    def get_narrative_significance(self, storyline_id: Optional[str] = None) -> float:
        """Get narrative significance score.
        
        Args:
            storyline_id: Optional specific storyline, otherwise returns average
            
        Returns:
            Significance score (0-1)
        """
        if storyline_id:
            return self.narrative_significance.get(storyline_id, 0.0)
        
        if not self.narrative_significance:
            return 0.0
        
        return sum(self.narrative_significance.values()) / len(self.narrative_significance)
    
    def set_narrative_significance(self, storyline_id: str, significance: float):
        """Set narrative significance in a storyline.
        
        Args:
            storyline_id: Storyline identifier
            significance: Significance score (0-1)
        """
        self.narrative_significance[storyline_id] = max(0.0, min(1.0, significance))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, including narrative attributes."""
        base_dict = super().to_dict()
        base_dict.update({
            "narrative_significance": self.narrative_significance,
            "causal_strength": self.causal_strength,
            "counterfactual_importance": self.counterfactual_importance
        })
        return base_dict
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NarrativeEdge":
        """Create NarrativeEdge from dictionary."""
        # Extract narrative-specific fields
        narrative_significance = data.pop("narrative_significance", {})
        causal_strength = data.pop("causal_strength", 0.0)
        counterfactual_importance = data.pop("counterfactual_importance", 0.0)
        
        # Create base TemporalEdge
        edge = super().from_dict(data)
        
        # Convert to NarrativeEdge and add narrative attributes
        narrative_edge = cls(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation_type=edge.relation_type,
            relation_embedding=edge.relation_embedding,
            temporal_scope=edge.temporal_scope,
            weight_function=edge.weight_function,
            base_weight=edge.base_weight,
            properties=edge.properties,
            narrative_significance=narrative_significance,
            causal_strength=causal_strength,
            counterfactual_importance=counterfactual_importance
        )
        
        return narrative_edge

