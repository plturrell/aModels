"""Temporal Edge data structure for semantic spacetime graphs.

A TemporalEdge represents a relationship that exists in spacetime, with:
- Static semantic attributes (relation type, embeddings)
- Temporal attributes (scope, time-varying weight)
- Dynamic strength that evolves over time
"""

import logging
from typing import Optional, Tuple, Callable, Dict, Any
import math

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class TemporalEdge:
    """An edge in a semantic spacetime graph with temporal scope and dynamic weight.
    
    Represents a relationship that exists over time with evolving strength.
    """
    
    def __init__(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        relation_embedding: Optional[Any] = None,
        temporal_scope: Optional[Tuple[float, Optional[float]]] = None,
        weight_function: Optional[Callable[[float], float]] = None,
        base_weight: float = 1.0,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Initialize a temporal edge.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relation (e.g., 'influences', 'cites', 'HAS_COLUMN')
            relation_embedding: Semantic embedding for the relation type
            temporal_scope: Tuple of (start_time, end_time) when edge is active
            weight_function: Callable w(t) that returns edge weight at time t
            base_weight: Base weight (used if weight_function is None)
            properties: Additional edge properties
        """
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.relation_embedding = relation_embedding
        self.temporal_scope = temporal_scope
        self.weight_function = weight_function
        self.base_weight = base_weight
        self.properties = properties or {}
        
        logger.debug(
            f"Created TemporalEdge {source_id}->{target_id} "
            f"({relation_type}) with scope {temporal_scope}"
        )
    
    def is_active_at_time(self, t: float) -> bool:
        """Check if edge exists at time t.
        
        Args:
            t: Time point to check
            
        Returns:
            True if edge is active at time t, False otherwise
        """
        if self.temporal_scope is None:
            return True  # Always active if no scope specified
        
        start_time, end_time = self.temporal_scope
        if t < start_time:
            return False
        if end_time is not None and t > end_time:
            return False
        return True
    
    def get_weight_at_time(self, t: float, reference_time: Optional[float] = None) -> float:
        """Compute edge weight at time t.
        
        Args:
            t: Time point to query
            reference_time: Optional reference time for relative weight functions
            
        Returns:
            Edge weight at time t
        """
        if not self.is_active_at_time(t):
            return 0.0
        
        if self.weight_function is not None:
            # Use provided weight function
            if reference_time is not None:
                # For relative time functions (e.g., decay since event)
                delta_t = t - reference_time
                return self.weight_function(delta_t)
            else:
                # For absolute time functions
                return self.weight_function(t)
        
        # Default: return base weight
        return self.base_weight
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary format.
        
        Returns:
            Dictionary representation of the edge
        """
        # Note: weight_function cannot be serialized, so we store its type/params if available
        weight_func_info = None
        if self.weight_function is not None:
            # Try to extract function info if it's a known type
            if hasattr(self.weight_function, '__name__'):
                weight_func_info = {
                    "type": self.weight_function.__name__,
                    "module": getattr(self.weight_function, '__module__', None)
                }
        
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "relation_embedding": (
                self.relation_embedding.tolist()
                if hasattr(self.relation_embedding, 'tolist')
                else self.relation_embedding
            ),
            "temporal_scope": self.temporal_scope,
            "weight_function_info": weight_func_info,
            "base_weight": self.base_weight,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalEdge":
        """Create TemporalEdge from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TemporalEdge instance
        """
        # Convert relation_embedding back to tensor if it was a list
        relation_emb = data.get("relation_embedding")
        if relation_emb is not None and isinstance(relation_emb, list) and HAS_TORCH:
            relation_emb = torch.tensor(relation_emb)
        
        # Note: weight_function cannot be restored from dict, will use base_weight
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            relation_embedding=relation_emb,
            temporal_scope=(
                tuple(data["temporal_scope"])
                if data.get("temporal_scope") is not None
                else None
            ),
            weight_function=None,  # Cannot restore from dict
            base_weight=data.get("base_weight", 1.0),
            properties=data.get("properties", {})
        )


# Common weight function factories

def exponential_decay(decay_rate: float = 0.1) -> Callable[[float], float]:
    """Create exponential decay weight function: w(t) = exp(-decay_rate * t).
    
    Args:
        decay_rate: Decay rate (higher = faster decay)
        
    Returns:
        Weight function
    """
    def weight_func(delta_t: float) -> float:
        return math.exp(-decay_rate * max(0, delta_t))
    return weight_func


def linear_decay(slope: float = 0.01, max_time: Optional[float] = None) -> Callable[[float], float]:
    """Create linear decay weight function: w(t) = max(0, 1 - slope * t).
    
    Args:
        slope: Decay slope
        max_time: Optional maximum time before weight becomes 0
        
    Returns:
        Weight function
    """
    def weight_func(delta_t: float) -> float:
        if max_time is not None and delta_t > max_time:
            return 0.0
        return max(0.0, 1.0 - slope * delta_t)
    return weight_func


def constant_weight(weight: float = 1.0) -> Callable[[float], float]:
    """Create constant weight function: w(t) = weight.
    
    Args:
        weight: Constant weight value
        
    Returns:
        Weight function
    """
    def weight_func(delta_t: float) -> float:
        return weight
    return weight_func

