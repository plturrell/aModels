"""Temporal Node data structure for semantic spacetime graphs.

A TemporalNode represents an entity that exists in spacetime, with:
- Static semantic attributes (type, embeddings)
- Temporal attributes (lifespan, state history)
- Dynamic state that evolves over time
"""

import logging
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

logger = logging.getLogger(__name__)


class TemporalNode:
    """A node in a semantic spacetime graph with temporal state evolution.
    
    Represents an entity that exists over time with evolving state.
    """
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        features: Optional[Dict[str, Any]] = None,
        static_embedding: Optional[Any] = None,
        lifespan: Optional[Tuple[float, Optional[float]]] = None,
        state_history: Optional[List[Tuple[float, Any]]] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Initialize a temporal node.
        
        Args:
            node_id: Unique identifier for the node
            node_type: Type of node (e.g., 'table', 'column', 'researcher', 'theory')
            features: Static node features (dict)
            static_embedding: Time-invariant semantic embedding (torch.Tensor or np.ndarray)
            lifespan: Tuple of (start_time, end_time) where end_time can be None for ongoing
            state_history: List of (time, state_vector) tuples representing state evolution
            properties: Additional node properties
        """
        self.node_id = node_id
        self.node_type = node_type
        self.features = features or {}
        self.static_embedding = static_embedding
        self.lifespan = lifespan or (0.0, None)  # Default: starts at 0, never ends
        self.state_history = state_history or []
        self.current_state = None
        self.properties = properties or {}
        
        # Sort state history by time
        self.state_history.sort(key=lambda x: x[0])
        
        logger.debug(f"Created TemporalNode {node_id} with lifespan {self.lifespan}")
    
    def is_valid_at_time(self, t: float) -> bool:
        """Check if node exists at time t.
        
        Args:
            t: Time point to check
            
        Returns:
            True if node exists at time t, False otherwise
        """
        start_time, end_time = self.lifespan
        if t < start_time:
            return False
        if end_time is not None and t > end_time:
            return False
        return True
    
    def get_state_at_time(self, t: float, interpolation: str = "linear") -> Optional[Any]:
        """Get node state at time t, with interpolation if needed.
        
        Args:
            t: Time point to query
            interpolation: Interpolation method ('linear', 'nearest', 'none')
            
        Returns:
            State vector at time t, or None if not available
        """
        if not self.is_valid_at_time(t):
            return None
        
        if not self.state_history:
            return self.current_state
        
        # Find closest states
        if interpolation == "nearest":
            # Return state from nearest time point
            closest_time, closest_state = min(
                self.state_history,
                key=lambda x: abs(x[0] - t)
            )
            return closest_state
        
        elif interpolation == "linear" and HAS_TORCH:
            # Linear interpolation between adjacent states
            if len(self.state_history) == 1:
                return self.state_history[0][1]
            
            # Find states before and after t
            before_state = None
            after_state = None
            before_time = None
            after_time = None
            
            for time, state in self.state_history:
                if time <= t:
                    before_time = time
                    before_state = state
                elif time > t and after_time is None:
                    after_time = time
                    after_state = state
                    break
            
            if before_state is None:
                return after_state
            if after_state is None:
                return before_state
            
            # Linear interpolation
            if isinstance(before_state, torch.Tensor) and isinstance(after_state, torch.Tensor):
                alpha = (t - before_time) / (after_time - before_time)
                interpolated = (1 - alpha) * before_state + alpha * after_state
                return interpolated
            else:
                # Fallback to nearest if not tensors
                return before_state
        
        elif interpolation == "none":
            # Return exact match or None
            for time, state in self.state_history:
                if abs(time - t) < 1e-6:  # Exact match
                    return state
            return None
        
        # Default: return most recent state before t
        for time, state in reversed(self.state_history):
            if time <= t:
                return state
        
        return None
    
    def add_state(self, t: float, state: Any):
        """Add a state to the node's history.
        
        Args:
            t: Time point
            state: State vector (torch.Tensor or np.ndarray)
        """
        # Remove any existing state at the same time (within epsilon)
        self.state_history = [
            (time, s) for time, s in self.state_history
            if abs(time - t) > 1e-6
        ]
        
        self.state_history.append((t, state))
        self.state_history.sort(key=lambda x: x[0])
        self.current_state = state
        
        logger.debug(f"Added state to node {self.node_id} at time {t}")
    
    def get_latest_state(self) -> Optional[Any]:
        """Get the most recent state.
        
        Returns:
            Most recent state vector, or None if no history
        """
        if self.state_history:
            return self.state_history[-1][1]
        return self.current_state
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary format.
        
        Returns:
            Dictionary representation of the node
        """
        return {
            "id": self.node_id,
            "type": self.node_type,
            "features": self.features,
            "static_embedding": self.static_embedding.tolist() if hasattr(self.static_embedding, 'tolist') else self.static_embedding,
            "lifespan": self.lifespan,
            "state_history": [
                (t, s.tolist() if hasattr(s, 'tolist') else s)
                for t, s in self.state_history
            ],
            "current_state": self.current_state.tolist() if hasattr(self.current_state, 'tolist') else self.current_state,
            "properties": self.properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TemporalNode":
        """Create TemporalNode from dictionary.
        
        Args:
            data: Dictionary representation
            
        Returns:
            TemporalNode instance
        """
        # Convert static_embedding back to tensor if it was a list
        static_emb = data.get("static_embedding")
        if static_emb is not None and isinstance(static_emb, list) and HAS_TORCH:
            static_emb = torch.tensor(static_emb)
        
        # Convert state_history back to tensors
        state_history = []
        for t, s in data.get("state_history", []):
            if isinstance(s, list) and HAS_TORCH:
                s = torch.tensor(s)
            state_history.append((t, s))
        
        # Convert current_state
        current_state = data.get("current_state")
        if current_state is not None and isinstance(current_state, list) and HAS_TORCH:
            current_state = torch.tensor(current_state)
        
        return cls(
            node_id=data["id"],
            node_type=data["type"],
            features=data.get("features", {}),
            static_embedding=static_emb,
            lifespan=tuple(data.get("lifespan", (0.0, None))),
            state_history=state_history,
            properties=data.get("properties", {})
        )

