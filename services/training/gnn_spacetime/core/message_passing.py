"""Spacetime-aware message passing for temporal GNNs.

Combines spatial (graph neighbors) and temporal (past states) aggregation.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None

from ..data.temporal_node import TemporalNode
from ..data.temporal_edge import TemporalEdge

logger = logging.getLogger(__name__)


class SpacetimeMessagePassing(nn.Module):
    """Spacetime-aware message passing layer.
    
    Combines spatial (graph neighbors) and temporal (past states) aggregation:
    agg_i(t) = RNN_aggregator({m_ij for j in spatial_neighbors}, h_i(t-1))
    """
    
    def __init__(
        self,
        node_dim: int,
        message_dim: int,
        hidden_dim: int,
        temporal_updater: Optional[nn.Module] = None,
        aggregation: str = "mean"  # "mean", "sum", "max", "attention"
    ):
        """Initialize spacetime message passing.
        
        Args:
            node_dim: Dimension of node features
            message_dim: Dimension of messages
            hidden_dim: Hidden dimension for aggregation
            temporal_updater: Optional temporal state updater (RNN/LSTM/GRU)
            aggregation: Aggregation method for spatial messages
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SpacetimeMessagePassing")
        
        super().__init__()
        
        self.node_dim = node_dim
        self.message_dim = message_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        # Message construction layers
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Temporal updater (if provided)
        self.temporal_updater = temporal_updater
        
        # Aggregation layers
        if aggregation == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=message_dim,
                num_heads=4,
                batch_first=False
            )
        
        logger.info(
            f"Initialized SpacetimeMessagePassing "
            f"(node_dim={node_dim}, message_dim={message_dim}, "
            f"hidden_dim={hidden_dim}, aggregation={aggregation})"
        )
    
    def spatial_message_construction(
        self,
        node_i: TemporalNode,
        neighbors: List[Tuple[str, TemporalNode, TemporalEdge, float]],
        time_t: float
    ) -> torch.Tensor:
        """Build messages from spatial neighbors.
        
        Args:
            node_i: Target node
            neighbors: List of (neighbor_id, neighbor_node, edge, weight) tuples
            time_t: Current time
            
        Returns:
            Aggregated spatial messages [message_dim]
        """
        if not neighbors:
            # No neighbors: return zero message
            return torch.zeros(self.message_dim)
        
        messages = []
        
        # Get node i's current state/embedding
        node_i_state = node_i.get_state_at_time(time_t)
        if node_i_state is None:
            node_i_state = node_i.static_embedding
        if node_i_state is None:
            # Fallback to zero
            node_i_state = torch.zeros(self.node_dim)
        if not isinstance(node_i_state, torch.Tensor):
            node_i_state = torch.tensor(node_i_state, dtype=torch.float)
        
        for neighbor_id, neighbor_node, edge, weight in neighbors:
            # Get neighbor's state/embedding
            neighbor_state = neighbor_node.get_state_at_time(time_t)
            if neighbor_state is None:
                neighbor_state = neighbor_node.static_embedding
            if neighbor_state is None:
                neighbor_state = torch.zeros(self.node_dim)
            if not isinstance(neighbor_state, torch.Tensor):
                neighbor_state = torch.tensor(neighbor_state, dtype=torch.float)
            
            # Get relation embedding
            if edge.relation_embedding is not None:
                rel_emb = edge.relation_embedding
                if not isinstance(rel_emb, torch.Tensor):
                    rel_emb = torch.tensor(rel_emb, dtype=torch.float)
            else:
                rel_emb = torch.zeros(self.message_dim)
            
            # Construct message: MLP(node_i, neighbor, relation, weight)
            message_input = torch.cat([
                node_i_state,
                neighbor_state,
                rel_emb[:self.message_dim] if rel_emb.shape[0] >= self.message_dim else F.pad(rel_emb, (0, self.message_dim - rel_emb.shape[0]))
            ])
            
            message = self.message_mlp(message_input)
            message = message * weight  # Weight by edge weight
            
            messages.append(message)
        
        # Aggregate messages
        if not messages:
            return torch.zeros(self.message_dim)
        
        messages_tensor = torch.stack(messages)  # [num_neighbors, message_dim]
        
        if self.aggregation == "mean":
            aggregated = messages_tensor.mean(dim=0)
        elif self.aggregation == "sum":
            aggregated = messages_tensor.sum(dim=0)
        elif self.aggregation == "max":
            aggregated = messages_tensor.max(dim=0)[0]
        elif self.aggregation == "attention":
            # Self-attention over messages
            messages_seq = messages_tensor.unsqueeze(0)  # [1, num_neighbors, message_dim]
            aggregated, _ = self.attention(messages_seq, messages_seq, messages_seq)
            aggregated = aggregated.squeeze(0).mean(dim=0)  # Average over neighbors
        else:
            aggregated = messages_tensor.mean(dim=0)
        
        return aggregated
    
    def temporal_message_construction(
        self,
        node_i: TemporalNode,
        past_states: List[Tuple[float, torch.Tensor]],
        time_t: float
    ) -> torch.Tensor:
        """Build messages from temporal neighbors (past states).
        
        Args:
            node_i: Target node
            past_states: List of (time, state) tuples from history
            time_t: Current time
            
        Returns:
            Aggregated temporal messages [message_dim]
        """
        if not past_states:
            return torch.zeros(self.message_dim)
        
        # Get most recent past state
        most_recent_time, most_recent_state = past_states[-1]
        
        if not isinstance(most_recent_state, torch.Tensor):
            most_recent_state = torch.tensor(most_recent_state, dtype=torch.float)
        
        # Project to message dimension if needed
        if most_recent_state.shape[0] != self.message_dim:
            if not hasattr(self, 'temporal_proj'):
                self.temporal_proj = nn.Linear(
                    most_recent_state.shape[0],
                    self.message_dim
                ).to(most_recent_state.device)
            most_recent_state = self.temporal_proj(most_recent_state)
        
        # Weight by time proximity (more recent = higher weight)
        time_delta = time_t - most_recent_time
        time_weight = torch.exp(-time_delta / 10.0)  # Exponential decay
        
        return most_recent_state * time_weight
    
    def spacetime_aggregation(
        self,
        spatial_msgs: torch.Tensor,
        temporal_msgs: torch.Tensor,
        previous_state: Optional[torch.Tensor],
        time_t: float
    ) -> torch.Tensor:
        """Combine spatial and temporal messages.
        
        Args:
            spatial_msgs: Aggregated spatial messages [message_dim]
            temporal_msgs: Aggregated temporal messages [message_dim]
            previous_state: Previous node state [hidden_dim] or None
            time_t: Current time
            
        Returns:
            Combined spacetime messages [message_dim]
        """
        # Combine spatial and temporal messages
        combined_msgs = spatial_msgs + temporal_msgs
        
        # If temporal updater is available, use it to update state
        if self.temporal_updater is not None and previous_state is not None:
            # Use temporal updater to combine previous state with new messages
            time_delta = torch.tensor(1.0)  # Could be computed from actual time delta
            updated_state = self.temporal_updater(previous_state, combined_msgs, time_delta)
            return updated_state
        
        # Otherwise, just return combined messages
        return combined_msgs
    
    def forward(
        self,
        node_i: TemporalNode,
        neighbors: List[Tuple[str, TemporalNode, TemporalEdge, float]],
        past_states: List[Tuple[float, torch.Tensor]],
        time_t: float,
        previous_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass: full spacetime message passing.
        
        Args:
            node_i: Target node
            neighbors: Spatial neighbors with edges and weights
            past_states: Temporal neighbors (past states)
            time_t: Current time
            previous_state: Previous node state
            
        Returns:
            Updated node state [message_dim] or [hidden_dim]
        """
        # Build spatial messages
        spatial_msgs = self.spatial_message_construction(node_i, neighbors, time_t)
        
        # Build temporal messages
        temporal_msgs = self.temporal_message_construction(node_i, past_states, time_t)
        
        # Aggregate spacetime
        updated_state = self.spacetime_aggregation(
            spatial_msgs, temporal_msgs, previous_state, time_t
        )
        
        return updated_state

