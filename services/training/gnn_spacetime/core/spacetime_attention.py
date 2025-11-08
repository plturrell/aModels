"""Temporal and semantic attention mechanisms for spacetime GNNs.

Provides attention mechanisms that consider both time proximity and semantic similarity.
"""

import logging
from typing import Optional, Tuple

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

logger = logging.getLogger(__name__)


class TemporalAttention(nn.Module):
    """Simple time-aware attention mechanism.
    
    Attention weights based on time proximity: α_ij = f(time_delta)
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        temperature: float = 1.0,
        time_decay: float = 0.1
    ):
        """Initialize temporal attention.
        
        Args:
            query_dim: Dimension of query vectors
            key_dim: Dimension of key vectors
            temperature: Temperature for attention scaling
            time_decay: Decay rate for time-based weighting
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for TemporalAttention")
        
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.temperature = temperature
        self.time_decay = time_decay
        
        # Projection layers
        self.query_proj = nn.Linear(query_dim, query_dim)
        self.key_proj = nn.Linear(key_dim, query_dim)
        
        logger.info(
            f"Initialized TemporalAttention "
            f"(query_dim={query_dim}, key_dim={key_dim}, "
            f"temperature={temperature}, time_decay={time_decay})"
        )
    
    def compute_temporal_attention(
        self,
        query_time: float,
        neighbor_times: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
        key_embs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute temporal attention weights.
        
        Args:
            query_time: Time point for query
            neighbor_times: Time points for neighbors [num_neighbors]
            query_emb: Optional query embedding [query_dim]
            key_embs: Optional key embeddings [num_neighbors, key_dim]
            
        Returns:
            Attention weights [num_neighbors]
        """
        num_neighbors = neighbor_times.shape[0]
        
        # Compute time deltas
        time_deltas = torch.abs(neighbor_times - query_time)
        
        # Time-based attention: exponential decay
        time_weights = torch.exp(-self.time_decay * time_deltas)
        
        # If embeddings provided, combine with semantic attention
        if query_emb is not None and key_embs is not None:
            # Project to same dimension
            query = self.query_proj(query_emb)  # [query_dim]
            keys = self.key_proj(key_embs)  # [num_neighbors, query_dim]
            
            # Compute semantic attention
            semantic_scores = torch.matmul(keys, query) / self.temperature
            semantic_weights = F.softmax(semantic_scores, dim=0)
            
            # Combine temporal and semantic attention
            combined_weights = time_weights * semantic_weights
            # Normalize
            combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
            
            return combined_weights
        
        # Otherwise, just use time-based weights
        time_weights = time_weights / (time_weights.sum() + 1e-8)
        return time_weights
    
    def forward(
        self,
        query_time: float,
        neighbor_times: torch.Tensor,
        query_emb: Optional[torch.Tensor] = None,
        key_embs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query_time: Query time
            neighbor_times: Neighbor times [num_neighbors]
            query_emb: Optional query embedding
            key_embs: Optional key embeddings
            
        Returns:
            Attention weights [num_neighbors]
        """
        return self.compute_temporal_attention(
            query_time, neighbor_times, query_emb, key_embs
        )


class SemanticTemporalAttention(nn.Module):
    """Attention based on semantic similarity combined with temporal proximity.
    
    α_ij = f(e_i, e_j, e_rij, time_delta)
    """
    
    def __init__(
        self,
        node_dim: int,
        relation_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        time_decay: float = 0.1
    ):
        """Initialize semantic-temporal attention.
        
        Args:
            node_dim: Dimension of node embeddings
            relation_dim: Dimension of relation embeddings
            hidden_dim: Hidden dimension for attention computation
            temperature: Temperature for attention scaling
            time_decay: Decay rate for time-based weighting
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SemanticTemporalAttention")
        
        super().__init__()
        
        self.node_dim = node_dim
        self.relation_dim = relation_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.time_decay = time_decay
        
        # Attention computation network
        self.attention_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + relation_dim + 1, hidden_dim),  # +1 for time_delta
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        logger.info(
            f"Initialized SemanticTemporalAttention "
            f"(node_dim={node_dim}, relation_dim={relation_dim}, "
            f"hidden_dim={hidden_dim})"
        )
    
    def compute_semantic_attention(
        self,
        node_emb_i: torch.Tensor,
        node_emb_j: torch.Tensor,
        relation_emb: torch.Tensor,
        time_delta: float
    ) -> float:
        """Compute semantic-temporal attention weight.
        
        Args:
            node_emb_i: Embedding of node i [node_dim]
            node_emb_j: Embedding of node j [node_dim]
            relation_emb: Embedding of relation [relation_dim]
            time_delta: Time difference between nodes
            
        Returns:
            Attention weight (scalar)
        """
        # Prepare input
        if not isinstance(node_emb_i, torch.Tensor):
            node_emb_i = torch.tensor(node_emb_i, dtype=torch.float)
        if not isinstance(node_emb_j, torch.Tensor):
            node_emb_j = torch.tensor(node_emb_j, dtype=torch.float)
        if not isinstance(relation_emb, torch.Tensor):
            relation_emb = torch.tensor(relation_emb, dtype=torch.float)
        
        # Concatenate features
        attention_input = torch.cat([
            node_emb_i,
            node_emb_j,
            relation_emb[:self.relation_dim] if relation_emb.shape[0] >= self.relation_dim else F.pad(relation_emb, (0, self.relation_dim - relation_emb.shape[0])),
            torch.tensor([time_delta], dtype=torch.float)
        ])
        
        # Compute attention score
        attention_score = self.attention_mlp(attention_input.unsqueeze(0)).squeeze()
        
        # Apply temperature and time decay
        time_weight = torch.exp(-self.time_decay * abs(time_delta))
        attention_weight = torch.sigmoid(attention_score / self.temperature) * time_weight
        
        return attention_weight.item()
    
    def forward(
        self,
        node_emb_i: torch.Tensor,
        node_emb_j: torch.Tensor,
        relation_emb: torch.Tensor,
        time_delta: float
    ) -> float:
        """Forward pass.
        
        Args:
            node_emb_i: Node i embedding
            node_emb_j: Node j embedding
            relation_emb: Relation embedding
            time_delta: Time difference
            
        Returns:
            Attention weight
        """
        return self.compute_semantic_attention(
            node_emb_i, node_emb_j, relation_emb, time_delta
        )

