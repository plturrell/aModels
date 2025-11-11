"""Time encoding utilities for temporal GNNs.

Provides functions to encode time and time deltas into vector representations
for use in neural networks.
"""

import logging
import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

logger = logging.getLogger(__name__)


def encode_time(
    t: float,
    dim: int,
    max_period: float = 10000.0,
    device: Optional[str] = None
) -> "torch.Tensor":
    """Encode time to vector using sinusoidal positional encoding (Transformer-style).
    
    Uses sin/cos encoding similar to Transformer positional encodings:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        t: Time value (can be timestamp, normalized time, etc.)
        dim: Output dimension (must be even)
        max_period: Maximum period for encoding (controls frequency range)
        device: Device to create tensor on ('cuda' or 'cpu')
        
    Returns:
        Encoded time vector [dim]
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for time encoding")
    
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    
    if device is None:
        device = "cpu"
    
    # Create position encoding
    position = torch.tensor(t, device=device).float()
    
    # Create dimension indices
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device).float() *
        -(math.log(max_period) / dim)
    )
    
    # Compute sin and cos encodings
    encoding = torch.zeros(dim, device=device)
    encoding[0::2] = torch.sin(position * div_term)
    encoding[1::2] = torch.cos(position * div_term)
    
    return encoding


def encode_time_delta(
    delta_t: float,
    dim: int,
    max_period: float = 10000.0,
    device: Optional[str] = None
) -> "torch.Tensor":
    """Encode time difference to vector.
    
    Similar to encode_time but for relative time differences.
    Uses absolute value to handle negative deltas.
    
    Args:
        delta_t: Time difference (can be negative)
        dim: Output dimension (must be even)
        max_period: Maximum period for encoding
        device: Device to create tensor on
        
    Returns:
        Encoded time delta vector [dim]
    """
    # Use absolute value for encoding
    return encode_time(abs(delta_t), dim, max_period, device)


class LearnedTimeEmbedding(nn.Module):
    """Learned time embedding layer.
    
    Alternative to sinusoidal encoding that learns time representations.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_time_bins: int = 1000,
        max_time: float = 10000.0
    ):
        """Initialize learned time embedding.
        
        Args:
            embedding_dim: Output embedding dimension
            num_time_bins: Number of discrete time bins for lookup
            max_time: Maximum time value to handle
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_time_bins = num_time_bins
        self.max_time = max_time
        
        # Create embedding table
        self.time_embedding = nn.Embedding(num_time_bins, embedding_dim)
        
        logger.info(
            f"Initialized LearnedTimeEmbedding "
            f"(dim={embedding_dim}, bins={num_time_bins})"
        )
    
    def forward(self, t: "torch.Tensor") -> "torch.Tensor":
        """Embed time values.
        
        Args:
            t: Time tensor [batch_size] or scalar
            
        Returns:
            Time embeddings [batch_size, embedding_dim] or [embedding_dim]
        """
        # Normalize time to [0, num_time_bins-1]
        t_normalized = (t / self.max_time * (self.num_time_bins - 1)).long()
        t_normalized = torch.clamp(t_normalized, 0, self.num_time_bins - 1)
        
        # Get embeddings
        if t_normalized.dim() == 0:
            # Scalar input
            return self.time_embedding(t_normalized.unsqueeze(0)).squeeze(0)
        else:
            # Batch input
            return self.time_embedding(t_normalized)

