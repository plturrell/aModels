"""Core spacetime GNN components."""

from .temporal_models import RNNStateUpdater, LSTMStateUpdater, GRUStateUpdater
from .message_passing import SpacetimeMessagePassing
from .spacetime_attention import TemporalAttention, SemanticTemporalAttention

try:
    from .liquid_neural_network import (
        LiquidLayer,
        LiquidStateUpdater,
        LiquidEdgeWeightUpdater
    )
    HAS_LNN = True
except ImportError:
    HAS_LNN = False
    LiquidLayer = None
    LiquidStateUpdater = None
    LiquidEdgeWeightUpdater = None

__all__ = [
    "RNNStateUpdater",
    "LSTMStateUpdater",
    "GRUStateUpdater",
    "SpacetimeMessagePassing",
    "TemporalAttention",
    "SemanticTemporalAttention",
]

if HAS_LNN:
    __all__.extend([
        "LiquidLayer",
        "LiquidStateUpdater",
        "LiquidEdgeWeightUpdater",
    ])

