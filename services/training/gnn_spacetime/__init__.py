"""Semantic Spacetime GNN Module.

This module extends existing GNN functionality with temporal and semantic spacetime capabilities,
enabling dynamic graph modeling where nodes and edges evolve over time with semantic meaning.
"""

from .data.temporal_node import TemporalNode
from .data.temporal_edge import TemporalEdge
from .data.temporal_graph import TemporalGraph
from .core.temporal_models import RNNStateUpdater, LSTMStateUpdater, GRUStateUpdater
from .core.message_passing import SpacetimeMessagePassing
from .core.spacetime_attention import TemporalAttention, SemanticTemporalAttention
from .models.spacetime_embedder import SpacetimeEmbedder
from .utils.time_encoding import encode_time, encode_time_delta, LearnedTimeEmbedding
from .utils.data_utils import convert_to_temporal_graph, extract_temporal_features

# Narrative intelligence
try:
    from .narrative import (
        NarrativeNode, NarrativeEdge, Storyline, NarrativeType,
        NarrativeGraph, MultiPurposeNarrativeGNN,
        ExplanationGenerator, NarrativePredictor, NarrativeAnomalyDetector
    )
    HAS_NARRATIVE = True
except ImportError:
    HAS_NARRATIVE = False
    NarrativeNode = None
    NarrativeEdge = None
    Storyline = None
    NarrativeType = None
    NarrativeGraph = None
    MultiPurposeNarrativeGNN = None
    ExplanationGenerator = None
    NarrativePredictor = None
    NarrativeAnomalyDetector = None

__all__ = [
    # Data structures
    "TemporalNode",
    "TemporalEdge",
    "TemporalGraph",
    # Core components
    "RNNStateUpdater",
    "LSTMStateUpdater",
    "GRUStateUpdater",
    "SpacetimeMessagePassing",
    "TemporalAttention",
    "SemanticTemporalAttention",
    # Models
    "SpacetimeEmbedder",
    # Utilities
    "encode_time",
    "encode_time_delta",
    "LearnedTimeEmbedding",
    "convert_to_temporal_graph",
    "extract_temporal_features",
]

# Add narrative components if available
if HAS_NARRATIVE:
    __all__.extend([
        "NarrativeNode",
        "NarrativeEdge",
        "Storyline",
        "NarrativeType",
        "NarrativeGraph",
        "MultiPurposeNarrativeGNN",
        "ExplanationGenerator",
        "NarrativePredictor",
        "NarrativeAnomalyDetector",
    ])

