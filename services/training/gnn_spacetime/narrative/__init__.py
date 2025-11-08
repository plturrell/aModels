"""Narrative Intelligence Module for Semantic Spacetime GNNs.

Provides narrative-first modeling for explanatory AI, causal prediction, and anomaly detection.
"""

from .narrative_node import NarrativeNode
from .narrative_edge import NarrativeEdge
from .storyline import Storyline, NarrativeType
from .narrative_graph import NarrativeGraph
from .explanation_generator import ExplanationGenerator
from .narrative_predictor import NarrativePredictor
from .anomaly_detector import NarrativeAnomalyDetector
from .multi_purpose_gnn import MultiPurposeNarrativeGNN

try:
    from .enhanced_narrative_gnn import EnhancedNarrativeGNN
    from .monte_carlo_tree_search import (
        NarrativeMCTS,
        GNNMCTS,
        NarrativePathMCTS,
        MCTSNode
    )
    from .reflective_mcts import (
        ReflectiveMCTS,
        GNNReflectiveMCTS,
        Reflection
    )
    HAS_ENHANCED = True
except ImportError:
    HAS_ENHANCED = False
    EnhancedNarrativeGNN = None
    NarrativeMCTS = None
    GNNMCTS = None
    NarrativePathMCTS = None
    MCTSNode = None
    ReflectiveMCTS = None
    GNNReflectiveMCTS = None
    Reflection = None

__all__ = [
    "NarrativeNode",
    "NarrativeEdge",
    "Storyline",
    "NarrativeType",
    "NarrativeGraph",
    "ExplanationGenerator",
    "NarrativePredictor",
    "NarrativeAnomalyDetector",
    "MultiPurposeNarrativeGNN",
]

if HAS_ENHANCED:
    __all__.extend([
        "EnhancedNarrativeGNN",
        "NarrativeMCTS",
        "GNNMCTS",
        "NarrativePathMCTS",
        "MCTSNode",
        "ReflectiveMCTS",
        "GNNReflectiveMCTS",
        "Reflection",
    ])

