"""API models and utilities for training service."""

from .gnn_models import (
    GNNEmbeddingsRequest,
    GNNClassifyRequest,
    GNNPredictLinksRequest,
    GNNStructuralInsightsRequest,
    GNNDomainQueryRequest
)

from .graph_data_models import (
    Node,
    Edge,
    Metadata,
    Quality,
    GraphData,
    from_neo4j,
    to_gnn_format,
    from_gnn,
)

__all__ = [
    "GNNEmbeddingsRequest",
    "GNNClassifyRequest",
    "GNNPredictLinksRequest",
    "GNNStructuralInsightsRequest",
    "GNNDomainQueryRequest",
    "Node",
    "Edge",
    "Metadata",
    "Quality",
    "GraphData",
    "from_neo4j",
    "to_gnn_format",
    "from_gnn",
]

