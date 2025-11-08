"""API models and utilities for training service."""

from .gnn_models import (
    GNNEmbeddingsRequest,
    GNNClassifyRequest,
    GNNPredictLinksRequest,
    GNNStructuralInsightsRequest,
    GNNDomainQueryRequest
)

__all__ = [
    "GNNEmbeddingsRequest",
    "GNNClassifyRequest",
    "GNNPredictLinksRequest",
    "GNNStructuralInsightsRequest",
    "GNNDomainQueryRequest"
]

