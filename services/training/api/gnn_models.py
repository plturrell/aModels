"""Pydantic models for GNN API endpoints."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GNNEmbeddingsRequest(BaseModel):
    """Request model for GNN embeddings endpoint."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    graph_level: bool = Field(True, description="Generate graph-level embeddings (if False, returns node embeddings)")


class GNNClassifyRequest(BaseModel):
    """Request model for GNN node classification endpoint."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes to classify")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    top_k: Optional[int] = Field(None, description="Number of top predictions per node")


class GNNPredictLinksRequest(BaseModel):
    """Request model for GNN link prediction endpoint."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Existing graph edges")
    candidate_pairs: Optional[List[List[str]]] = Field(None, description="Candidate node pairs to evaluate as [[source_id, target_id], ...]")
    top_k: int = Field(10, description="Number of top link predictions to return")


class GNNStructuralInsightsRequest(BaseModel):
    """Request model for GNN structural insights endpoint."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    insight_type: str = Field("anomalies", description="Type of insights: 'anomalies', 'patterns', 'all'")
    threshold: Optional[float] = Field(0.5, description="Threshold for anomaly detection")


class GNNDomainQueryRequest(BaseModel):
    """Request model for domain-specific GNN query."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    query_type: str = Field("embeddings", description="Query type: 'embeddings', 'classify', 'predict-links', 'insights'")
    query_params: Optional[Dict[str, Any]] = Field(None, description="Additional query parameters")

