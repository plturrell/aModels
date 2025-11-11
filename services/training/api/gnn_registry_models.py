"""Pydantic models for GNN domain registry API."""

from typing import Dict, Optional, Any, List
from pydantic import BaseModel, Field


class RegisterModelRequest(BaseModel):
    """Request model for registering a domain-specific GNN model."""
    domain_id: str = Field(..., description="Domain identifier (e.g., 'finance', 'supply_chain')")
    model_type: str = Field(..., description="Type of model: 'embeddings', 'classifier', 'link_predictor', 'anomaly_detector', 'schema_matcher'")
    model_path: str = Field(..., description="Path to the model file")
    version: Optional[str] = Field(None, description="Model version (auto-generated if not provided)")
    training_metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics dictionary")
    model_config: Optional[Dict[str, Any]] = Field(None, description="Model configuration dictionary")
    description: Optional[str] = Field(None, description="Optional description")
    is_active: bool = Field(True, description="Whether the model is active")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    domain_id: str
    model_type: str
    version: str
    model_path: str
    created_at: str
    updated_at: str
    is_active: bool
    description: Optional[str] = None
    has_metrics: bool = False


class DomainModelInfoResponse(BaseModel):
    """Response model for domain model information."""
    domain_id: str
    models_available: bool
    models: Dict[str, Dict[str, Any]]


class ListDomainsResponse(BaseModel):
    """Response model for listing domains."""
    domains: List[str]
    count: int

