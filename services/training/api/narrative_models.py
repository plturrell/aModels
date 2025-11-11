"""Pydantic models for Narrative GNN API endpoints."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class NarrativeExplainRequest(BaseModel):
    """Request model for narrative explanation generation."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    storyline_id: Optional[str] = Field(None, description="Optional specific storyline ID")
    focus_node_id: Optional[str] = Field(None, description="Optional node to focus explanation on")
    current_time: Optional[float] = Field(None, description="Current time point for temporal analysis")


class NarrativePredictRequest(BaseModel):
    """Request model for narrative prediction."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    storyline_id: Optional[str] = Field(None, description="Optional specific storyline ID")
    current_time: float = Field(..., description="Current time point")
    future_time: Optional[float] = Field(None, description="Future time point to predict to")
    num_trajectories: int = Field(5, description="Number of trajectory candidates to generate")


class NarrativeAnomalyRequest(BaseModel):
    """Request model for narrative anomaly detection."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    storyline_id: Optional[str] = Field(None, description="Optional specific storyline ID")
    current_time: Optional[float] = Field(None, description="Current time point")
    threshold: float = Field(0.5, description="Anomaly detection threshold")


class NarrativeMCTSRequest(BaseModel):
    """Request model for MCTS what-if analysis."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    storyline_id: Optional[str] = Field(None, description="Optional specific storyline ID")
    current_time: float = Field(..., description="Current time point")
    num_rollouts: int = Field(100, description="Number of MCTS rollouts")
    max_depth: int = Field(10, description="Maximum search depth")
    exploration_c: float = Field(1.414, description="MCTS exploration constant")


class NarrativeStorylineRequest(BaseModel):
    """Request model for storyline operations."""
    nodes: List[Dict[str, Any]] = Field(..., description="Graph nodes")
    edges: List[Dict[str, Any]] = Field(..., description="Graph edges")
    storyline_id: Optional[str] = Field(None, description="Optional specific storyline ID")
    operation: str = Field("list", description="Operation: 'list', 'get', 'key_actors', 'turning_points', 'causal_chain'")

