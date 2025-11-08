"""Multi-Purpose Narrative GNN: Unified model for explanation, prediction, and anomaly detection."""

import logging
from typing import Dict, List, Optional, Any, Tuple, Literal

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from ..core.message_passing import SpacetimeMessagePassing
from ..core.spacetime_attention import SemanticTemporalAttention
from .narrative_graph import NarrativeGraph
from .storyline import Storyline, NarrativeType
from .explanation_generator import ExplanationGenerator
from .narrative_predictor import NarrativePredictor
from .anomaly_detector import NarrativeAnomalyDetector

logger = logging.getLogger(__name__)


class MultiPurposeNarrativeGNN:
    """Unified narrative intelligence system for explanation, prediction, and anomaly detection.
    
    Switches between task modes while maintaining narrative consistency.
    """
    
    def __init__(
        self,
        narrative_graph: Optional[NarrativeGraph] = None,
        node_dim: int = 128,
        hidden_dim: int = 64,
        message_dim: int = 128
    ):
        """Initialize multi-purpose narrative GNN.
        
        Args:
            narrative_graph: Optional narrative graph
            node_dim: Node embedding dimension
            hidden_dim: Hidden dimension
            message_dim: Message dimension
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MultiPurposeNarrativeGNN")
        
        self.narrative_graph = narrative_graph
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.message_dim = message_dim
        
        # Initialize narrative-aware message passing
        self.message_passing = SpacetimeMessagePassing(
            node_dim=node_dim,
            message_dim=message_dim,
            hidden_dim=hidden_dim,
            aggregation="attention"
        )
        
        # Initialize semantic-temporal attention
        self.attention = SemanticTemporalAttention(
            node_dim=node_dim,
            relation_dim=64,  # Default relation dimension
            hidden_dim=hidden_dim
        )
        
        # Initialize task-specific components
        self.explanation_generator = ExplanationGenerator(narrative_graph)
        self.narrative_predictor = NarrativePredictor(narrative_graph)
        self.anomaly_detector = NarrativeAnomalyDetector(narrative_graph)
        
        logger.info("Initialized MultiPurposeNarrativeGNN")
    
    def narrative_aware_message_passing(
        self,
        source_node: Any,
        target_node: Any,
        edge: Any,
        current_storyline: Optional[Storyline],
        task_mode: Literal["explain", "predict", "detect_anomalies"],
        time_t: float
    ) -> float:
        """Compute narrative-aware message weight.
        
        Args:
            source_node: Source node
            target_node: Target node
            edge: Edge between nodes
            current_storyline: Current storyline context
            task_mode: Task mode ("explain", "predict", "detect_anomalies")
            time_t: Current time
            
        Returns:
            Message weight
        """
        # Base semantic similarity
        if hasattr(source_node, 'static_embedding') and hasattr(target_node, 'static_embedding'):
            source_emb = source_node.static_embedding
            target_emb = target_node.static_embedding
            
            if source_emb is not None and target_emb is not None:
                if not isinstance(source_emb, torch.Tensor):
                    source_emb = torch.tensor(source_emb, dtype=torch.float)
                if not isinstance(target_emb, torch.Tensor):
                    target_emb = torch.tensor(target_emb, dtype=torch.float)
                
                # Compute semantic similarity
                if source_emb.shape == target_emb.shape:
                    semantic_sim = torch.cosine_similarity(
                        source_emb.unsqueeze(0),
                        target_emb.unsqueeze(0)
                    ).item()
                else:
                    semantic_sim = 0.5  # Default if shapes don't match
            else:
                semantic_sim = 0.5
        else:
            semantic_sim = 0.5
        
        # Task-specific weighting
        if task_mode == "explain":
            # Weight by narrative relevance
            narrative_relevance = self._get_narrative_relevance(
                source_node, target_node, current_storyline
            )
            weight = semantic_sim * narrative_relevance
        
        elif task_mode == "predict":
            # Weight by causal importance
            causal_importance = self._get_causal_importance(source_node, target_node, edge)
            weight = semantic_sim * causal_importance
        
        elif task_mode == "detect_anomalies":
            # Weight by pattern deviation
            pattern_deviation = self._get_pattern_deviation(
                source_node, target_node, current_storyline
            )
            weight = semantic_sim * (1.0 - pattern_deviation)  # Lower deviation = higher weight
        
        else:
            weight = semantic_sim
        
        return weight
    
    def _get_narrative_relevance(
        self,
        source_node: Any,
        target_node: Any,
        storyline: Optional[Storyline]
    ) -> float:
        """Get narrative relevance between nodes.
        
        Args:
            source_node: Source node
            target_node: Target node
            storyline: Storyline context
            
        Returns:
            Relevance score (0-1)
        """
        if not storyline:
            return 0.5
        
        relevance = 0.5
        
        # Check if both nodes are in storyline
        from .narrative_node import NarrativeNode
        if isinstance(source_node, NarrativeNode):
            if storyline.storyline_id in source_node.narrative_roles:
                relevance += 0.25
        if isinstance(target_node, NarrativeNode):
            if storyline.storyline_id in target_node.narrative_roles:
                relevance += 0.25
        
        return min(1.0, relevance)
    
    def _get_causal_importance(
        self,
        source_node: Any,
        target_node: Any,
        edge: Any
    ) -> float:
        """Get causal importance of edge.
        
        Args:
            source_node: Source node
            target_node: Target node
            edge: Edge
            
        Returns:
            Causal importance (0-1)
        """
        from .narrative_edge import NarrativeEdge
        from .narrative_node import NarrativeNode
        if isinstance(edge, NarrativeEdge):
            return edge.causal_strength
        
        # Default based on node influence
        if isinstance(source_node, NarrativeNode):
            return source_node.get_causal_influence() or 0.5
        
        return 0.5
    
    def _get_pattern_deviation(
        self,
        source_node: Any,
        target_node: Any,
        storyline: Optional[Storyline]
    ) -> float:
        """Get pattern deviation score.
        
        Args:
            source_node: Source node
            target_node: Target node
            storyline: Storyline context
            
        Returns:
            Deviation score (0-1, higher = more deviation)
        """
        # In practice, would compare to learned patterns
        # For now, return low deviation if nodes are in storyline
        from .narrative_node import NarrativeNode
        if storyline:
            if isinstance(source_node, NarrativeNode):
                if storyline.storyline_id in source_node.narrative_roles:
                    return 0.2  # Low deviation
        
        return 0.5  # Medium deviation
    
    def forward(
        self,
        graph: NarrativeGraph,
        current_time: float,
        task_mode: Literal["explain", "predict", "detect_anomalies"] = "explain",
        storyline_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Forward pass for narrative intelligence.
        
        Args:
            graph: Narrative graph
            current_time: Current time point
            task_mode: Task mode
            storyline_id: Optional specific storyline
            
        Returns:
            Task-specific results
        """
        # Get storyline
        if storyline_id:
            storyline = graph.get_storyline(storyline_id)
        else:
            # Use first available storyline
            if graph.storylines:
                storyline = list(graph.storylines.values())[0]
            else:
                storyline = None
        
        # Run narrative-aware message passing
        narrative_states = self._narrative_message_passing(graph, current_time, storyline, task_mode)
        
        # Task-specific processing
        if task_mode == "explain":
            return self.generate_explanation(narrative_states, storyline, graph)
        elif task_mode == "predict":
            return self.predict_future_narrative(narrative_states, storyline, graph, current_time)
        elif task_mode == "detect_anomalies":
            return self.find_narrative_violations(narrative_states, storyline, graph, current_time)
        else:
            return {"error": f"Unknown task mode: {task_mode}"}
    
    def _narrative_message_passing(
        self,
        graph: NarrativeGraph,
        time_t: float,
        storyline: Optional[Storyline],
        task_mode: str
    ) -> Dict[str, Any]:
        """Run narrative-aware message passing.
        
        Args:
            graph: Narrative graph
            time_t: Current time
            storyline: Storyline context
            task_mode: Task mode
            
        Returns:
            Narrative states dict
        """
        # Get graph snapshot
        nodes, edges = graph.get_graph_at_time(time_t)
        
        narrative_states = {}
        
        from .narrative_node import NarrativeNode
        for node in nodes:
            if isinstance(node, NarrativeNode):
                node_id = node.node_id
                
                # Get neighbors
                neighbors = graph.get_temporal_neighbors(node_id, time_t)
                
                # Compute narrative-aware messages
                messages = []
                for neighbor_id, weight, time_delta in neighbors:
                    if neighbor_id in graph.nodes:
                        neighbor_node = graph.nodes[neighbor_id]
                        # Find edge
                        edge = None
                        for e in edges:
                            if (e.source_id == node_id and e.target_id == neighbor_id) or \
                               (e.target_id == node_id and e.source_id == neighbor_id):
                                edge = e
                                break
                        
                        if edge:
                            msg_weight = self.narrative_aware_message_passing(
                                node, neighbor_node, edge, storyline, task_mode, time_t
                            )
                            messages.append((neighbor_id, msg_weight, time_delta))
                
                narrative_states[node_id] = {
                    "node": node,
                    "messages": messages,
                    "time": time_t
                }
        
        return narrative_states
    
    def generate_explanation(
        self,
        narrative_states: Dict[str, Any],
        storyline: Optional[Storyline],
        graph: NarrativeGraph
    ) -> Dict[str, Any]:
        """Generate explanation.
        
        Args:
            narrative_states: Narrative states from message passing
            storyline: Storyline to explain
            graph: Narrative graph
            
        Returns:
            Explanation dict
        """
        if not storyline:
            return {"error": "Storyline required for explanation"}
        
        explanation = self.explanation_generator.generate_narrative_explanation(
            storyline, graph
        )
        
        return {
            "task": "explain",
            "storyline_id": storyline.storyline_id,
            "explanation": explanation,
            "explanatory_quality": storyline.explanatory_quality
        }
    
    def predict_future_narrative(
        self,
        narrative_states: Dict[str, Any],
        storyline: Optional[Storyline],
        graph: NarrativeGraph,
        current_time: float
    ) -> Dict[str, Any]:
        """Predict future narrative.
        
        Args:
            narrative_states: Narrative states
            storyline: Storyline to predict
            graph: Narrative graph
            current_time: Current time
            
        Returns:
            Prediction dict
        """
        if not storyline:
            return {"error": "Storyline required for prediction"}
        
        prediction = self.narrative_predictor.predict_next_chapter(
            storyline, graph, current_time
        )
        
        return {
            "task": "predict",
            "storyline_id": storyline.storyline_id,
            "prediction": prediction,
            "predictive_confidence": storyline.predictive_confidence
        }
    
    def find_narrative_violations(
        self,
        narrative_states: Dict[str, Any],
        storyline: Optional[Storyline],
        graph: NarrativeGraph,
        current_time: float
    ) -> Dict[str, Any]:
        """Find narrative violations.
        
        Args:
            narrative_states: Narrative states
            storyline: Storyline to check
            graph: Narrative graph
            current_time: Current time
            
        Returns:
            Anomaly detection dict
        """
        if not storyline:
            return {"error": "Storyline required for anomaly detection"}
        
        # Get events at current time
        events = [
            e for e in storyline.key_events
            if abs(e.get("time", 0.0) - current_time) < 5.0
        ]
        
        anomalies = self.anomaly_detector.detect_story_violations(
            events, storyline, graph
        )
        
        return {
            "task": "detect_anomalies",
            "storyline_id": storyline.storyline_id,
            "anomalies": anomalies,
            "anomaly_score": storyline.anomaly_score
        }

