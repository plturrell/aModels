"""Enhanced Narrative GNN with LNN and MCTS integration.

Combines:
- Liquid Neural Networks for continuous-time temporal modeling
- Monte Carlo Tree Search for narrative path planning
- GNN-accelerated MCTS for efficient exploration
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Literal
import torch

from .multi_purpose_gnn import MultiPurposeNarrativeGNN
from .monte_carlo_tree_search import NarrativePathMCTS, GNNMCTS
from ..core.liquid_neural_network import LiquidStateUpdater, LiquidEdgeWeightUpdater
from ..core.temporal_models import GRUStateUpdater
from .narrative_graph import NarrativeGraph
from .storyline import Storyline, NarrativeType

logger = logging.getLogger(__name__)


class EnhancedNarrativeGNN(MultiPurposeNarrativeGNN):
    """Enhanced narrative GNN with LNN and MCTS.
    
    Extends MultiPurposeNarrativeGNN with:
    - LNN-based temporal state evolution
    - MCTS for narrative path planning
    - GNN-accelerated MCTS rollouts
    """
    
    def __init__(
        self,
        narrative_graph: Optional[NarrativeGraph] = None,
        node_dim: int = 128,
        hidden_dim: int = 64,
        message_dim: int = 128,
        use_lnn: bool = True,
        use_mcts: bool = True,
        lnn_time_constant: float = 1.0,
        mcts_rollouts: int = 100,
        mcts_exploration_c: float = 1.414
    ):
        """Initialize enhanced narrative GNN.
        
        Args:
            narrative_graph: Optional narrative graph
            node_dim: Node embedding dimension
            hidden_dim: Hidden dimension
            message_dim: Message dimension
            use_lnn: Whether to use LNN for temporal modeling
            use_mcts: Whether to use MCTS for planning
            lnn_time_constant: Time constant for LNN
            mcts_rollouts: Number of MCTS rollouts
            mcts_exploration_c: MCTS exploration constant
        """
        super().__init__(
            narrative_graph=narrative_graph,
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            message_dim=message_dim
        )
        
        self.use_lnn = use_lnn
        self.use_mcts = use_mcts
        
        # Initialize LNN state updater if enabled
        if use_lnn:
            self.lnn_state_updater = LiquidStateUpdater(
                input_dim=message_dim + hidden_dim,
                hidden_dim=hidden_dim,
                time_constant=lnn_time_constant
            )
            self.lnn_edge_updater = LiquidEdgeWeightUpdater(
                input_dim=node_dim * 2 + 64,  # source + target + relation
                hidden_dim=32,
                time_constant=lnn_time_constant
            )
            logger.info("Initialized LNN state updaters")
        else:
            # Fallback to GRU
            self.lnn_state_updater = None
            self.lnn_edge_updater = None
        
        # Initialize MCTS if enabled
        if use_mcts:
            # Create GNN-accelerated MCTS
            self.mcts = GNNMCTS(
                gnn_predictor=self._gnn_predict_state_value,
                state_evaluator=self._evaluate_narrative_state,
                action_generator=self._generate_narrative_actions,
                exploration_c=mcts_exploration_c,
                max_depth=10,
                num_rollouts=mcts_rollouts
            )
            
            # Also create narrative path MCTS for specialized planning
            self.path_mcts = NarrativePathMCTS(
                narrative_graph=narrative_graph,
                state_evaluator=self._evaluate_narrative_state,
                action_generator=self._generate_narrative_actions,
                exploration_c=mcts_exploration_c,
                max_depth=10,
                num_rollouts=mcts_rollouts
            )
            logger.info(f"Initialized MCTS (rollouts={mcts_rollouts})")
        else:
            self.mcts = None
            self.path_mcts = None
        
        logger.info("Initialized EnhancedNarrativeGNN with LNN={}, MCTS={}".format(use_lnn, use_mcts))
    
    def update_node_state_lnn(
        self,
        node_id: str,
        messages: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update node state using LNN.
        
        Args:
            node_id: Node ID
            messages: Aggregated messages [message_dim]
            prev_state: Previous state [hidden_dim]
            time_delta: Time delta since last update
            
        Returns:
            Updated state [hidden_dim]
        """
        if not self.use_lnn or self.lnn_state_updater is None:
            # Fallback to standard update
            return self._default_state_update(messages, prev_state)
        
        # Use LNN for continuous-time update
        return self.lnn_state_updater(messages, prev_state, time_delta)
    
    def update_edge_weight_lnn(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        relation_embedding: torch.Tensor,
        prev_weight: Optional[torch.Tensor] = None,
        time_delta: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Update edge weight using LNN.
        
        Args:
            source_features: Source node features [node_dim]
            target_features: Target node features [node_dim]
            relation_embedding: Relation embedding [rel_dim]
            prev_weight: Previous weight
            time_delta: Time delta
            
        Returns:
            Updated weight
        """
        if not self.use_lnn or self.lnn_edge_updater is None:
            # Fallback to static weight
            return torch.tensor(1.0)
        
        return self.lnn_edge_updater(
            source_features.unsqueeze(0),
            target_features.unsqueeze(0),
            relation_embedding.unsqueeze(0),
            prev_weight.unsqueeze(0) if prev_weight is not None else None,
            time_delta
        ).squeeze(0)
    
    def plan_narrative_path_mcts(
        self,
        current_state: Dict[str, Any],
        storyline_id: Optional[str] = None,
        num_iterations: Optional[int] = None
    ) -> Tuple[List[Any], float]:
        """Plan narrative path using MCTS.
        
        Args:
            current_state: Current narrative state
            storyline_id: Optional storyline to focus on
            num_iterations: Number of MCTS iterations
            
        Returns:
            Tuple of (best_action_sequence, best_value)
        """
        if not self.use_mcts or self.path_mcts is None:
            logger.warning("MCTS not enabled, returning empty plan")
            return [], 0.0
        
        # Update storyline if provided
        if storyline_id:
            self.path_mcts.storyline_id = storyline_id
        
        # Perform MCTS search
        best_action, best_value = self.path_mcts.search(current_state, num_iterations)
        
        if best_action:
            # Extract action sequence from MCTS tree
            action_sequence = self._extract_action_sequence(best_action)
            return action_sequence, best_value
        
        return [], 0.0
    
    def what_if_analysis_mcts(
        self,
        counterfactual_condition: Dict[str, Any],
        storyline_id: Optional[str] = None,
        num_iterations: int = 200
    ) -> Dict[str, Any]:
        """Perform what-if analysis using MCTS.
        
        Args:
            counterfactual_condition: Counterfactual change to apply
            storyline_id: Optional storyline
            num_iterations: Number of MCTS iterations
            
        Returns:
            Analysis results with original vs counterfactual outcomes
        """
        if not self.use_mcts or self.mcts is None:
            logger.warning("MCTS not enabled, using basic what-if")
            return self.explanation_generator.answer_what_if(
                self.narrative_graph,
                counterfactual_condition,
                storyline_id or ""
            )
        
        # Create initial state from counterfactual condition
        initial_state = {
            "counterfactual": counterfactual_condition,
            "storyline_id": storyline_id,
            "events": []
        }
        
        # Run MCTS to explore counterfactual outcomes
        best_action, counterfactual_value = self.mcts.search(initial_state, num_iterations)
        
        # Compare with original (run MCTS on original state)
        original_state = {
            "counterfactual": None,
            "storyline_id": storyline_id,
            "events": []
        }
        _, original_value = self.mcts.search(original_state, num_iterations // 2)
        
        return {
            "original_value": original_value,
            "counterfactual_value": counterfactual_value,
            "difference": counterfactual_value - original_value,
            "recommended_action": best_action,
            "analysis": f"Counterfactual scenario {'improves' if counterfactual_value > original_value else 'worsens'} outcome by {abs(counterfactual_value - original_value):.2%}"
        }
    
    def _gnn_predict_state_value(self, state: Any) -> float:
        """Use GNN to predict state value (for GNN-accelerated MCTS).
        
        Args:
            state: Narrative state
            
        Returns:
            Predicted value [0, 1]
        """
        # Simple heuristic: use GNN embeddings to predict value
        if isinstance(state, dict) and self.narrative_graph:
            # Extract entities from state
            events = state.get("events", [])
            if events:
                # Count events as proxy for narrative richness
                return min(1.0, len(events) * 0.1)
        
        return 0.5
    
    def _evaluate_narrative_state(self, state: Any) -> float:
        """Evaluate narrative state quality.
        
        Args:
            state: Narrative state
            
        Returns:
            Evaluation score [0, 1]
        """
        if isinstance(state, dict):
            events = state.get("events", [])
            coherence = state.get("coherence", 0.5)
            
            # Combine event count and coherence
            event_score = min(1.0, len(events) * 0.15)
            coherence_score = coherence * 0.5
            
            return min(1.0, event_score + coherence_score)
        
        return 0.5
    
    def _generate_narrative_actions(self, state: Any) -> List[Any]:
        """Generate possible narrative actions.
        
        Args:
            state: Current state
            
        Returns:
            List of possible actions
        """
        actions = []
        
        if self.narrative_graph:
            # Get active nodes
            time_t = state.get("time", 0.0) if isinstance(state, dict) else 0.0
            storyline_id = state.get("storyline_id") if isinstance(state, dict) else None
            
            nodes, edges = self.narrative_graph.get_narrative_snapshot_at_time(
                time=time_t,
                storyline_id=storyline_id
            )
            
            # Generate actions: create edges, modify nodes
            for i, source in enumerate(nodes[:5]):
                for target in nodes[i+1:min(i+3, len(nodes))]:
                    actions.append({
                        "type": "create_edge",
                        "source": source.node_id if hasattr(source, 'node_id') else str(source),
                        "target": target.node_id if hasattr(target, 'node_id') else str(target),
                        "relation": "develops"
                    })
        
        return actions if actions else [{"type": "no_op"}]
    
    def _extract_action_sequence(self, action: Any) -> List[Any]:
        """Extract action sequence from MCTS result.
        
        Args:
            action: Action from MCTS
            
        Returns:
            List of actions
        """
        if isinstance(action, dict):
            return [action]
        elif isinstance(action, list):
            return action
        return []
    
    def _default_state_update(
        self,
        messages: torch.Tensor,
        prev_state: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Default state update (fallback).
        
        Args:
            messages: Messages
            prev_state: Previous state
            
        Returns:
            Updated state
        """
        if prev_state is not None:
            return (prev_state + messages) / 2.0
        return messages

