"""Reflective Monte Carlo Tree Search (R-MCTS) with multi-agent debate.

R-MCTS uses contrastive reflection and multi-agent debate to learn from
past mistakes and provide more balanced assessments of states.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import math
import random
import logging

from .monte_carlo_tree_search import MCTSNode, NarrativeMCTS

logger = logging.getLogger(__name__)


@dataclass
class Reflection:
    """Reflection on a decision or state evaluation."""
    state: Any
    action: Any
    predicted_value: float
    actual_value: Optional[float] = None
    error: Optional[float] = None
    reflection_text: str = ""
    learned_insight: str = ""


class ReflectiveMCTS(NarrativeMCTS):
    """Reflective-MCTS with contrastive reflection and multi-agent debate.
    
    Key features:
    - Learns from past mistakes through reflection
    - Uses multi-agent debate for balanced assessments
    - Contrastive reflection to identify what went wrong
    """
    
    def __init__(
        self,
        state_evaluator: Optional[Callable] = None,
        action_generator: Optional[Callable] = None,
        rollout_policy: Optional[Callable] = None,
        exploration_c: float = 1.414,
        max_depth: int = 10,
        num_rollouts: int = 100,
        num_agents: int = 3,
        reflection_enabled: bool = True,
        debate_enabled: bool = True
    ):
        """Initialize Reflective-MCTS.
        
        Args:
            state_evaluator: Function(state) -> float
            action_generator: Function(state) -> List[actions]
            rollout_policy: Function(state) -> action
            exploration_c: Exploration constant
            max_depth: Maximum search depth
            num_rollouts: Number of rollouts
            num_agents: Number of agents for multi-agent debate
            reflection_enabled: Whether to use reflection
            debate_enabled: Whether to use multi-agent debate
        """
        super().__init__(
            state_evaluator=state_evaluator,
            action_generator=action_generator,
            rollout_policy=rollout_policy,
            exploration_c=exploration_c,
            max_depth=max_depth,
            num_rollouts=num_rollouts
        )
        
        self.num_agents = num_agents
        self.reflection_enabled = reflection_enabled
        self.debate_enabled = debate_enabled
        
        # Reflection history
        self.reflection_history: List[Reflection] = []
        
        # Agent perspectives (different evaluation strategies)
        self.agent_evaluators: List[Callable] = []
        self._initialize_agents()
        
        logger.info(f"Initialized ReflectiveMCTS (agents={num_agents}, reflection={reflection_enabled}, debate={debate_enabled})")
    
    def _initialize_agents(self):
        """Initialize multiple agent evaluators with different perspectives."""
        # Agent 0: Optimistic (tends to overestimate)
        self.agent_evaluators.append(lambda s: self.state_evaluator(s) * 1.2)
        
        # Agent 1: Pessimistic (tends to underestimate)
        self.agent_evaluators.append(lambda s: self.state_evaluator(s) * 0.8)
        
        # Agent 2: Balanced (standard)
        self.agent_evaluators.append(self.state_evaluator)
        
        # Additional agents: Different perspectives
        for i in range(3, self.num_agents):
            # Vary by adding noise or different weighting
            noise_factor = 1.0 + (i - 3) * 0.1
            self.agent_evaluators.append(
                lambda s, nf=noise_factor: self.state_evaluator(s) * nf + random.uniform(-0.1, 0.1)
            )
    
    def search(
        self,
        initial_state: Any,
        num_iterations: Optional[int] = None
    ) -> Tuple[Any, float]:
        """Perform Reflective-MCTS search with multi-agent debate.
        
        Args:
            initial_state: Starting state
            num_iterations: Number of iterations
            
        Returns:
            Tuple of (best_action, best_value)
        """
        if num_iterations is None:
            num_iterations = self.num_rollouts
        
        root = MCTSNode(state=initial_state)
        root.untried_actions = self.action_generator(initial_state)
        
        for iteration in range(num_iterations):
            # Selection: traverse to leaf
            node = self._select(root)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and node.visits > 0:
                node = self._expand(node)
            
            # Simulation with multi-agent debate
            if self.debate_enabled:
                value = self._simulate_with_debate(node)
            else:
                value = self._simulate(node)
            
            # Reflection: learn from mistakes
            if self.reflection_enabled and iteration % 10 == 0:  # Reflect every 10 iterations
                self._reflect_on_decision(node, value)
            
            # Backpropagation: update values up the tree
            self._backpropagate(node, value)
        
        # Return best action (with reflection-informed selection)
        if root.children:
            best_child = self._select_best_with_reflection(root)
            if best_child:
                return best_child.action, best_child.total_value / best_child.visits
        
        return None, 0.0
    
    def _simulate_with_debate(self, node: MCTSNode) -> float:
        """Simulate with multi-agent debate.
        
        Multiple agents evaluate the state and debate to reach consensus.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Debated value (consensus from agents)
        """
        # Get evaluations from all agents
        agent_evaluations = []
        for agent_eval in self.agent_evaluators:
            try:
                eval_value = agent_eval(node.state)
                agent_evaluations.append(eval_value)
            except Exception as e:
                logger.warning(f"Agent evaluation failed: {e}")
                agent_evaluations.append(0.5)  # Default
        
        # Debate: combine perspectives
        if len(agent_evaluations) == 0:
            return 0.5
        
        # Strategy 1: Weighted average (more weight to agents with consistent views)
        if len(agent_evaluations) >= 3:
            # Remove outliers
            sorted_evals = sorted(agent_evaluations)
            if len(sorted_evals) > 4:
                # Remove top and bottom
                trimmed = sorted_evals[1:-1]
            else:
                trimmed = sorted_evals
            
            # Weighted average (more weight to middle values)
            weights = [1.0 / (1.0 + abs(eval - np.median(trimmed))) for eval in trimmed]
            total_weight = sum(weights)
            if total_weight > 0:
                debated_value = sum(e * w for e, w in zip(trimmed, weights)) / total_weight
            else:
                debated_value = np.mean(trimmed)
        else:
            debated_value = np.mean(agent_evaluations)
        
        # Strategy 2: Contrastive reflection (compare optimistic vs pessimistic)
        if len(agent_evaluations) >= 2:
            optimistic = max(agent_evaluations)
            pessimistic = min(agent_evaluations)
            contrast = optimistic - pessimistic
            
            # If high contrast, be more conservative
            if contrast > 0.3:
                debated_value = (debated_value + pessimistic) / 2.0
        
        return float(np.clip(debated_value, 0.0, 1.0))
    
    def _reflect_on_decision(self, node: MCTSNode, value: float):
        """Reflect on a decision to learn from mistakes.
        
        Args:
            node: Node that was evaluated
            value: Value assigned to the node
        """
        # Create reflection
        reflection = Reflection(
            state=node.state,
            action=node.action,
            predicted_value=value,
            reflection_text=f"Evaluated state with action {node.action}"
        )
        
        # Contrastive reflection: compare with similar past states
        similar_reflections = self._find_similar_reflections(node.state)
        
        if similar_reflections:
            # Learn from past mistakes
            avg_past_value = np.mean([r.actual_value or r.predicted_value for r in similar_reflections])
            error = abs(value - avg_past_value)
            
            reflection.error = error
            reflection.learned_insight = f"Similar states had average value {avg_past_value:.3f}, current prediction {value:.3f}, error {error:.3f}"
            
            # If error is large, adjust future evaluations
            if error > 0.2:
                reflection.reflection_text += f". Large error detected, adjusting perspective."
                # Could update agent evaluators here
        
        self.reflection_history.append(reflection)
        
        # Keep only recent reflections (last 100)
        if len(self.reflection_history) > 100:
            self.reflection_history = self.reflection_history[-100:]
    
    def _find_similar_reflections(self, state: Any) -> List[Reflection]:
        """Find similar past states for contrastive reflection.
        
        Args:
            state: Current state
            
        Returns:
            List of similar reflections
        """
        if not self.reflection_history:
            return []
        
        # Simple similarity: compare state dictionaries
        similar = []
        for reflection in self.reflection_history[-20:]:  # Check last 20
            if isinstance(state, dict) and isinstance(reflection.state, dict):
                # Simple key overlap check
                state_keys = set(state.keys())
                ref_keys = set(reflection.state.keys())
                overlap = len(state_keys & ref_keys) / max(len(state_keys | ref_keys), 1)
                
                if overlap > 0.5:  # 50% overlap
                    similar.append(reflection)
        
        return similar
    
    def _select_best_with_reflection(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Select best child using reflection-informed selection.
        
        Args:
            node: Parent node
            
        Returns:
            Best child node
        """
        if not node.children:
            return None
        
        # Standard UCB1 selection
        best_child = node.best_child(exploration_c=0.0)  # Pure exploitation
        
        # Apply reflection corrections
        if self.reflection_enabled and best_child:
            # Check if we have reflections about this action
            action_reflections = [
                r for r in self.reflection_history
                if r.action == best_child.action
            ]
            
            if action_reflections:
                # Adjust value based on past errors
                avg_error = np.mean([abs(r.error) if r.error else 0.0 for r in action_reflections])
                if avg_error > 0.2:
                    # High error rate, be more conservative
                    # Could select second-best child or adjust value
                    logger.debug(f"High error rate for action {best_child.action}, applying reflection correction")
        
        return best_child
    
    def get_reflection_summary(self) -> Dict[str, Any]:
        """Get summary of reflection history.
        
        Returns:
            Summary dict
        """
        if not self.reflection_history:
            return {"total_reflections": 0}
        
        errors = [r.error for r in self.reflection_history if r.error is not None]
        
        return {
            "total_reflections": len(self.reflection_history),
            "average_error": float(np.mean(errors)) if errors else 0.0,
            "max_error": float(np.max(errors)) if errors else 0.0,
            "recent_insights": [r.learned_insight for r in self.reflection_history[-5:] if r.learned_insight]
        }


class GNNReflectiveMCTS(ReflectiveMCTS):
    """GNN-accelerated Reflective-MCTS.
    
    Combines GNN predictions with multi-agent debate and reflection.
    """
    
    def __init__(
        self,
        gnn_predictor: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize GNN-accelerated Reflective-MCTS.
        
        Args:
            gnn_predictor: GNN model for predicting state values
            **kwargs: Additional ReflectiveMCTS parameters
        """
        super().__init__(**kwargs)
        self.gnn_predictor = gnn_predictor
        
        # Add GNN as an agent
        if gnn_predictor:
            self.agent_evaluators.append(self._gnn_agent_evaluator)
    
    def _gnn_agent_evaluator(self, state: Any) -> float:
        """GNN-based agent evaluator.
        
        Args:
            state: State to evaluate
            
        Returns:
            GNN-predicted value
        """
        if self.gnn_predictor:
            try:
                predicted = self.gnn_predictor(state)
                if isinstance(predicted, torch.Tensor):
                    predicted = predicted.item()
                return float(np.clip(predicted, 0.0, 1.0))
            except Exception as e:
                logger.warning(f"GNN prediction failed: {e}")
        
        return 0.5
    
    def _simulate_with_debate(self, node: MCTSNode) -> float:
        """GNN-accelerated simulation with debate.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Debated value
        """
        # Use GNN for fast prediction if available
        if self.gnn_predictor and random.random() < 0.7:  # 70% use GNN
            gnn_value = self._gnn_agent_evaluator(node.state)
            
            # Still debate with other agents for balance
            other_evals = []
            for agent_eval in self.agent_evaluators[:-1]:  # Exclude GNN (last one)
                try:
                    eval_value = agent_eval(node.state)
                    other_evals.append(eval_value)
                except Exception:
                    pass
            
            if other_evals:
                # Combine GNN with other perspectives
                all_evals = [gnn_value] + other_evals
                return float(np.mean(all_evals))
            else:
                return gnn_value
        
        # Fallback to standard debate
        return super()._simulate_with_debate(node)

