"""Monte Carlo Tree Search (MCTS) for narrative reasoning.

MCTS provides:
- Planning complex narrative paths
- Improved "what-if" analysis
- Balanced exploration and exploitation
- GNN-accelerated rollouts
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
import math
import random
import logging

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""
    state: Any  # Narrative state (graph snapshot or state representation)
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    total_value: float = 0.0
    untried_actions: List[Any] = field(default_factory=list)
    action: Optional[Any] = None  # Action that led to this node
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return len(self.children) == 0 and len(self.untried_actions) == 0
    
    def ucb_value(self, exploration_c: float = 1.414) -> float:
        """Calculate UCB1 value for node selection.
        
        UCB1 = Q/N + c * sqrt(ln(Parent.N) / N)
        
        Args:
            exploration_c: Exploration constant (sqrt(2) by default)
            
        Returns:
            UCB1 value
        """
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.total_value / self.visits
        if self.parent:
            exploration = exploration_c * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
        else:
            exploration = 0.0
        
        return exploitation + exploration
    
    def best_child(self, exploration_c: float = 1.414) -> Optional['MCTSNode']:
        """Select best child using UCB1.
        
        Args:
            exploration_c: Exploration constant
            
        Returns:
            Best child node
        """
        if not self.children:
            return None
        
        best_value = -float('inf')
        best_child = None
        
        for child in self.children:
            ucb = child.ucb_value(exploration_c)
            if ucb > best_value:
                best_value = ucb
                best_child = child
        
        return best_child


class NarrativeMCTS:
    """Monte Carlo Tree Search for narrative reasoning.
    
    Uses MCTS to:
    - Plan narrative paths
    - Evaluate what-if scenarios
    - Balance exploration and exploitation
    """
    
    def __init__(
        self,
        state_evaluator: Optional[Callable] = None,
        action_generator: Optional[Callable] = None,
        rollout_policy: Optional[Callable] = None,
        exploration_c: float = 1.414,
        max_depth: int = 10,
        num_rollouts: int = 100
    ):
        """Initialize narrative MCTS.
        
        Args:
            state_evaluator: Function(state) -> float (reward/value)
            action_generator: Function(state) -> List[actions]
            rollout_policy: Function(state) -> action (for random rollouts)
            exploration_c: Exploration constant for UCB1
            max_depth: Maximum search depth
            num_rollouts: Number of rollouts per search
        """
        self.state_evaluator = state_evaluator or self._default_evaluator
        self.action_generator = action_generator or self._default_action_generator
        self.rollout_policy = rollout_policy or self._default_rollout_policy
        self.exploration_c = exploration_c
        self.max_depth = max_depth
        self.num_rollouts = num_rollouts
    
    def search(
        self,
        initial_state: Any,
        num_iterations: Optional[int] = None
    ) -> Tuple[Any, float]:
        """Perform MCTS search from initial state.
        
        Args:
            initial_state: Starting state
            num_iterations: Number of MCTS iterations (uses self.num_rollouts if None)
            
        Returns:
            Tuple of (best_action, best_value)
        """
        if num_iterations is None:
            num_iterations = self.num_rollouts
        
        root = MCTSNode(state=initial_state)
        root.untried_actions = self.action_generator(initial_state)
        
        for _ in range(num_iterations):
            # Selection: traverse to leaf
            node = self._select(root)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and node.visits > 0:
                node = self._expand(node)
            
            # Simulation: rollout from node
            value = self._simulate(node)
            
            # Backpropagation: update values up the tree
            self._backpropagate(node, value)
        
        # Return best action
        if root.children:
            best_child = root.best_child(exploration_c=0.0)  # Pure exploitation
            if best_child:
                return best_child.action, best_child.total_value / best_child.visits
        
        return None, 0.0
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selection phase: traverse to leaf using UCB1.
        
        Args:
            node: Root node
            
        Returns:
            Selected leaf node
        """
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.best_child(self.exploration_c)
            if node is None:
                break
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expansion phase: add new child node.
        
        Args:
            node: Node to expand
            
        Returns:
            New child node
        """
        if not node.untried_actions:
            return node
        
        # Select random untried action
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        # Generate new state from action
        new_state = self._apply_action(node.state, action)
        
        # Create child node
        child = MCTSNode(
            state=new_state,
            parent=node,
            action=action
        )
        child.untried_actions = self.action_generator(new_state)
        
        node.children.append(child)
        
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulation phase: random rollout to terminal state.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Estimated value from rollout
        """
        state = node.state
        depth = 0
        
        while depth < self.max_depth:
            # Check if terminal
            if self._is_terminal_state(state):
                break
            
            # Select action using rollout policy
            action = self.rollout_policy(state)
            
            # Apply action
            state = self._apply_action(state, action)
            depth += 1
        
        # Evaluate final state
        return self.state_evaluator(state)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagation phase: update values up the tree.
        
        Args:
            node: Node to start backpropagation from
            value: Value to propagate
        """
        while node is not None:
            node.visits += 1
            node.total_value += value
            node = node.parent
    
    def _apply_action(self, state: Any, action: Any) -> Any:
        """Apply action to state (to be implemented by subclasses).
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            New state
        """
        # Default: assume state is a dict and action modifies it
        if isinstance(state, dict):
            new_state = state.copy()
            if isinstance(action, dict):
                new_state.update(action)
            return new_state
        return state
    
    def _is_terminal_state(self, state: Any) -> bool:
        """Check if state is terminal (to be implemented by subclasses).
        
        Args:
            state: State to check
            
        Returns:
            True if terminal
        """
        return False
    
    def _default_evaluator(self, state: Any) -> float:
        """Default state evaluator (returns random value).
        
        Args:
            state: State to evaluate
            
        Returns:
            Random value in [0, 1]
        """
        return random.random()
    
    def _default_action_generator(self, state: Any) -> List[Any]:
        """Default action generator (returns empty list).
        
        Args:
            state: Current state
            
        Returns:
            Empty list of actions
        """
        return []
    
    def _default_rollout_policy(self, state: Any) -> Any:
        """Default rollout policy (returns None).
        
        Args:
            state: Current state
            
        Returns:
            None (no action)
        """
        return None


class GNNMCTS(NarrativeMCTS):
    """GNN-accelerated MCTS for narrative reasoning.
    
    Uses GNN to predict rollout outcomes, massively speeding up search.
    """
    
    def __init__(
        self,
        gnn_predictor: Optional[Any] = None,  # GNN model for predicting outcomes
        state_evaluator: Optional[Callable] = None,
        action_generator: Optional[Callable] = None,
        exploration_c: float = 1.414,
        max_depth: int = 10,
        num_rollouts: int = 100,
        gnn_rollout_prob: float = 0.8  # Probability of using GNN vs random rollout
    ):
        """Initialize GNN-accelerated MCTS.
        
        Args:
            gnn_predictor: GNN model for predicting state values
            state_evaluator: Function(state) -> float
            action_generator: Function(state) -> List[actions]
            exploration_c: Exploration constant
            max_depth: Maximum search depth
            num_rollouts: Number of rollouts
            gnn_rollout_prob: Probability of using GNN for rollout
        """
        super().__init__(
            state_evaluator=state_evaluator,
            action_generator=action_generator,
            exploration_c=exploration_c,
            max_depth=max_depth,
            num_rollouts=num_rollouts
        )
        self.gnn_predictor = gnn_predictor
        self.gnn_rollout_prob = gnn_rollout_prob
    
    def _simulate(self, node: MCTSNode) -> float:
        """GNN-accelerated simulation.
        
        Uses GNN to predict outcome instead of full random rollout.
        
        Args:
            node: Node to simulate from
            
        Returns:
            Predicted value
        """
        # Decide whether to use GNN or random rollout
        use_gnn = random.random() < self.gnn_rollout_prob
        
        if use_gnn and self.gnn_predictor is not None:
            # Use GNN to predict value directly
            try:
                predicted_value = self.gnn_predictor(node.state)
                if isinstance(predicted_value, torch.Tensor):
                    predicted_value = predicted_value.item()
                return float(predicted_value)
            except Exception as e:
                logger.warning(f"GNN prediction failed, falling back to rollout: {e}")
        
        # Fallback to standard rollout
        return super()._simulate(node)


class NarrativePathMCTS(NarrativeMCTS):
    """MCTS specialized for narrative path planning.
    
    Plans sequences of events/actions in narrative space.
    """
    
    def __init__(
        self,
        narrative_graph: Any,
        storyline_id: Optional[str] = None,
        state_evaluator: Optional[Callable] = None,
        **kwargs
    ):
        """Initialize narrative path MCTS.
        
        Args:
            narrative_graph: NarrativeGraph instance
            storyline_id: Optional storyline to focus on
            state_evaluator: Function to evaluate narrative states
            **kwargs: Additional MCTS parameters
        """
        super().__init__(**kwargs)
        self.narrative_graph = narrative_graph
        self.storyline_id = storyline_id
        
        if state_evaluator is None:
            self.state_evaluator = self._narrative_state_evaluator
    
    def _narrative_state_evaluator(self, state: Any) -> float:
        """Evaluate narrative state based on coherence and plausibility.
        
        Args:
            state: Narrative state (graph snapshot or event sequence)
            
        Returns:
            Evaluation score [0, 1]
        """
        # Simple heuristic: more events = potentially better narrative
        if isinstance(state, dict):
            events = state.get("events", [])
            coherence = state.get("coherence", 0.5)
            return min(1.0, len(events) * 0.1 + coherence * 0.5)
        
        return 0.5
    
    def _default_action_generator(self, state: Any) -> List[Any]:
        """Generate narrative actions (events, state changes).
        
        Args:
            state: Current narrative state
            
        Returns:
            List of possible actions
        """
        # Generate possible narrative events
        # In practice, this would use the narrative graph to suggest plausible next events
        actions = []
        
        if self.narrative_graph:
            # Get active nodes
            nodes, edges = self.narrative_graph.get_narrative_snapshot_at_time(
                time=0.0,
                storyline_id=self.storyline_id
            )
            
            # Generate actions: create new edges, modify nodes, etc.
            for i, source in enumerate(nodes[:5]):  # Limit to first 5
                for target in nodes[i+1:min(i+3, len(nodes))]:
                    actions.append({
                        "type": "create_edge",
                        "source": source.node_id if hasattr(source, 'node_id') else str(source),
                        "target": target.node_id if hasattr(target, 'node_id') else str(target),
                        "relation": "develops"
                    })
        
        return actions if actions else [{"type": "no_op"}]
    
    def _apply_action(self, state: Any, action: Any) -> Any:
        """Apply narrative action to state.
        
        Args:
            state: Current state
            action: Action to apply
            
        Returns:
            New state
        """
        if isinstance(state, dict):
            new_state = state.copy()
            events = new_state.get("events", [])
            events.append(action)
            new_state["events"] = events
            new_state["coherence"] = self._calculate_coherence(events)
            return new_state
        
        return {"events": [action], "coherence": 0.5}
    
    def _calculate_coherence(self, events: List[Any]) -> float:
        """Calculate narrative coherence score.
        
        Args:
            events: List of events
            
        Returns:
            Coherence score [0, 1]
        """
        if not events:
            return 0.0
        
        # Simple heuristic: check for consistent event types
        event_types = [e.get("type", "unknown") for e in events if isinstance(e, dict)]
        unique_types = len(set(event_types))
        total = len(event_types)
        
        # More diverse event types = potentially more coherent narrative
        diversity = unique_types / max(total, 1)
        
        return min(1.0, diversity * 0.7 + 0.3)

