"""Causal Prediction: Predict future narrative states and events."""

import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

from .narrative_graph import NarrativeGraph
from .storyline import Storyline
from .narrative_node import NarrativeNode

logger = logging.getLogger(__name__)


class NarrativePredictor:
    """Predicts future narrative states and events based on current storyline."""
    
    def __init__(
        self,
        narrative_graph: Optional[NarrativeGraph] = None,
        prediction_horizon: float = 100.0
    ):
        """Initialize narrative predictor.
        
        Args:
            narrative_graph: Optional narrative graph
            prediction_horizon: Default prediction horizon in time units
        """
        self.narrative_graph = narrative_graph
        self.prediction_horizon = prediction_horizon
        logger.info(f"Initialized NarrativePredictor (horizon: {prediction_horizon})")
    
    def predict_next_chapter(
        self,
        current_storyline: Storyline,
        graph_state: NarrativeGraph,
        time_t: float,
        num_trajectories: int = 5
    ) -> Dict[str, Any]:
        """Predict most plausible next events in storyline.
        
        Args:
            current_storyline: Current storyline state
            graph_state: Current graph state
            time_t: Current time
            num_trajectories: Number of possible futures to consider
            
        Returns:
            Dict with predicted_events, confidence, and trajectory_details
        """
        # Generate plausible narrative arcs
        possible_futures = self._generate_plausible_narrative_arcs(
            current_storyline, graph_state, time_t, num_trajectories
        )
        
        # Score by coherence and causal plausibility
        scored_futures = []
        for future in possible_futures:
            coherence_score = self._calculate_narrative_coherence(future, graph_state)
            causal_plausibility = self._assess_causal_plausibility(future, graph_state)
            
            combined_score = coherence_score * causal_plausibility
            scored_futures.append((future, combined_score, coherence_score, causal_plausibility))
        
        # Get best prediction
        best_future, best_score, coherence, plausibility = max(
            scored_futures, key=lambda x: x[1]
        )
        
        return {
            "predicted_events": best_future.get("events", []),
            "confidence": best_score,
            "coherence_score": coherence,
            "causal_plausibility": plausibility,
            "trajectory_details": {
                "all_trajectories": [
                    {
                        "events": f.get("events", []),
                        "score": score,
                        "coherence": coh,
                        "plausibility": plau
                    }
                    for f, score, coh, plau in scored_futures
                ]
            }
        }
    
    def _generate_plausible_narrative_arcs(
        self,
        storyline: Storyline,
        graph: NarrativeGraph,
        time_t: float,
        num_arcs: int
    ) -> List[Dict[str, Any]]:
        """Generate plausible future narrative arcs.
        
        Args:
            storyline: Current storyline
            graph: Current graph state
            time_t: Current time
            num_arcs: Number of arcs to generate
            
        Returns:
            List of future arc dicts
        """
        arcs = []
        
        # Get key actors
        key_actors = graph.identify_key_actors(storyline.storyline_id)
        
        # Get current causal chain
        causal_chain = graph.build_causal_chain(storyline.storyline_id)
        
        # Generate variations based on:
        # 1. Continuation of current trends
        # 2. Reversal of current trends
        # 3. New actors entering
        # 4. Existing actors changing roles
        
        for i in range(num_arcs):
            arc = {
                "arc_id": f"arc_{i}",
                "events": [],
                "time_range": (time_t, time_t + self.prediction_horizon)
            }
            
            # Generate events based on current state
            if key_actors:
                # Continue with key actors
                for node, influence in key_actors[:3]:
                    event = {
                        "node_id": node.node_id,
                        "time": time_t + (i + 1) * (self.prediction_horizon / num_arcs),
                        "type": "continuation" if i % 2 == 0 else "reversal",
                        "description": f"{node.node_id} continues development" if i % 2 == 0 else f"{node.node_id} reverses course"
                    }
                    arc["events"].append(event)
            
            # Add events from causal chain continuation
            if causal_chain:
                last_link = causal_chain[-1]
                target_id = last_link[1]
                if target_id in graph.nodes:
                    event = {
                        "node_id": target_id,
                        "time": time_t + self.prediction_horizon * 0.5,
                        "type": "causal_consequence",
                        "description": f"Consequence of {last_link[0]} -> {target_id}"
                    }
                    arc["events"].append(event)
            
            arcs.append(arc)
        
        return arcs
    
    def _calculate_narrative_coherence(
        self,
        future_arc: Dict[str, Any],
        graph: NarrativeGraph
    ) -> float:
        """Calculate narrative coherence of a future arc.
        
        Args:
            future_arc: Future arc dict
            graph: Current graph state
            
        Returns:
            Coherence score (0-1)
        """
        # Check consistency of events
        events = future_arc.get("events", [])
        if not events:
            return 0.5  # Neutral if no events
        
        # Score based on:
        # 1. Temporal ordering (events should be in order)
        times = [e.get("time", 0.0) for e in events]
        is_ordered = all(times[i] <= times[i+1] for i in range(len(times)-1))
        ordering_score = 1.0 if is_ordered else 0.5
        
        # 2. Node consistency (nodes should exist in graph)
        node_consistency = 0.0
        for event in events:
            node_id = event.get("node_id")
            if node_id and node_id in graph.nodes:
                node_consistency += 1.0
        node_consistency = node_consistency / len(events) if events else 0.0
        
        # 3. Event type diversity (some diversity is good)
        event_types = [e.get("type", "unknown") for e in events]
        diversity = len(set(event_types)) / len(event_types) if event_types else 0.0
        
        # Combined score
        coherence = (ordering_score * 0.4 + node_consistency * 0.4 + diversity * 0.2)
        
        return coherence
    
    def _assess_causal_plausibility(
        self,
        future_arc: Dict[str, Any],
        graph: NarrativeGraph
    ) -> float:
        """Assess causal plausibility of future arc.
        
        Args:
            future_arc: Future arc dict
            graph: Current graph state
            
        Returns:
            Plausibility score (0-1)
        """
        events = future_arc.get("events", [])
        if not events:
            return 0.5
        
        # Check if events follow from existing causal chains
        plausibility_scores = []
        
        for event in events:
            node_id = event.get("node_id")
            if not node_id or node_id not in graph.nodes:
                plausibility_scores.append(0.3)  # Low if node doesn't exist
                continue
            
            node = graph.nodes[node_id]
            if isinstance(node, NarrativeNode):
                # Check causal influence
                avg_influence = node.get_causal_influence()
                plausibility_scores.append(0.5 + avg_influence * 0.5)  # Scale to 0.5-1.0
            else:
                plausibility_scores.append(0.5)
        
        return sum(plausibility_scores) / len(plausibility_scores) if plausibility_scores else 0.5
    
    def predict_future_narrative(
        self,
        storyline_id: str,
        graph: NarrativeGraph,
        time_t: float,
        future_time: float
    ) -> Dict[str, Any]:
        """Predict narrative state at future time.
        
        Args:
            storyline_id: Storyline to predict
            graph: Current graph state
            time_t: Current time
            future_time: Time to predict at
            
        Returns:
            Dict with predicted state, confidence, and key changes
        """
        storyline = graph.get_storyline(storyline_id)
        if not storyline:
            return {"error": "Storyline not found"}
        
        # Predict next chapter
        prediction = self.predict_next_chapter(storyline, graph, time_t)
        
        # Extrapolate to future time
        time_delta = future_time - time_t
        if time_delta > self.prediction_horizon:
            # Beyond horizon, reduce confidence
            confidence_decay = self.prediction_horizon / time_delta
            prediction["confidence"] *= confidence_decay
        
        # Identify key changes
        current_nodes = graph.get_nodes_in_storyline(storyline_id)
        predicted_events = prediction.get("predicted_events", [])
        
        key_changes = []
        for event in predicted_events:
            if event.get("time", float('inf')) <= future_time:
                key_changes.append({
                    "node_id": event.get("node_id"),
                    "change_type": event.get("type"),
                    "description": event.get("description")
                })
        
        return {
            "predicted_state": {
                "storyline_id": storyline_id,
                "time": future_time,
                "theme": storyline.theme,
                "predicted_coherence": prediction.get("coherence_score", 0.0),
                "key_changes": key_changes
            },
            "confidence": prediction.get("confidence", 0.0),
            "key_changes": key_changes
        }

