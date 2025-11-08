"""Anomaly Detection: Identify narrative violations and inconsistencies."""

import logging
from typing import Dict, List, Optional, Any, Tuple

from .narrative_graph import NarrativeGraph
from .storyline import Storyline
from .narrative_node import NarrativeNode
from .narrative_edge import NarrativeEdge

logger = logging.getLogger(__name__)


class NarrativeAnomalyDetector:
    """Detects anomalies and violations in narrative patterns."""
    
    def __init__(self, narrative_graph: Optional[NarrativeGraph] = None):
        """Initialize anomaly detector.
        
        Args:
            narrative_graph: Optional narrative graph
        """
        self.narrative_graph = narrative_graph
        logger.info("Initialized NarrativeAnomalyDetector")
    
    def detect_story_violations(
        self,
        observed_events: List[Dict[str, Any]],
        expected_storyline: Storyline,
        narrative_graph: Optional[NarrativeGraph] = None
    ) -> List[Dict[str, Any]]:
        """Detect events that violate expected narrative patterns.
        
        Args:
            observed_events: List of observed events
            expected_storyline: Expected storyline
            narrative_graph: Narrative graph (uses self.narrative_graph if None)
            
        Returns:
            List of anomaly dicts with type, event, and severity
        """
        graph = narrative_graph or self.narrative_graph
        if not graph:
            raise ValueError("Narrative graph required")
        
        anomalies = []
        
        for event in observed_events:
            # Check character arc consistency
            if self._breaks_character_arc(event, expected_storyline, graph):
                anomalies.append({
                    "type": "character_inconsistency",
                    "event": event,
                    "severity": "high",
                    "description": f"Event breaks character arc for {event.get('node_id', 'unknown')}"
                })
            
            # Check causal plausibility
            if self._has_implausible_cause(event, expected_storyline, graph):
                anomalies.append({
                    "type": "causal_violation",
                    "event": event,
                    "severity": "medium",
                    "description": f"Event has implausible causal chain: {event.get('description', 'unknown')}"
                })
            
            # Check plot coherence
            if self._disrupts_narrative_flow(event, expected_storyline, graph):
                anomalies.append({
                    "type": "plot_disruption",
                    "event": event,
                    "severity": "medium",
                    "description": f"Event disrupts narrative flow: {event.get('description', 'unknown')}"
                })
        
        logger.info(f"Detected {len(anomalies)} anomalies in storyline {expected_storyline.storyline_id}")
        return anomalies
    
    def _breaks_character_arc(
        self,
        event: Dict[str, Any],
        storyline: Storyline,
        graph: NarrativeGraph
    ) -> bool:
        """Check if event breaks character arc consistency.
        
        Args:
            event: Event to check
            storyline: Expected storyline
            graph: Narrative graph
            
        Returns:
            True if breaks character arc
        """
        node_id = event.get("node_id")
        if not node_id or node_id not in graph.nodes:
            return False
        
        node = graph.nodes[node_id]
        if not isinstance(node, NarrativeNode):
            return False
        
        # Get node's role in storyline
        role_info = node.get_narrative_role(storyline.storyline_id)
        if not role_info:
            return False  # No role defined, can't break it
        
        role = role_info.get("role")
        arc_phase = role_info.get("arc_phase")
        event_type = event.get("type")
        
        # Check if event type is inconsistent with role
        if role == "protagonist":
            # Protagonist should have positive/forward-moving events
            if event_type in ["reversal", "decline", "failure"]:
                # Check if we're in falling/resolution phase
                if arc_phase not in ["falling", "resolution"]:
                    return True  # Premature decline
        
        elif role == "antagonist":
            # Antagonist can have various events, but should maintain opposition
            if event_type == "cooperation" and arc_phase in ["rising", "climax"]:
                return True  # Unexpected cooperation
        
        return False
    
    def _has_implausible_cause(
        self,
        event: Dict[str, Any],
        storyline: Storyline,
        graph: NarrativeGraph
    ) -> bool:
        """Check if event has implausible causal chain.
        
        Args:
            event: Event to check
            storyline: Expected storyline
            graph: Narrative graph
            
        Returns:
            True if has implausible cause
        """
        node_id = event.get("node_id")
        if not node_id or node_id not in graph.nodes:
            return True  # Node doesn't exist = implausible
        
        # Get causal chain leading to this node
        causal_chain = graph.build_causal_chain(storyline.storyline_id, end_node_id=node_id)
        
        if not causal_chain:
            # No causal chain = potentially implausible
            # But might be a new actor, so not always anomalous
            return False
        
        # Check if causal chain is coherent
        # Look for gaps or inconsistencies
        for i in range(len(causal_chain) - 1):
            source_id, target_id, rel_type = causal_chain[i]
            next_source, next_target, next_rel = causal_chain[i + 1]
            
            # Check if chain is connected
            if target_id != next_source:
                return True  # Gap in causal chain
        
        # Check edge causal strengths
        for source_id, target_id, rel_type in causal_chain:
            # Find edge
            edge = None
            for e in graph.edges:
                if isinstance(e, NarrativeEdge):
                    if e.source_id == source_id and e.target_id == target_id:
                        edge = e
                        break
            
            if edge and edge.causal_strength < 0.3:
                return True  # Weak causal link
        
        return False
    
    def _disrupts_narrative_flow(
        self,
        event: Dict[str, Any],
        storyline: Storyline,
        graph: NarrativeGraph
    ) -> bool:
        """Check if event disrupts narrative flow.
        
        Args:
            event: Event to check
            storyline: Expected storyline
            graph: Narrative graph
            
        Returns:
            True if disrupts flow
        """
        # Check coherence metrics
        coherence = storyline.get_overall_coherence()
        
        # If event would significantly reduce coherence
        event_time = event.get("time", 0.0)
        
        # Check if event is out of sequence with key events
        key_events = storyline.key_events
        if key_events:
            # Check temporal ordering
            last_key_event_time = max(e.get("time", 0.0) for e in key_events)
            if event_time < last_key_event_time - 10.0:  # Too far in past
                return True
        
        # Check if event contradicts established patterns
        node_id = event.get("node_id")
        if node_id and node_id in graph.nodes:
            node = graph.nodes[node_id]
            if isinstance(node, NarrativeNode):
                # Check if event contradicts node's established influence
                influence = node.get_causal_influence(storyline.storyline_id)
                event_type = event.get("type")
                
                if influence > 0.7 and event_type in ["minor", "insignificant"]:
                    return True  # High-influence node with minor event
        
        return False
    
    def score_anomaly(
        self,
        event: Dict[str, Any],
        storyline: Storyline,
        graph: NarrativeGraph
    ) -> float:
        """Score how anomalous an event is.
        
        Args:
            event: Event to score
            storyline: Expected storyline
            graph: Narrative graph
            
        Returns:
            Anomaly score (0-1, higher = more anomalous)
        """
        score = 0.0
        
        # Character inconsistency
        if self._breaks_character_arc(event, storyline, graph):
            score += 0.4
        
        # Causal violation
        if self._has_implausible_cause(event, storyline, graph):
            score += 0.3
        
        # Plot disruption
        if self._disrupts_narrative_flow(event, storyline, graph):
            score += 0.3
        
        return min(1.0, score)
    
    def detect_narrative_anomalies(
        self,
        storyline_id: str,
        graph: NarrativeGraph,
        time_t: float,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in a storyline at a specific time.
        
        Args:
            storyline_id: Storyline to check
            graph: Narrative graph
            time_t: Time point to check
            threshold: Anomaly score threshold
            
        Returns:
            List of detected anomalies
        """
        storyline = graph.get_storyline(storyline_id)
        if not storyline:
            return []
        
        # Get events at this time
        # In practice, would extract from graph state
        # For now, use key events near this time
        events = [
            e for e in storyline.key_events
            if abs(e.get("time", 0.0) - time_t) < 5.0
        ]
        
        anomalies = []
        for event in events:
            anomaly_score = self.score_anomaly(event, storyline, graph)
            if anomaly_score >= threshold:
                anomalies.append({
                    "event": event,
                    "anomaly_score": anomaly_score,
                    "storyline_id": storyline_id,
                    "time": time_t
                })
        
        return anomalies

