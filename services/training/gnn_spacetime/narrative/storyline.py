"""Storyline data structure for narrative intelligence.

Represents a coherent narrative thread across the temporal graph.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class NarrativeType(Enum):
    """Types of narratives."""
    EXPLANATION = "explanation"
    PREDICTION = "prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    GENERAL = "general"


class Storyline:
    """Represents a coherent narrative thread in the temporal graph.
    
    Tracks narrative coherence, explanatory quality, predictive confidence,
    and anomaly scores for different use cases.
    """
    
    def __init__(
        self,
        storyline_id: str,
        theme: str,
        narrative_type: NarrativeType = NarrativeType.GENERAL,
        coherence_metrics: Optional[Dict[str, float]] = None,
        explanatory_quality: Optional[float] = None,
        predictive_confidence: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        causal_links: Optional[List[Tuple[str, str, str]]] = None,
        key_events: Optional[List[Dict[str, Any]]] = None
    ):
        """Initialize storyline.
        
        Args:
            storyline_id: Unique identifier for storyline
            theme: Theme/description of the storyline
            narrative_type: Type of narrative (explanation, prediction, anomaly_detection)
            coherence_metrics: Dict with "plot_consistency", "character_development", "causal_plausibility"
            explanatory_quality: Quality score for explanations (0-1)
            predictive_confidence: Confidence in predictions (0-1)
            anomaly_score: Anomaly score (0-1, higher = more anomalous)
            causal_links: List of (source_id, target_id, relation_type) tuples
            key_events: List of key events in the storyline
        """
        self.storyline_id = storyline_id
        self.theme = theme
        self.narrative_type = narrative_type
        self.coherence_metrics = coherence_metrics or {
            "plot_consistency": 0.0,
            "character_development": 0.0,
            "causal_plausibility": 0.0
        }
        self.explanatory_quality = explanatory_quality or 0.0
        self.predictive_confidence = predictive_confidence or 0.0
        self.anomaly_score = anomaly_score or 0.0
        self.causal_links = causal_links or []
        self.key_events = key_events or []
        
        logger.info(
            f"Created Storyline {storyline_id}: {theme} "
            f"(type: {narrative_type.value})"
        )
    
    def update_coherence_metrics(
        self,
        plot_consistency: Optional[float] = None,
        character_development: Optional[float] = None,
        causal_plausibility: Optional[float] = None
    ):
        """Update coherence metrics.
        
        Args:
            plot_consistency: Plot consistency score (0-1)
            character_development: Character development score (0-1)
            causal_plausibility: Causal plausibility score (0-1)
        """
        if plot_consistency is not None:
            self.coherence_metrics["plot_consistency"] = max(0.0, min(1.0, plot_consistency))
        if character_development is not None:
            self.coherence_metrics["character_development"] = max(0.0, min(1.0, character_development))
        if causal_plausibility is not None:
            self.coherence_metrics["causal_plausibility"] = max(0.0, min(1.0, causal_plausibility))
    
    def add_causal_link(self, source_id: str, target_id: str, relation_type: str):
        """Add a causal link to the storyline.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of causal relation
        """
        link = (source_id, target_id, relation_type)
        if link not in self.causal_links:
            self.causal_links.append(link)
            logger.debug(f"Added causal link to storyline {self.storyline_id}: {source_id}->{target_id}")
    
    def add_key_event(self, event: Dict[str, Any]):
        """Add a key event to the storyline.
        
        Args:
            event: Event dict with at least "time", "node_id", "description"
        """
        self.key_events.append(event)
        # Sort by time
        self.key_events.sort(key=lambda x: x.get("time", 0.0))
        logger.debug(f"Added key event to storyline {self.storyline_id}: {event.get('description', 'unknown')}")
    
    def get_overall_coherence(self) -> float:
        """Get overall coherence score.
        
        Returns:
            Average of all coherence metrics (0-1)
        """
        if not self.coherence_metrics:
            return 0.0
        return sum(self.coherence_metrics.values()) / len(self.coherence_metrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "storyline_id": self.storyline_id,
            "theme": self.theme,
            "narrative_type": self.narrative_type.value,
            "coherence_metrics": self.coherence_metrics,
            "explanatory_quality": self.explanatory_quality,
            "predictive_confidence": self.predictive_confidence,
            "anomaly_score": self.anomaly_score,
            "causal_links": self.causal_links,
            "key_events": self.key_events
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Storyline":
        """Create Storyline from dictionary."""
        narrative_type = NarrativeType(data.get("narrative_type", "general"))
        return cls(
            storyline_id=data["storyline_id"],
            theme=data["theme"],
            narrative_type=narrative_type,
            coherence_metrics=data.get("coherence_metrics", {}),
            explanatory_quality=data.get("explanatory_quality"),
            predictive_confidence=data.get("predictive_confidence"),
            anomaly_score=data.get("anomaly_score"),
            causal_links=data.get("causal_links", []),
            key_events=data.get("key_events", [])
        )

