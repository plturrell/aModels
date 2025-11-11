"""Evaluation metrics for narrative system capabilities."""

import logging
from typing import Dict, List, Optional, Any, Tuple
import re

logger = logging.getLogger(__name__)


def evaluate_explanation_quality(
    generated_explanation: str,
    reference_explanation: Optional[str] = None,
    key_entities: Optional[List[str]] = None
) -> Dict[str, float]:
    """Evaluate quality of generated explanation.
    
    Args:
        generated_explanation: Generated explanation text
        reference_explanation: Optional reference explanation for comparison
        key_entities: Optional list of key entities that should be mentioned
        
    Returns:
        Dict with quality metrics
    """
    metrics = {}
    
    # Length check (should be substantial)
    metrics["length"] = len(generated_explanation)
    metrics["length_score"] = min(1.0, metrics["length"] / 200.0)  # Normalize to 200 chars
    
    # Entity coverage (if key entities provided)
    if key_entities:
        mentioned_entities = []
        for entity in key_entities:
            if entity.lower() in generated_explanation.lower():
                mentioned_entities.append(entity)
        metrics["entity_coverage"] = len(mentioned_entities) / len(key_entities) if key_entities else 0.0
    else:
        metrics["entity_coverage"] = 0.5  # Neutral if no entities provided
    
    # Causal language detection (words like "because", "led to", "caused")
    causal_words = ["because", "due to", "led to", "caused", "resulted in", "therefore", "as a result"]
    causal_count = sum(1 for word in causal_words if word in generated_explanation.lower())
    metrics["causal_language"] = min(1.0, causal_count / 3.0)  # Normalize to 3 occurrences
    
    # Structure check (should have some structure)
    has_structure = any(marker in generated_explanation for marker in [".", ",", "and", "but"])
    metrics["structure_score"] = 1.0 if has_structure else 0.0
    
    # ROUGE-like score if reference provided
    if reference_explanation:
        metrics["rouge_like"] = _compute_rouge_like(generated_explanation, reference_explanation)
    else:
        metrics["rouge_like"] = 0.0
    
    # Overall quality score (weighted average)
    metrics["overall_quality"] = (
        metrics["length_score"] * 0.2 +
        metrics["entity_coverage"] * 0.3 +
        metrics["causal_language"] * 0.3 +
        metrics["structure_score"] * 0.1 +
        metrics["rouge_like"] * 0.1
    )
    
    return metrics


def _compute_rouge_like(text1: str, text2: str) -> float:
    """Simple ROUGE-like score (word overlap).
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Overlap score (0-1)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def evaluate_prediction_accuracy(
    predicted_events: List[Dict[str, Any]],
    actual_events: List[Dict[str, Any]],
    time_tolerance: float = 5.0
) -> Dict[str, float]:
    """Evaluate prediction accuracy.
    
    Args:
        predicted_events: List of predicted events
        actual_events: List of actual events
        time_tolerance: Time tolerance for matching events
        
    Returns:
        Dict with accuracy metrics (precision, recall, F1)
    """
    if not predicted_events:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_predicted": 0,
            "num_actual": len(actual_events)
        }
    
    if not actual_events:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_predicted": len(predicted_events),
            "num_actual": 0
        }
    
    # Match predicted to actual events
    matched = 0
    for pred_event in predicted_events:
        pred_time = pred_event.get("time", 0.0)
        pred_node = pred_event.get("node_id")
        
        for actual_event in actual_events:
            actual_time = actual_event.get("time", 0.0)
            actual_node = actual_event.get("node_id")
            
            # Check if match (same node, close time)
            if pred_node == actual_node and abs(pred_time - actual_time) <= time_tolerance:
                matched += 1
                break
    
    precision = matched / len(predicted_events) if predicted_events else 0.0
    recall = matched / len(actual_events) if actual_events else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_predicted": len(predicted_events),
        "num_actual": len(actual_events),
        "num_matched": matched
    }


def evaluate_anomaly_detection(
    detected_anomalies: List[Dict[str, Any]],
    ground_truth_anomalies: List[Dict[str, Any]],
    time_tolerance: float = 5.0
) -> Dict[str, float]:
    """Evaluate anomaly detection performance.
    
    Args:
        detected_anomalies: List of detected anomalies
        ground_truth_anomalies: List of actual anomalies
        time_tolerance: Time tolerance for matching
        
    Returns:
        Dict with detection metrics (precision, recall, F1)
    """
    if not detected_anomalies:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_detected": 0,
            "num_actual": len(ground_truth_anomalies)
        }
    
    if not ground_truth_anomalies:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "num_detected": len(detected_anomalies),
            "num_actual": 0
        }
    
    # Match detected to ground truth
    matched = 0
    for detected in detected_anomalies:
        detected_event = detected.get("event", {})
        detected_time = detected_event.get("time", 0.0)
        detected_node = detected_event.get("node_id")
        
        for truth in ground_truth_anomalies:
            truth_event = truth.get("event", {})
            truth_time = truth_event.get("time", 0.0)
            truth_node = truth_event.get("node_id")
            
            if detected_node == truth_node and abs(detected_time - truth_time) <= time_tolerance:
                matched += 1
                break
    
    precision = matched / len(detected_anomalies) if detected_anomalies else 0.0
    recall = matched / len(ground_truth_anomalies) if ground_truth_anomalies else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_detected": len(detected_anomalies),
        "num_actual": len(ground_truth_anomalies),
        "num_matched": matched
    }


def compute_narrative_coherence(
    storyline,
    graph
) -> Dict[str, float]:
    """Compute narrative coherence metrics.
    
    Args:
        storyline: Storyline instance
        graph: NarrativeGraph instance
        
    Returns:
        Dict with coherence metrics
    """
    # Get coherence from storyline
    coherence_metrics = storyline.coherence_metrics.copy()
    
    # Additional metrics
    causal_links = storyline.causal_links
    key_events = storyline.key_events
    
    # Causal chain consistency
    causal_consistency = 1.0
    if len(causal_links) > 1:
        # Check if chain is connected
        for i in range(len(causal_links) - 1):
            _, target, _ = causal_links[i]
            next_source, _, _ = causal_links[i + 1]
            if target != next_source:
                causal_consistency -= 0.2  # Penalty for gaps
        causal_consistency = max(0.0, causal_consistency)
    
    coherence_metrics["causal_chain_consistency"] = causal_consistency
    
    # Temporal ordering
    if key_events:
        times = [e.get("time", 0.0) for e in key_events]
        is_ordered = all(times[i] <= times[i+1] for i in range(len(times)-1))
        coherence_metrics["temporal_ordering"] = 1.0 if is_ordered else 0.5
    else:
        coherence_metrics["temporal_ordering"] = 0.5
    
    # Overall coherence
    coherence_metrics["overall_coherence"] = sum(coherence_metrics.values()) / len(coherence_metrics)
    
    return coherence_metrics

