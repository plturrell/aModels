"""Performance benchmarking utilities for narrative system."""

import time
import logging
from typing import Dict, List, Any, Optional
from statistics import mean, stdev

from ..narrative import MultiPurposeNarrativeGNN, NarrativeGraph
from ..evaluation.metrics import (
    evaluate_explanation_quality,
    evaluate_prediction_accuracy,
    evaluate_anomaly_detection
)

logger = logging.getLogger(__name__)


def benchmark_explanation_generation(
    gnn: MultiPurposeNarrativeGNN,
    graph: NarrativeGraph,
    time_points: List[float],
    storyline_ids: List[str],
    reference_explanations: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Benchmark explanation generation performance.
    
    Args:
        gnn: MultiPurposeNarrativeGNN instance
        graph: Narrative graph
        time_points: List of time points to test
        storyline_ids: List of storyline IDs to test
        reference_explanations: Optional reference explanations for quality evaluation
        
    Returns:
        Dict with benchmark results
    """
    logger.info(f"Benchmarking explanation generation on {len(time_points)} time points")
    
    results = {
        "runtimes": [],
        "quality_scores": [],
        "num_explanations": 0
    }
    
    for time_t in time_points:
        for storyline_id in storyline_ids:
            start_time = time.time()
            
            result = gnn.forward(
                graph, current_time=time_t, task_mode="explain", storyline_id=storyline_id
            )
            
            runtime = time.time() - start_time
            results["runtimes"].append(runtime)
            
            if "explanation" in result:
                results["num_explanations"] += 1
                
                # Evaluate quality if reference provided
                if reference_explanations:
                    ref_key = f"{storyline_id}_{time_t}"
                    if ref_key in reference_explanations:
                        quality = evaluate_explanation_quality(
                            result["explanation"],
                            reference_explanations[ref_key]
                        )
                        results["quality_scores"].append(quality.get("overall_quality", 0.0))
    
    # Compute statistics
    if results["runtimes"]:
        results["avg_runtime"] = mean(results["runtimes"])
        results["std_runtime"] = stdev(results["runtimes"]) if len(results["runtimes"]) > 1 else 0.0
        results["min_runtime"] = min(results["runtimes"])
        results["max_runtime"] = max(results["runtimes"])
    else:
        results["avg_runtime"] = 0.0
        results["std_runtime"] = 0.0
        results["min_runtime"] = 0.0
        results["max_runtime"] = 0.0
    
    if results["quality_scores"]:
        results["avg_quality"] = mean(results["quality_scores"])
        results["std_quality"] = stdev(results["quality_scores"]) if len(results["quality_scores"]) > 1 else 0.0
    else:
        results["avg_quality"] = 0.0
        results["std_quality"] = 0.0
    
    logger.info(
        f"Explanation benchmark complete: "
        f"avg_runtime={results['avg_runtime']:.3f}s, "
        f"avg_quality={results['avg_quality']:.3f}"
    )
    
    return results


def benchmark_prediction_accuracy(
    gnn: MultiPurposeNarrativeGNN,
    graph: NarrativeGraph,
    time_points: List[float],
    storyline_ids: List[str],
    ground_truth_events: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """Benchmark prediction accuracy.
    
    Args:
        gnn: MultiPurposeNarrativeGNN instance
        graph: Narrative graph
        time_points: List of time points to test
        storyline_ids: List of storyline IDs to test
        ground_truth_events: Optional ground truth events for accuracy evaluation
        
    Returns:
        Dict with benchmark results
    """
    logger.info(f"Benchmarking prediction accuracy on {len(time_points)} time points")
    
    results = {
        "runtimes": [],
        "accuracy_scores": [],
        "num_predictions": 0
    }
    
    for time_t in time_points:
        for storyline_id in storyline_ids:
            start_time = time.time()
            
            result = gnn.forward(
                graph, current_time=time_t, task_mode="predict", storyline_id=storyline_id
            )
            
            runtime = time.time() - start_time
            results["runtimes"].append(runtime)
            
            if "prediction" in result:
                results["num_predictions"] += 1
                
                predicted_events = result["prediction"].get("predicted_events", [])
                
                # Evaluate accuracy if ground truth provided
                if ground_truth_events:
                    truth_key = f"{storyline_id}_{time_t}"
                    if truth_key in ground_truth_events:
                        accuracy = evaluate_prediction_accuracy(
                            predicted_events,
                            ground_truth_events[truth_key]
                        )
                        results["accuracy_scores"].append(accuracy.get("f1", 0.0))
    
    # Compute statistics
    if results["runtimes"]:
        results["avg_runtime"] = mean(results["runtimes"])
        results["std_runtime"] = stdev(results["runtimes"]) if len(results["runtimes"]) > 1 else 0.0
    else:
        results["avg_runtime"] = 0.0
        results["std_runtime"] = 0.0
    
    if results["accuracy_scores"]:
        results["avg_accuracy"] = mean(results["accuracy_scores"])
        results["std_accuracy"] = stdev(results["accuracy_scores"]) if len(results["accuracy_scores"]) > 1 else 0.0
    else:
        results["avg_accuracy"] = 0.0
        results["std_accuracy"] = 0.0
    
    logger.info(
        f"Prediction benchmark complete: "
        f"avg_runtime={results['avg_runtime']:.3f}s, "
        f"avg_accuracy={results['avg_accuracy']:.3f}"
    )
    
    return results


def benchmark_anomaly_detection(
    gnn: MultiPurposeNarrativeGNN,
    graph: NarrativeGraph,
    time_points: List[float],
    storyline_ids: List[str],
    ground_truth_anomalies: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Any]:
    """Benchmark anomaly detection performance.
    
    Args:
        gnn: MultiPurposeNarrativeGNN instance
        graph: Narrative graph
        time_points: List of time points to test
        storyline_ids: List of storyline IDs to test
        ground_truth_anomalies: Optional ground truth anomalies for evaluation
        
    Returns:
        Dict with benchmark results
    """
    logger.info(f"Benchmarking anomaly detection on {len(time_points)} time points")
    
    results = {
        "runtimes": [],
        "detection_scores": [],
        "num_detections": 0
    }
    
    for time_t in time_points:
        for storyline_id in storyline_ids:
            start_time = time.time()
            
            result = gnn.forward(
                graph, current_time=time_t, task_mode="detect_anomalies", storyline_id=storyline_id
            )
            
            runtime = time.time() - start_time
            results["runtimes"].append(runtime)
            
            if "anomalies" in result:
                detected_anomalies = result["anomalies"]
                results["num_detections"] += len(detected_anomalies)
                
                # Evaluate detection if ground truth provided
                if ground_truth_anomalies:
                    truth_key = f"{storyline_id}_{time_t}"
                    if truth_key in ground_truth_anomalies:
                        detection = evaluate_anomaly_detection(
                            detected_anomalies,
                            ground_truth_anomalies[truth_key]
                        )
                        results["detection_scores"].append(detection.get("f1", 0.0))
    
    # Compute statistics
    if results["runtimes"]:
        results["avg_runtime"] = mean(results["runtimes"])
        results["std_runtime"] = stdev(results["runtimes"]) if len(results["runtimes"]) > 1 else 0.0
    else:
        results["avg_runtime"] = 0.0
        results["std_runtime"] = 0.0
    
    if results["detection_scores"]:
        results["avg_detection"] = mean(results["detection_scores"])
        results["std_detection"] = stdev(results["detection_scores"]) if len(results["detection_scores"]) > 1 else 0.0
    else:
        results["avg_detection"] = 0.0
        results["std_detection"] = 0.0
    
    logger.info(
        f"Anomaly detection benchmark complete: "
        f"avg_runtime={results['avg_runtime']:.3f}s, "
        f"avg_detection={results['avg_detection']:.3f}"
    )
    
    return results


def benchmark_runtime_performance(
    gnn: MultiPurposeNarrativeGNN,
    graph: NarrativeGraph,
    num_iterations: int = 100,
    task_modes: List[str] = ["explain", "predict", "detect_anomalies"]
) -> Dict[str, Any]:
    """Benchmark runtime performance (messages/sec).
    
    Args:
        gnn: MultiPurposeNarrativeGNN instance
        graph: Narrative graph
        num_iterations: Number of iterations to run
        task_modes: List of task modes to benchmark
        
    Returns:
        Dict with performance metrics
    """
    logger.info(f"Benchmarking runtime performance: {num_iterations} iterations")
    
    results = {}
    
    for task_mode in task_modes:
        runtimes = []
        
        # Get a storyline ID
        storyline_id = None
        if graph.storylines:
            storyline_id = list(graph.storylines.keys())[0]
        
        for _ in range(num_iterations):
            start_time = time.time()
            
            gnn.forward(
                graph,
                current_time=10.0,
                task_mode=task_mode,
                storyline_id=storyline_id
            )
            
            runtime = time.time() - start_time
            runtimes.append(runtime)
        
        # Compute throughput
        avg_runtime = mean(runtimes)
        messages_per_sec = 1.0 / avg_runtime if avg_runtime > 0 else 0.0
        
        results[task_mode] = {
            "avg_runtime": avg_runtime,
            "std_runtime": stdev(runtimes) if len(runtimes) > 1 else 0.0,
            "messages_per_sec": messages_per_sec,
            "min_runtime": min(runtimes),
            "max_runtime": max(runtimes)
        }
    
    logger.info(f"Runtime benchmark complete: {results}")
    
    return results

