"""Training evaluation and metrics export.

This module provides functions to evaluate training results and
export metrics to Glean Catalog, including pattern-specific evaluation
and ARC-AGI style intelligence metrics.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .intelligence_metrics import DomainIntelligenceEvaluator

logger = logging.getLogger(__name__)


def evaluate_training_results(
    model_metrics: Dict[str, Any],
    training_context: Optional[Dict[str, Any]] = None,
    checkpoint_path: Optional[str] = None,
    model_predictions: Optional[Dict[str, Any]] = None,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    enable_intelligence_metrics: bool = True
) -> Dict[str, Any]:
    """Evaluate training results and generate evaluation metrics.
    
    Args:
        model_metrics: Model training metrics (loss, accuracy, etc.)
        training_context: Optional training context (graph data, patterns, etc.)
        checkpoint_path: Optional path to model checkpoint
        model_predictions: Optional model predictions for test cases
        test_cases: Optional test cases for intelligence evaluation
        enable_intelligence_metrics: Whether to evaluate intelligence metrics
    
    Returns:
        Dictionary with evaluation results:
        - model_metrics: Original model metrics
        - training_quality: Quality assessment of training
        - pattern_coverage: How well patterns were learned
        - pattern_specific_metrics: Detailed pattern evaluation metrics
        - intelligence_metrics: ARC-AGI style intelligence metrics
        - learning_rate: Learning rate metrics
        - recommendations: Recommendations for improvement
    """
    logger.info("Evaluating training results with pattern-specific and intelligence metrics")
    
    evaluation = {
        "model_metrics": model_metrics,
        "evaluated_at": datetime.now().isoformat(),
        "training_quality": {},
        "pattern_coverage": {},
        "pattern_specific_metrics": {},
        "intelligence_metrics": None,
        "learning_rate": None,
        "recommendations": [],
    }
    
    # Assess training quality
    if model_metrics:
        loss = model_metrics.get("loss")
        accuracy = model_metrics.get("accuracy")
        
        if loss is not None:
            if loss < 0.1:
                evaluation["training_quality"]["loss_assessment"] = "excellent"
            elif loss < 0.3:
                evaluation["training_quality"]["loss_assessment"] = "good"
            elif loss < 0.5:
                evaluation["training_quality"]["loss_assessment"] = "fair"
            else:
                evaluation["training_quality"]["loss_assessment"] = "poor"
                evaluation["recommendations"].append("Consider reducing learning rate or increasing training steps")
        
        if accuracy is not None:
            if accuracy > 0.9:
                evaluation["training_quality"]["accuracy_assessment"] = "excellent"
            elif accuracy > 0.7:
                evaluation["training_quality"]["accuracy_assessment"] = "good"
            elif accuracy > 0.5:
                evaluation["training_quality"]["accuracy_assessment"] = "fair"
            else:
                evaluation["training_quality"]["accuracy_assessment"] = "poor"
                evaluation["recommendations"].append("Consider increasing training data or adjusting model architecture")
    
    # Assess pattern coverage and pattern-specific metrics
    if training_context:
        learned_patterns = training_context.get("learned_patterns")
        if learned_patterns:
            column_patterns = learned_patterns.get("column_patterns", {})
            relationship_patterns = learned_patterns.get("relationship_patterns", {})
            temporal_patterns = learned_patterns.get("temporal_patterns")
            
            evaluation["pattern_coverage"] = {
                "column_types": column_patterns.get("unique_types", 0),
                "relationship_types": relationship_patterns.get("unique_labels", 0),
                "has_historical_data": training_context.get("glean_data") is not None,
                "has_temporal_patterns": temporal_patterns is not None,
            }
            
            # Pattern-specific evaluation
            pattern_metrics = _evaluate_pattern_specific_metrics(
                learned_patterns,
                training_context.get("graph_data")
            )
            evaluation["pattern_specific_metrics"] = pattern_metrics
            
            # Calculate learning rate
            if training_context.get("learned_patterns_history"):
                from .intelligence_metrics import DomainIntelligenceEvaluator
                evaluator = DomainIntelligenceEvaluator()
                learning_rate = evaluator._calculate_learning_rate(training_context)
                evaluation["learning_rate"] = learning_rate
        
        # Intelligence metrics evaluation
        if enable_intelligence_metrics and model_predictions and test_cases:
            try:
                evaluator = DomainIntelligenceEvaluator()
                intelligence = evaluator.evaluate_domain_intelligence(
                    model_predictions=model_predictions,
                    test_cases=test_cases,
                    training_context=training_context
                )
                evaluation["intelligence_metrics"] = intelligence
                logger.info(f"Intelligence level: {intelligence.get('intelligence_level')}, Domain expertise: {intelligence.get('domain_expertise', 0):.2f}")
            except Exception as e:
                logger.warning(f"Intelligence evaluation failed: {e}")
                evaluation["intelligence_metrics"] = {"error": str(e)}
        
        # Generate test cases if not provided but intelligence evaluation enabled
        if enable_intelligence_metrics and not test_cases and training_context:
            try:
                evaluator = DomainIntelligenceEvaluator()
                graph_data = training_context.get("graph_data", {})
                if graph_data:
                    test_cases = evaluator.create_domain_test_cases(
                        knowledge_graph=graph_data,
                        learned_patterns=learned_patterns or {}
                    )
                    if test_cases and model_predictions:
                        intelligence = evaluator.evaluate_domain_intelligence(
                            model_predictions=model_predictions,
                            test_cases=test_cases,
                            training_context=training_context
                        )
                        evaluation["intelligence_metrics"] = intelligence
            except Exception as e:
                logger.warning(f"Auto-generated intelligence evaluation failed: {e}")
    
    # Add checkpoint information
    if checkpoint_path:
        evaluation["checkpoint_path"] = checkpoint_path
        evaluation["checkpoint_exists"] = os.path.exists(checkpoint_path) if checkpoint_path else False
    
    logger.info(f"Evaluation completed: {len(evaluation['recommendations'])} recommendations")
    
    return evaluation


def _evaluate_pattern_specific_metrics(
    learned_patterns: Dict[str, Any],
    graph_data: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Evaluate pattern-specific metrics.
    
    Args:
        learned_patterns: Learned patterns from training
        graph_data: Knowledge graph data
    
    Returns:
        Dictionary with pattern-specific metrics:
        - column_type_accuracy: Accuracy of column type predictions
        - relationship_accuracy: Accuracy of relationship predictions
        - temporal_pattern_accuracy: Accuracy of temporal pattern predictions
        - pattern_coverage_score: How many source patterns were learned
    """
    metrics = {
        "column_type_accuracy": 0.0,
        "relationship_accuracy": 0.0,
        "temporal_pattern_accuracy": 0.0,
        "pattern_coverage_score": 0.0,
    }
    
    # Column type patterns
    column_patterns = learned_patterns.get("column_patterns", {})
    if column_patterns and graph_data:
        nodes = graph_data.get("nodes", [])
        column_nodes = [n for n in nodes if n.get("type") == "Column"]
        
        if column_nodes:
            learned_types = set(column_patterns.get("unique_types", []))
            actual_types = set()
            for node in column_nodes:
                props = node.get("properties", {})
                if "type" in props:
                    actual_types.add(props["type"])
            
            if actual_types:
                coverage = len(learned_types & actual_types) / len(actual_types)
                metrics["column_type_accuracy"] = coverage
                metrics["pattern_coverage_score"] += coverage * 0.4  # 40% weight
    
    # Relationship patterns
    relationship_patterns = learned_patterns.get("relationship_patterns", {})
    if relationship_patterns and graph_data:
        edges = graph_data.get("edges", [])
        
        if edges:
            learned_labels = set(relationship_patterns.get("unique_labels", []))
            actual_labels = set(e.get("label") for e in edges if e.get("label"))
            
            if actual_labels:
                coverage = len(learned_labels & actual_labels) / len(actual_labels)
                metrics["relationship_accuracy"] = coverage
                metrics["pattern_coverage_score"] += coverage * 0.4  # 40% weight
    
    # Temporal patterns
    temporal_patterns = learned_patterns.get("temporal_patterns")
    if temporal_patterns:
        # Simple temporal pattern evaluation
        has_evolution = bool(temporal_patterns.get("evolution_patterns"))
        has_temporal_metrics = bool(temporal_patterns.get("temporal_metrics"))
        
        if has_evolution or has_temporal_metrics:
            metrics["temporal_pattern_accuracy"] = 0.8  # Good if patterns exist
            metrics["pattern_coverage_score"] += 0.2  # 20% weight
    
    return metrics


def export_training_metrics_to_glean(
    evaluation: Dict[str, Any],
    glean_client: Optional[Any] = None,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Export training evaluation metrics to Glean Catalog.
    
    Args:
        evaluation: Training evaluation results
        glean_client: Optional GleanTrainingClient instance
        output_dir: Optional directory to save metrics file
    
    Returns:
        Dictionary with export information
    """
    logger.info("Exporting training metrics to Glean")
    
    # Create export manifest
    export_manifest = {
        "type": "training_metrics",
        "evaluated_at": evaluation.get("evaluated_at"),
        "model_metrics": evaluation.get("model_metrics", {}),
        "training_quality": evaluation.get("training_quality", {}),
        "pattern_coverage": evaluation.get("pattern_coverage", {}),
        "recommendations": evaluation.get("recommendations", []),
        "checkpoint_path": evaluation.get("checkpoint_path"),
    }
    
    export_info = {
        "exported_at": datetime.now().isoformat(),
        "manifest": export_manifest,
    }
    
    # Save to file if output_dir provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        metrics_file = os.path.join(output_dir, f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(metrics_file, 'w') as f:
            json.dump(export_info, f, indent=2)
        
        export_info["output_file"] = metrics_file
        logger.info(f"Saved training metrics to {metrics_file}")
    
    # Export to Glean Catalog if client provided
    if glean_client:
        try:
            glean_export = _export_to_glean_catalog(
                evaluation,
                glean_client,
                output_dir
            )
            export_info["glean_export"] = glean_export
            logger.info("Training metrics exported to Glean Catalog")
        except Exception as e:
            logger.warning(f"Glean export failed: {e}")
            export_info["glean_export_error"] = str(e)
    
    return export_info


def _export_to_glean_catalog(
    evaluation: Dict[str, Any],
    glean_client: Any,
    output_dir: Optional[str]
) -> Dict[str, Any]:
    """Export training metrics to Glean Catalog.
    
    Args:
        evaluation: Training evaluation results
        glean_client: GleanTrainingClient instance
        output_dir: Optional directory for Glean batch files
    
    Returns:
        Dictionary with Glean export information
    """
    import subprocess
    import tempfile
    
    # Create Glean batch file
    batch_data = {
        "facts": [
            {
                "predicate": "agenticAiETH.ETL.TrainingMetrics.1",
                "key": {
                    "evaluated_at": evaluation.get("evaluated_at"),
                    "model_metrics": evaluation.get("model_metrics", {}),
                    "training_quality": evaluation.get("training_quality", {}),
                    "pattern_coverage": evaluation.get("pattern_coverage", {}),
                    "pattern_specific_metrics": evaluation.get("pattern_specific_metrics", {}),
                    "intelligence_level": evaluation.get("intelligence_metrics", {}).get("intelligence_level", 0),
                    "domain_expertise": evaluation.get("intelligence_metrics", {}).get("domain_expertise", 0.0),
                    "learning_rate": evaluation.get("learning_rate", {}),
                }
            }
        ]
    }
    
    # Save batch file
    if output_dir:
        batch_dir = os.path.join(output_dir, "glean_batches")
        os.makedirs(batch_dir, exist_ok=True)
    else:
        batch_dir = tempfile.mkdtemp()
    
    batch_file = os.path.join(
        batch_dir,
        f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(batch_file, 'w') as f:
        json.dump(batch_data, f, indent=2)
    
    # Export to Glean using glean write command
    try:
        db_name = getattr(glean_client, 'db_name', None)
        if db_name:
            result = subprocess.run(
                ["glean", "write", "--db", db_name, batch_file],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "batch_file": batch_file,
                    "glean_output": result.stdout
                }
            else:
                return {
                    "status": "error",
                    "batch_file": batch_file,
                    "error": result.stderr
                }
        else:
            return {
                "status": "skipped",
                "reason": "Glean DB name not configured",
                "batch_file": batch_file
            }
    except FileNotFoundError:
        return {
            "status": "error",
            "error": "glean command not found in PATH",
            "batch_file": batch_file
        }
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": "Glean export timeout",
            "batch_file": batch_file
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "batch_file": batch_file
        }

