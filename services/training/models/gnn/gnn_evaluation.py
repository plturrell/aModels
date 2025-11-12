"""GNN Evaluation Metrics.

This module provides evaluation metrics for GNN models including classification
accuracy, link prediction precision/recall, and embedding quality metrics.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        roc_auc_score, average_precision_score,
        silhouette_score, adjusted_rand_score
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class GNNEvaluator:
    """Evaluator for GNN models.
    
    Provides metrics for:
    - Node classification
    - Link prediction
    - Embedding quality
    """
    
    def __init__(self):
        """Initialize GNN evaluator."""
        if not HAS_SKLEARN:
            logger.warning("scikit-learn not available. Some metrics will be limited.")
    
    def evaluate_classification(
        self,
        y_true: List[Any],
        y_pred: List[Any],
        y_proba: Optional[List[List[float]]] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate node classification performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            class_names: Class names for reporting (optional)
        
        Returns:
            Dictionary with classification metrics
        """
        if not HAS_SKLEARN:
            # Basic metrics without sklearn
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
            return {
                "accuracy": accuracy,
                "num_samples": len(y_true),
                "error": "scikit-learn not available for detailed metrics"
            }
        
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            
            # Per-class metrics
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            # Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0
            )
            
            # ROC AUC if probabilities available
            roc_auc = None
            if y_proba is not None and len(y_proba) > 0:
                try:
                    # Convert to binary if needed
                    if len(set(y_true)) == 2:
                        y_proba_binary = [p[1] if len(p) > 1 else p[0] for p in y_proba]
                        roc_auc = roc_auc_score(y_true, y_proba_binary)
                    else:
                        # Multi-class ROC AUC
                        y_true_binary = np.eye(len(set(y_true)))[[int(c) for c in y_true]]
                        roc_auc = roc_auc_score(y_true_binary, y_proba, multi_class='ovr', average='weighted')
                except Exception as e:
                    logger.warning(f"Failed to compute ROC AUC: {e}")
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "confusion_matrix": cm.tolist(),
                "precision_per_class": precision_per_class.tolist(),
                "recall_per_class": recall_per_class.tolist(),
                "f1_per_class": f1_per_class.tolist(),
                "classification_report": report,
                "num_samples": len(y_true),
                "num_classes": len(set(y_true))
            }
            
            if roc_auc is not None:
                metrics["roc_auc"] = float(roc_auc)
            
            logger.info(
                f"Classification metrics: accuracy={accuracy:.4f}, "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Classification evaluation failed: {e}")
            return {
                "error": str(e),
                "num_samples": len(y_true) if y_true else 0
            }
    
    def evaluate_link_prediction(
        self,
        y_true: List[bool],
        y_pred: List[bool],
        y_proba: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Evaluate link prediction performance.
        
        Args:
            y_true: True labels (True = link exists, False = no link)
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
        
        Returns:
            Dictionary with link prediction metrics
        """
        if not HAS_SKLEARN:
            # Basic metrics without sklearn
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0
            tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
            fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "num_samples": len(y_true),
                "error": "scikit-learn not available for detailed metrics"
            }
        
        try:
            # Basic metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            tp, fp, fn, tn = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # ROC AUC and PR AUC if probabilities available
            roc_auc = None
            pr_auc = None
            if y_proba is not None and len(y_proba) > 0:
                try:
                    roc_auc = roc_auc_score(y_true, y_proba)
                    pr_auc = average_precision_score(y_true, y_proba)
                except Exception as e:
                    logger.warning(f"Failed to compute AUC metrics: {e}")
            
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "confusion_matrix": cm.tolist(),
                "num_samples": len(y_true),
                "num_positive": int(sum(y_true)),
                "num_negative": int(len(y_true) - sum(y_true))
            }
            
            if roc_auc is not None:
                metrics["roc_auc"] = float(roc_auc)
            if pr_auc is not None:
                metrics["pr_auc"] = float(pr_auc)
            
            logger.info(
                f"Link prediction metrics: accuracy={accuracy:.4f}, "
                f"precision={precision:.4f}, recall={recall:.4f}, f1={f1:.4f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Link prediction evaluation failed: {e}")
            return {
                "error": str(e),
                "num_samples": len(y_true) if y_true else 0
            }
    
    def evaluate_embeddings(
        self,
        embeddings: np.ndarray,
        labels: Optional[List[Any]] = None,
        metric: str = "silhouette"
    ) -> Dict[str, Any]:
        """Evaluate embedding quality.
        
        Args:
            embeddings: Embedding vectors [num_samples, embedding_dim]
            labels: Optional labels for supervised metrics
            metric: Metric to use ('silhouette', 'inertia', 'separation')
        
        Returns:
            Dictionary with embedding quality metrics
        """
        if not HAS_SKLEARN:
            return {
                "error": "scikit-learn required for embedding evaluation",
                "num_samples": len(embeddings) if embeddings is not None else 0
            }
        
        try:
            metrics = {
                "num_samples": len(embeddings),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 1
            }
            
            # Silhouette score (requires labels)
            if labels is not None and metric == "silhouette":
                try:
                    silhouette = silhouette_score(embeddings, labels)
                    metrics["silhouette_score"] = float(silhouette)
                except Exception as e:
                    logger.warning(f"Failed to compute silhouette score: {e}")
            
            # Adjusted Rand Index (requires labels)
            if labels is not None:
                try:
                    # Need predictions for ARI - use k-means clustering
                    from sklearn.cluster import KMeans
                    n_clusters = len(set(labels))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    ari = adjusted_rand_score(labels, cluster_labels)
                    metrics["adjusted_rand_index"] = float(ari)
                    metrics["inertia"] = float(kmeans.inertia_)
                except Exception as e:
                    logger.warning(f"Failed to compute clustering metrics: {e}")
            
            # Embedding statistics
            metrics["mean_norm"] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
            metrics["std_norm"] = float(np.std(np.linalg.norm(embeddings, axis=1)))
            metrics["mean_value"] = float(np.mean(embeddings))
            metrics["std_value"] = float(np.std(embeddings))
            
            logger.info(f"Embedding quality metrics computed for {len(embeddings)} samples")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Embedding evaluation failed: {e}")
            return {
                "error": str(e),
                "num_samples": len(embeddings) if embeddings is not None else 0
            }
    
    def evaluate_schema_matching(
        self,
        similarities: List[float],
        true_matches: List[bool],
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate schema matching performance.
        
        Args:
            similarities: Similarity scores between schema pairs
            true_matches: True labels (True = match, False = no match)
            threshold: Similarity threshold for matching
        
        Returns:
            Dictionary with schema matching metrics
        """
        # Convert similarities to predictions
        y_pred = [s >= threshold for s in similarities]
        
        # Use link prediction evaluation (same problem)
        return self.evaluate_link_prediction(true_matches, y_pred, similarities)
    
    def compare_with_baseline(
        self,
        gnn_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        metric_name: str = "accuracy"
    ) -> Dict[str, Any]:
        """Compare GNN metrics with baseline.
        
        Args:
            gnn_metrics: GNN model metrics
            baseline_metrics: Baseline model metrics
            metric_name: Metric to compare
        
        Returns:
            Dictionary with comparison results
        """
        gnn_value = gnn_metrics.get(metric_name, 0.0)
        baseline_value = baseline_metrics.get(metric_name, 0.0)
        
        improvement = gnn_value - baseline_value
        improvement_pct = (improvement / baseline_value * 100) if baseline_value > 0 else 0.0
        
        return {
            "metric": metric_name,
            "gnn_value": gnn_value,
            "baseline_value": baseline_value,
            "improvement": improvement,
            "improvement_percent": improvement_pct,
            "is_better": gnn_value > baseline_value
        }

