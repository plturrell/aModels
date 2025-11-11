"""Cross-Validation for GNN Models.

This module provides K-fold cross-validation for robust evaluation of GNN models.
"""

import logging
import random
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    from sklearn.model_selection import KFold, StratifiedKFold
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

logger = logging.getLogger(__name__)


class GNNCrossValidator:
    """K-fold cross-validator for GNN models."""
    
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None
    ):
        """Initialize cross-validator.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data before splitting
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def _split_nodes(
        self,
        nodes: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None
    ) -> List[Tuple[List[int], List[int]]]:
        """Split nodes into K folds.
        
        Args:
            nodes: List of nodes
            labels: Optional labels for stratified splitting
        
        Returns:
            List of (train_indices, val_indices) tuples
        """
        n_nodes = len(nodes)
        indices = list(range(n_nodes))
        
        if self.shuffle:
            random.shuffle(indices)
        
        fold_size = n_nodes // self.n_splits
        folds = []
        
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_nodes
            
            val_indices = indices[val_start:val_end]
            train_indices = indices[:val_start] + indices[val_end:]
            
            folds.append((train_indices, val_indices))
        
        return folds
    
    def cross_validate_node_classifier(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Dict[str, str],
        epochs: int = 100,
        lr: float = 0.01,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform K-fold cross-validation for node classifier.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            labels: Node labels
            epochs: Number of epochs per fold
            lr: Learning rate
            device: Device to use
        
        Returns:
            Dictionary with cross-validation results
        """
        from .gnn_node_classifier import GNNNodeClassifier
        from .gnn_evaluation import GNNEvaluator
        
        evaluator = GNNEvaluator()
        fold_results = []
        all_predictions = []
        all_true_labels = []
        
        # Split nodes into folds
        folds = self._split_nodes(nodes, labels)
        
        logger.info(f"Starting {self.n_splits}-fold cross-validation...")
        
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")
            
            # Split data
            train_nodes = [nodes[i] for i in train_indices]
            val_nodes = [nodes[i] for i in val_indices]
            
            # Extract labels
            train_labels = {
                nodes[i].get("id", str(i)): labels.get(nodes[i].get("id", str(i)), "unknown")
                for i in train_indices
            }
            val_labels = {
                nodes[i].get("id", str(i)): labels.get(nodes[i].get("id", str(i)), "unknown")
                for i in val_indices
            }
            
            # Train model
            try:
                classifier = GNNNodeClassifier(device=device)
                train_result = classifier.train(
                    train_nodes,
                    edges,  # Use all edges for training
                    labels=train_labels,
                    epochs=epochs,
                    lr=lr
                )
                
                if "error" in train_result:
                    logger.warning(f"Fold {fold_idx + 1} training failed: {train_result['error']}")
                    continue
                
                # Evaluate on validation set
                val_classifications = classifier.classify_nodes(val_nodes, edges)
                
                if "error" not in val_classifications:
                    # Extract predictions
                    y_true = []
                    y_pred = []
                    
                    for classification in val_classifications.get("classifications", []):
                        node_id = classification["node_id"]
                        if node_id in val_labels:
                            y_true.append(val_labels[node_id])
                            y_pred.append(classification["predicted_class"])
                            all_true_labels.append(val_labels[node_id])
                            all_predictions.append(classification["predicted_class"])
                    
                    if y_true and y_pred:
                        # Evaluate
                        metrics = evaluator.evaluate_classification(y_true, y_pred)
                        
                        fold_results.append({
                            "fold": fold_idx + 1,
                            "train_size": len(train_nodes),
                            "val_size": len(val_nodes),
                            "metrics": metrics,
                            "train_accuracy": train_result.get("accuracy", 0.0)
                        })
                        
                        logger.info(
                            f"Fold {fold_idx + 1}: accuracy={metrics.get('accuracy', 0.0):.4f}, "
                            f"precision={metrics.get('precision', 0.0):.4f}, "
                            f"recall={metrics.get('recall', 0.0):.4f}"
                        )
            
            except Exception as e:
                logger.warning(f"Fold {fold_idx + 1} failed: {e}")
                continue
        
        # Aggregate results
        if fold_results:
            accuracies = [r["metrics"].get("accuracy", 0.0) for r in fold_results]
            precisions = [r["metrics"].get("precision", 0.0) for r in fold_results]
            recalls = [r["metrics"].get("recall", 0.0) for r in fold_results]
            f1_scores = [r["metrics"].get("f1_score", 0.0) for r in fold_results]
            
            # Overall evaluation on all predictions
            overall_metrics = None
            if all_true_labels and all_predictions:
                overall_metrics = evaluator.evaluate_classification(all_true_labels, all_predictions)
            
            return {
                "n_splits": self.n_splits,
                "fold_results": fold_results,
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_precision": float(np.mean(precisions)),
                "std_precision": float(np.std(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "std_recall": float(np.std(recalls)),
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores)),
                "overall_metrics": overall_metrics
            }
        else:
            return {"error": "All folds failed"}
    
    def cross_validate_link_predictor(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        epochs: int = 100,
        lr: float = 0.01,
        device: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform K-fold cross-validation for link predictor.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            epochs: Number of epochs per fold
            lr: Learning rate
            device: Device to use
        
        Returns:
            Dictionary with cross-validation results
        """
        from .gnn_link_predictor import GNNLinkPredictor
        from .gnn_evaluation import GNNEvaluator
        
        evaluator = GNNEvaluator()
        fold_results = []
        
        # Split edges into folds
        n_edges = len(edges)
        edge_indices = list(range(n_edges))
        
        if self.shuffle:
            random.shuffle(edge_indices)
        
        fold_size = n_edges // self.n_splits
        folds = []
        
        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_edges
            
            val_indices = edge_indices[val_start:val_end]
            train_indices = edge_indices[:val_start] + edge_indices[val_end:]
            
            folds.append((train_indices, val_indices))
        
        logger.info(f"Starting {self.n_splits}-fold cross-validation for link prediction...")
        
        for fold_idx, (train_indices, val_indices) in enumerate(folds):
            logger.info(f"Fold {fold_idx + 1}/{self.n_splits}")
            
            # Split edges
            train_edges = [edges[i] for i in train_indices]
            val_edges = [edges[i] for i in val_indices]
            
            # Train model
            try:
                predictor = GNNLinkPredictor(device=device)
                train_result = predictor.train(
                    nodes,
                    train_edges,
                    epochs=epochs,
                    lr=lr
                )
                
                if "error" in train_result:
                    logger.warning(f"Fold {fold_idx + 1} training failed: {train_result['error']}")
                    continue
                
                # Evaluate on validation set
                val_pairs = [
                    (e.get("source_id", ""), e.get("target_id", ""))
                    for e in val_edges
                ]
                
                val_predictions = predictor.predict_links(
                    nodes, train_edges,
                    candidate_pairs=val_pairs,
                    top_k=len(val_pairs)
                )
                
                if "error" not in val_predictions:
                    # Extract predictions
                    val_pairs_set = set(val_pairs)
                    y_true = []
                    y_pred = []
                    y_proba = []
                    
                    for pred in val_predictions.get("predictions", []):
                        pair = (pred["source_id"], pred["target_id"])
                        if pair in val_pairs_set:
                            y_true.append(True)
                            y_pred.append(pred["probability"] >= 0.5)
                            y_proba.append(pred["probability"])
                    
                    # Add false negatives
                    predicted_pairs = {(p["source_id"], p["target_id"]) for p in val_predictions.get("predictions", [])}
                    for pair in val_pairs:
                        if pair not in predicted_pairs:
                            y_true.append(True)
                            y_pred.append(False)
                            y_proba.append(0.0)
                    
                    if y_true:
                        metrics = evaluator.evaluate_link_prediction(y_true, y_pred, y_proba)
                        
                        fold_results.append({
                            "fold": fold_idx + 1,
                            "train_size": len(train_edges),
                            "val_size": len(val_edges),
                            "metrics": metrics,
                            "train_accuracy": train_result.get("accuracy", 0.0)
                        })
                        
                        logger.info(
                            f"Fold {fold_idx + 1}: accuracy={metrics.get('accuracy', 0.0):.4f}, "
                            f"precision={metrics.get('precision', 0.0):.4f}, "
                            f"recall={metrics.get('recall', 0.0):.4f}"
                        )
            
            except Exception as e:
                logger.warning(f"Fold {fold_idx + 1} failed: {e}")
                continue
        
        # Aggregate results
        if fold_results:
            accuracies = [r["metrics"].get("accuracy", 0.0) for r in fold_results]
            precisions = [r["metrics"].get("precision", 0.0) for r in fold_results]
            recalls = [r["metrics"].get("recall", 0.0) for r in fold_results]
            f1_scores = [r["metrics"].get("f1_score", 0.0) for r in fold_results]
            
            return {
                "n_splits": self.n_splits,
                "fold_results": fold_results,
                "mean_accuracy": float(np.mean(accuracies)),
                "std_accuracy": float(np.std(accuracies)),
                "mean_precision": float(np.mean(precisions)),
                "std_precision": float(np.std(precisions)),
                "mean_recall": float(np.mean(recalls)),
                "std_recall": float(np.std(recalls)),
                "mean_f1": float(np.mean(f1_scores)),
                "std_f1": float(np.std(f1_scores))
            }
        else:
            return {"error": "All folds failed"}

