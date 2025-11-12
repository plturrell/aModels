"""Model Ensembling for GNN.

This module provides ensemble methods to combine multiple GNN models for improved performance.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class GNNEnsemble:
    """Ensemble of GNN models."""
    
    def __init__(
        self,
        models: List[Any],
        ensemble_method: str = "voting",  # "voting", "weighted", "stacking"
        weights: Optional[List[float]] = None
    ):
        """Initialize ensemble.
        
        Args:
            models: List of trained GNN models
            ensemble_method: Ensemble method ("voting", "weighted", "stacking")
            weights: Optional weights for weighted voting
        """
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = weights
        
        if weights is None and ensemble_method == "weighted":
            # Equal weights
            self.weights = [1.0 / len(models)] * len(models)
        elif weights is not None:
            # Normalize weights
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def predict_classification(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Ensemble prediction for node classification.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
        
        Returns:
            Dictionary with ensemble predictions
        """
        if not self.models:
            return {"error": "No models in ensemble"}
        
        all_predictions = []
        all_probabilities = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                result = model.classify_nodes(nodes, edges)
                if "error" not in result:
                    classifications = result.get("classifications", [])
                    all_predictions.append(classifications)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not all_predictions:
            return {"error": "All model predictions failed"}
        
        # Ensemble predictions
        if self.ensemble_method == "voting":
            return self._voting_ensemble(all_predictions)
        elif self.ensemble_method == "weighted":
            return self._weighted_ensemble(all_predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking_ensemble(all_predictions, nodes, edges)
        else:
            return {"error": f"Unknown ensemble method: {self.ensemble_method}"}
    
    def _voting_ensemble(
        self,
        all_predictions: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Majority voting ensemble.
        
        Args:
            all_predictions: List of prediction lists from each model
        
        Returns:
            Ensemble predictions
        """
        # Group predictions by node_id
        node_predictions = {}
        
        for predictions in all_predictions:
            for pred in predictions:
                node_id = pred["node_id"]
                if node_id not in node_predictions:
                    node_predictions[node_id] = []
                node_predictions[node_id].append(pred["predicted_class"])
        
        # Majority vote
        ensemble_predictions = []
        for node_id, classes in node_predictions.items():
            # Count votes
            from collections import Counter
            vote_counts = Counter(classes)
            predicted_class = vote_counts.most_common(1)[0][0]
            confidence = vote_counts[predicted_class] / len(classes)
            
            ensemble_predictions.append({
                "node_id": node_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "votes": dict(vote_counts)
            })
        
        return {
            "classifications": ensemble_predictions,
            "num_models": len(all_predictions),
            "ensemble_method": "voting"
        }
    
    def _weighted_ensemble(
        self,
        all_predictions: List[List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Weighted voting ensemble.
        
        Args:
            all_predictions: List of prediction lists from each model
        
        Returns:
            Ensemble predictions
        """
        # Group predictions by node_id
        node_predictions = {}
        
        for model_idx, predictions in enumerate(all_predictions):
            weight = self.weights[model_idx] if self.weights else 1.0 / len(all_predictions)
            
            for pred in predictions:
                node_id = pred["node_id"]
                if node_id not in node_predictions:
                    node_predictions[node_id] = {}
                
                predicted_class = pred["predicted_class"]
                if predicted_class not in node_predictions[node_id]:
                    node_predictions[node_id][predicted_class] = 0.0
                
                node_predictions[node_id][predicted_class] += weight
        
        # Weighted vote
        ensemble_predictions = []
        for node_id, class_weights in node_predictions.items():
            predicted_class = max(class_weights.items(), key=lambda x: x[1])[0]
            confidence = class_weights[predicted_class]
            
            ensemble_predictions.append({
                "node_id": node_id,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "class_weights": class_weights
            })
        
        return {
            "classifications": ensemble_predictions,
            "num_models": len(all_predictions),
            "ensemble_method": "weighted",
            "weights": self.weights
        }
    
    def _stacking_ensemble(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Stacking ensemble (requires meta-learner).
        
        Args:
            all_predictions: List of prediction lists from each model
            nodes: Graph nodes
            edges: Graph edges
        
        Returns:
            Ensemble predictions
        """
        # For now, use simple voting (stacking requires training meta-learner)
        logger.warning("Stacking ensemble not fully implemented, using voting")
        return self._voting_ensemble(all_predictions)
    
    def predict_links(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        candidate_pairs: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Ensemble prediction for link prediction.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            candidate_pairs: Candidate pairs to evaluate
            top_k: Number of top predictions
        
        Returns:
            Dictionary with ensemble link predictions
        """
        if not self.models:
            return {"error": "No models in ensemble"}
        
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            try:
                result = model.predict_links(
                    nodes, edges,
                    candidate_pairs=candidate_pairs,
                    top_k=top_k * 2  # Get more candidates for ensemble
                )
                if "error" not in result:
                    predictions = result.get("predictions", [])
                    all_predictions.append(predictions)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if not all_predictions:
            return {"error": "All model predictions failed"}
        
        # Ensemble predictions
        if self.ensemble_method == "voting":
            return self._voting_ensemble_links(all_predictions, top_k)
        elif self.ensemble_method == "weighted":
            return self._weighted_ensemble_links(all_predictions, top_k)
        else:
            return {"error": f"Unknown ensemble method: {self.ensemble_method}"}
    
    def _voting_ensemble_links(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        top_k: int
    ) -> Dict[str, Any]:
        """Majority voting for links.
        
        Args:
            all_predictions: List of prediction lists from each model
            top_k: Number of top predictions
        
        Returns:
            Ensemble link predictions
        """
        # Aggregate probabilities by pair
        pair_probs = {}
        
        for predictions in all_predictions:
            for pred in predictions:
                pair = (pred["source_id"], pred["target_id"])
                if pair not in pair_probs:
                    pair_probs[pair] = []
                pair_probs[pair].append(pred["probability"])
        
        # Average probabilities
        ensemble_predictions = []
        for pair, probs in pair_probs.items():
            avg_prob = sum(probs) / len(probs)
            ensemble_predictions.append({
                "source_id": pair[0],
                "target_id": pair[1],
                "probability": avg_prob,
                "num_models": len(probs)
            })
        
        # Sort by probability and take top_k
        ensemble_predictions.sort(key=lambda x: x["probability"], reverse=True)
        ensemble_predictions = ensemble_predictions[:top_k]
        
        return {
            "predictions": ensemble_predictions,
            "num_models": len(all_predictions),
            "ensemble_method": "voting"
        }
    
    def _weighted_ensemble_links(
        self,
        all_predictions: List[List[Dict[str, Any]]],
        top_k: int
    ) -> Dict[str, Any]:
        """Weighted ensemble for links.
        
        Args:
            all_predictions: List of prediction lists from each model
            top_k: Number of top predictions
        
        Returns:
            Ensemble link predictions
        """
        # Aggregate weighted probabilities by pair
        pair_probs = {}
        
        for model_idx, predictions in enumerate(all_predictions):
            weight = self.weights[model_idx] if self.weights else 1.0 / len(all_predictions)
            
            for pred in predictions:
                pair = (pred["source_id"], pred["target_id"])
                if pair not in pair_probs:
                    pair_probs[pair] = 0.0
                pair_probs[pair] += pred["probability"] * weight
        
        # Create predictions
        ensemble_predictions = []
        for pair, prob in pair_probs.items():
            ensemble_predictions.append({
                "source_id": pair[0],
                "target_id": pair[1],
                "probability": prob
            })
        
        # Sort by probability and take top_k
        ensemble_predictions.sort(key=lambda x: x["probability"], reverse=True)
        ensemble_predictions = ensemble_predictions[:top_k]
        
        return {
            "predictions": ensemble_predictions,
            "num_models": len(all_predictions),
            "ensemble_method": "weighted",
            "weights": self.weights
        }


class GNNEnsembleBuilder:
    """Builder for creating GNN ensembles."""
    
    def __init__(
        self,
        device: Optional[str] = None
    ):
        """Initialize ensemble builder.
        
        Args:
            device: Device to use
        """
        self.device = device
    
    def build_diverse_ensemble(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        num_models: int = 3
    ) -> GNNEnsemble:
        """Build diverse ensemble with different architectures.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            labels: Node labels (for classification)
            num_models: Number of models in ensemble
        
        Returns:
            GNN ensemble
        """
        from .gnn_node_classifier import GNNNodeClassifier
        
        models = []
        
        # Model 1: GraphSAGE
        try:
            model1 = GNNNodeClassifier(
                device=self.device,
                hidden_dim=64,
                num_layers=3,
                dropout=0.1,
                use_sage=True
            )
            if labels:
                model1.train(nodes, edges, labels=labels, epochs=50, lr=0.01)
            models.append(model1)
        except Exception as e:
            logger.warning(f"Failed to create GraphSAGE model: {e}")
        
        # Model 2: GCN
        try:
            model2 = GNNNodeClassifier(
                device=self.device,
                hidden_dim=128,
                num_layers=2,
                dropout=0.2,
                use_sage=False
            )
            if labels:
                model2.train(nodes, edges, labels=labels, epochs=50, lr=0.01)
            models.append(model2)
        except Exception as e:
            logger.warning(f"Failed to create GCN model: {e}")
        
        # Model 3: Deep GraphSAGE
        if num_models >= 3:
            try:
                model3 = GNNNodeClassifier(
                    device=self.device,
                    hidden_dim=96,
                    num_layers=4,
                    dropout=0.15,
                    use_sage=True
                )
                if labels:
                    model3.train(nodes, edges, labels=labels, epochs=50, lr=0.005)
                models.append(model3)
            except Exception as e:
                logger.warning(f"Failed to create deep GraphSAGE model: {e}")
        
        if not models:
            raise ValueError("Failed to create any models for ensemble")
        
        return GNNEnsemble(models, ensemble_method="voting")
    
    def build_from_pretrained(
        self,
        model_paths: List[str],
        ensemble_method: str = "voting"
    ) -> GNNEnsemble:
        """Build ensemble from pre-trained models.
        
        Args:
            model_paths: List of paths to pre-trained models
            ensemble_method: Ensemble method
        
        Returns:
            GNN ensemble
        """
        from .gnn_node_classifier import GNNNodeClassifier
        
        models = []
        
        for model_path in model_paths:
            try:
                model = GNNNodeClassifier(device=self.device)
                model.load_model(model_path)
                models.append(model)
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
        
        if not models:
            raise ValueError("Failed to load any models for ensemble")
        
        return GNNEnsemble(models, ensemble_method=ensemble_method)

