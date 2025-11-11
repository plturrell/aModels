"""Active Learning for GNN Models.

This module provides active learning capabilities: uncertainty-based sampling,
user feedback integration, and continuous learning.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class GNNActiveLearner:
    """Active learning for GNN models."""
    
    def __init__(
        self,
        model=None,
        sampling_strategy: str = "uncertainty",  # "uncertainty", "diversity", "hybrid"
        device: Optional[str] = None
    ):
        """Initialize active learner.
        
        Args:
            model: GNN model to use
            sampling_strategy: Sampling strategy
            device: Device to use
        """
        self.model = model
        self.sampling_strategy = sampling_strategy
        self.device = device
        
        # Track labeled and unlabeled data
        self.labeled_nodes = set()
        self.unlabeled_nodes = set()
        self.user_feedback = {}  # node_id -> label
    
    def compute_uncertainty(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute prediction uncertainty for nodes.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
        
        Returns:
            Dictionary mapping node_id to uncertainty score
        """
        if self.model is None:
            return {"error": "Model not available"}
        
        try:
            # Get predictions with probabilities
            result = self.model.classify_nodes(nodes, edges)
            
            if "error" in result:
                return result
            
            uncertainties = {}
            for classification in result.get("classifications", []):
                node_id = classification["node_id"]
                probabilities = classification.get("probabilities", [])
                
                if probabilities:
                    # Entropy-based uncertainty
                    probs = np.array(probabilities)
                    probs = probs[probs > 0]  # Remove zeros
                    entropy = -np.sum(probs * np.log(probs))
                    uncertainties[node_id] = float(entropy)
                else:
                    # If no probabilities, use confidence
                    confidence = classification.get("confidence", 0.5)
                    uncertainties[node_id] = 1.0 - confidence
            
            return uncertainties
        except Exception as e:
            logger.error(f"Uncertainty computation failed: {e}")
            return {"error": str(e)}
    
    def select_samples_for_labeling(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_samples: int = 10,
        exclude_labeled: bool = True
    ) -> List[Dict[str, Any]]:
        """Select samples for labeling based on active learning strategy.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            num_samples: Number of samples to select
            exclude_labeled: Whether to exclude already labeled nodes
        
        Returns:
            List of selected nodes with uncertainty scores
        """
        # Filter unlabeled nodes
        candidate_nodes = nodes
        if exclude_labeled:
            candidate_nodes = [
                n for n in nodes
                if n.get("id", "") not in self.labeled_nodes
            ]
        
        if not candidate_nodes:
            return []
        
        if self.sampling_strategy == "uncertainty":
            return self._uncertainty_sampling(candidate_nodes, edges, num_samples)
        elif self.sampling_strategy == "diversity":
            return self._diversity_sampling(candidate_nodes, edges, num_samples)
        elif self.sampling_strategy == "hybrid":
            return self._hybrid_sampling(candidate_nodes, edges, num_samples)
        else:
            return candidate_nodes[:num_samples]
    
    def _uncertainty_sampling(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Select samples with highest uncertainty.
        
        Args:
            nodes: Candidate nodes
            edges: Graph edges
            num_samples: Number of samples
        
        Returns:
            Selected nodes
        """
        uncertainties = self.compute_uncertainty(nodes, edges)
        
        if "error" in uncertainties:
            return nodes[:num_samples]
        
        # Sort by uncertainty (highest first)
        node_uncertainties = [
            (n, uncertainties.get(n.get("id", ""), 0.0))
            for n in nodes
        ]
        node_uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        selected = [
            {
                **node,
                "uncertainty": uncertainty,
                "selection_reason": "high_uncertainty"
            }
            for node, uncertainty in node_uncertainties[:num_samples]
        ]
        
        return selected
    
    def _diversity_sampling(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Select diverse samples (farthest from labeled set).
        
        Args:
            nodes: Candidate nodes
            edges: Graph edges
            num_samples: Number of samples
        
        Returns:
            Selected nodes
        """
        # For now, use random sampling (diversity requires embeddings)
        # TODO: Implement proper diversity sampling using embeddings
        import random
        selected = random.sample(nodes, min(num_samples, len(nodes)))
        
        return [
            {
                **node,
                "selection_reason": "diversity"
            }
            for node in selected
        ]
    
    def _hybrid_sampling(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_samples: int
    ) -> List[Dict[str, Any]]:
        """Hybrid sampling: combine uncertainty and diversity.
        
        Args:
            nodes: Candidate nodes
            edges: Graph edges
            num_samples: Number of samples
        
        Returns:
            Selected nodes
        """
        # Get uncertainty scores
        uncertainties = self.compute_uncertainty(nodes, edges)
        
        if "error" in uncertainties:
            return nodes[:num_samples]
        
        # Combine uncertainty with diversity (simple: weight uncertainty 70%, diversity 30%)
        node_scores = []
        for node in nodes:
            node_id = node.get("id", "")
            uncertainty = uncertainties.get(node_id, 0.0)
            # Diversity score (placeholder: distance from labeled set)
            diversity = 1.0  # TODO: Compute actual diversity
            score = 0.7 * uncertainty + 0.3 * diversity
            node_scores.append((node, score))
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = [
            {
                **node,
                "uncertainty": uncertainties.get(node.get("id", ""), 0.0),
                "selection_reason": "hybrid"
            }
            for node, _ in node_scores[:num_samples]
        ]
        
        return selected
    
    def add_user_feedback(
        self,
        node_id: str,
        label: str,
        confidence: float = 1.0
    ):
        """Add user feedback for a node.
        
        Args:
            node_id: Node ID
            label: User-provided label
            confidence: User confidence (0.0 to 1.0)
        """
        self.user_feedback[node_id] = {
            "label": label,
            "confidence": confidence
        }
        self.labeled_nodes.add(node_id)
        if node_id in self.unlabeled_nodes:
            self.unlabeled_nodes.remove(node_id)
    
    def update_model_with_feedback(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        epochs: int = 10,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Update model with user feedback.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Training result
        """
        if self.model is None:
            return {"error": "Model not available"}
        
        # Extract labels from feedback
        labels = {
            node_id: feedback["label"]
            for node_id, feedback in self.user_feedback.items()
        }
        
        if not labels:
            return {"error": "No user feedback available"}
        
        # Filter nodes to only those with labels
        labeled_node_ids = set(labels.keys())
        training_nodes = [
            n for n in nodes
            if n.get("id", "") in labeled_node_ids
        ]
        
        if not training_nodes:
            return {"error": "No labeled nodes found"}
        
        # Update model
        try:
            result = self.model.train(
                training_nodes,
                edges,
                labels=labels,
                epochs=epochs,
                lr=lr
            )
            
            return result
        except Exception as e:
            return {"error": str(e)}
    
    def continuous_learning_loop(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        num_iterations: int = 5,
        samples_per_iteration: int = 10,
        epochs_per_update: int = 10
    ) -> Dict[str, Any]:
        """Continuous learning loop: select samples, get feedback, update model.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            num_iterations: Number of active learning iterations
            samples_per_iteration: Samples to label per iteration
            epochs_per_update: Training epochs per update
        
        Returns:
            Learning history
        """
        history = []
        
        # Initialize unlabeled set
        self.unlabeled_nodes = {n.get("id", "") for n in nodes}
        self.unlabeled_nodes -= self.labeled_nodes
        
        for iteration in range(num_iterations):
            logger.info(f"Active learning iteration {iteration + 1}/{num_iterations}")
            
            # Select samples
            selected = self.select_samples_for_labeling(
                nodes, edges,
                num_samples=samples_per_iteration,
                exclude_labeled=True
            )
            
            if not selected:
                logger.info("No more samples to label")
                break
            
            # Store selected samples (in real scenario, user would label these)
            history.append({
                "iteration": iteration + 1,
                "selected_samples": [
                    {
                        "node_id": s.get("id", ""),
                        "uncertainty": s.get("uncertainty", 0.0),
                        "selection_reason": s.get("selection_reason", "unknown")
                    }
                    for s in selected
                ],
                "num_labeled": len(self.labeled_nodes),
                "num_unlabeled": len(self.unlabeled_nodes)
            })
            
            # In real scenario, user would provide labels here
            # For now, simulate by using ground truth from nodes
            for sample in selected:
                node_id = sample.get("id", "")
                # Find corresponding node
                for node in nodes:
                    if node.get("id", "") == node_id:
                        label = node.get("type", "unknown")
                        self.add_user_feedback(node_id, label)
                        break
            
            # Update model
            update_result = self.update_model_with_feedback(
                nodes, edges,
                epochs=epochs_per_update
            )
            
            if "error" not in update_result:
                history[-1]["update_result"] = {
                    "accuracy": update_result.get("accuracy", 0.0),
                    "status": "success"
                }
            else:
                history[-1]["update_result"] = {
                    "error": update_result.get("error"),
                    "status": "failed"
                }
        
        return {
            "history": history,
            "final_num_labeled": len(self.labeled_nodes),
            "final_num_unlabeled": len(self.unlabeled_nodes)
        }

