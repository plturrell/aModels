"""GNN Training Integration.

This module provides training capabilities for GNN models integrated with the
training pipeline, including data preparation, training loops, and model persistence.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

from .gnn_node_classifier import GNNNodeClassifier
from .gnn_link_predictor import GNNLinkPredictor
from .gnn_embeddings import GNNEmbedder
from .gnn_anomaly_detector import GNNAnomalyDetector
from .gnn_schema_matcher import GNNSchemaMatcher
from .gnn_evaluation import GNNEvaluator

# AMP imports
try:
    import torch
    from torch.cuda.amp import autocast, GradScaler
    HAS_AMP = True
except ImportError:
    HAS_AMP = False
    autocast = None
    GradScaler = None

logger = logging.getLogger(__name__)


class GNNTrainer:
    """Trainer for GNN models.
    
    Provides training capabilities for all GNN modules with data preparation,
    training loops, evaluation, and model persistence.
    """
    
    def __init__(
        self,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        use_amp: Optional[bool] = None
    ):
        """Initialize GNN trainer.
        
        Args:
            output_dir: Directory for trained models
            device: Device to use ('cuda' or 'cpu')
            use_amp: Whether to use Automatic Mixed Precision (auto-detect if None)
        """
        self.output_dir = output_dir or os.getenv("GNN_MODELS_DIR", "./gnn_models")
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.device = device or os.getenv("GNN_DEVICE", "auto")
        if self.device == "auto":
            self.device = None
        
        # AMP setup
        if use_amp is None:
            # Auto-detect: use AMP if CUDA is available and AMP is supported
            use_amp = HAS_AMP and HAS_PYG and torch.cuda.is_available() if HAS_PYG else False
        self.use_amp = use_amp and HAS_AMP
        
        # Initialize GradScaler for AMP
        self.scaler = None
        if self.use_amp:
            self.scaler = GradScaler()
            logger.info("AMP (Automatic Mixed Precision) enabled for GNN training")
        else:
            logger.info("AMP disabled for GNN training")
        
        self.evaluator = GNNEvaluator()
        
        logger.info(f"Initialized GNN Trainer (output_dir: {self.output_dir}, use_amp: {self.use_amp})")
    
    def prepare_training_data_from_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        label_key: str = "type"
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, str]]]:
        """Prepare training data from knowledge graph.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            label_key: Key to extract labels from nodes
        
        Returns:
            Tuple of (nodes, edges, labels_dict)
        """
        # Extract labels from nodes
        labels = {}
        for node in nodes:
            node_id = node.get("id", node.get("key", {}).get("id", ""))
            label = node.get(label_key, "unknown")
            labels[node_id] = label
        
        return nodes, edges, labels
    
    def train_node_classifier(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        label_key: str = "type",
        epochs: int = 100,
        lr: float = 0.01,
        validation_split: float = 0.2,
        save_model: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train node classifier.
        
        Args:
            nodes: Training nodes
            edges: Training edges
            labels: Optional labels dict (if None, extracts from nodes)
            label_key: Key to extract labels from nodes
            epochs: Number of training epochs
            lr: Learning rate
            validation_split: Fraction of data for validation
            save_model: Whether to save trained model
            model_name: Model name for saving
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training node classifier...")
        
        # Prepare labels if not provided
        if labels is None:
            _, _, labels = self.prepare_training_data_from_graph(nodes, edges, label_key)
        
        # Split data
        num_nodes = len(nodes)
        num_val = int(num_nodes * validation_split)
        val_indices = set(range(num_nodes - num_val, num_nodes))
        
        train_nodes = [n for i, n in enumerate(nodes) if i not in val_indices]
        train_edges = [e for e in edges]  # Use all edges for training
        train_labels = {nodes[i].get("id", str(i)): labels.get(nodes[i].get("id", str(i)), "unknown")
                       for i in range(num_nodes) if i not in val_indices}
        
        val_nodes = [n for i, n in enumerate(nodes) if i in val_indices]
        val_labels = {nodes[i].get("id", str(i)): labels.get(nodes[i].get("id", str(i)), "unknown")
                     for i in val_indices}
        
        # Initialize classifier
        classifier = GNNNodeClassifier(device=self.device)
        
        # Train with AMP support
        train_result = classifier.train(
            train_nodes,
            train_edges,
            labels=train_labels,
            epochs=epochs,
            lr=lr,
            use_amp=self.use_amp,
            scaler=self.scaler
        )
        
        # Evaluate on validation set
        val_classifications = classifier.classify_nodes(val_nodes, train_edges)
        val_metrics = None
        
        if "error" not in val_classifications and val_labels:
            # Extract true and predicted labels
            y_true = []
            y_pred = []
            for classification in val_classifications.get("classifications", []):
                node_id = classification["node_id"]
                if node_id in val_labels:
                    y_true.append(val_labels[node_id])
                    y_pred.append(classification["predicted_class"])
            
            if y_true and y_pred:
                val_metrics = self.evaluator.evaluate_classification(y_true, y_pred)
        
        # Save model
        model_path = None
        if save_model:
            model_name = model_name or f"node_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = os.path.join(self.output_dir, model_name)
            classifier.save_model(model_path)
            logger.info(f"Saved node classifier to {model_path}")
        
        result = {
            "status": "success",
            "training_accuracy": train_result.get("accuracy", 0.0),
            "validation_metrics": val_metrics,
            "num_train_samples": len(train_nodes),
            "num_val_samples": len(val_nodes),
            "num_classes": train_result.get("num_classes", 0),
            "model_path": model_path,
            "class_mapping": train_result.get("class_mapping", {})
        }
        
        logger.info(f"Node classifier training complete: accuracy={train_result.get('accuracy', 0.0):.4f}")
        
        return result
    
    def train_link_predictor(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        epochs: int = 100,
        lr: float = 0.01,
        neg_samples: int = 1,
        validation_split: float = 0.2,
        save_model: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train link predictor.
        
        Args:
            nodes: Training nodes
            edges: Training edges (positive examples)
            epochs: Number of training epochs
            lr: Learning rate
            neg_samples: Number of negative samples per positive edge
            validation_split: Fraction of data for validation
            save_model: Whether to save trained model
            model_name: Model name for saving
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training link predictor...")
        
        # Split edges
        num_edges = len(edges)
        num_val = int(num_edges * validation_split)
        train_edges = edges[:-num_val] if num_val > 0 else edges
        val_edges = edges[-num_val:] if num_val > 0 else []
        
        # Initialize predictor
        predictor = GNNLinkPredictor(device=self.device)
        
        # Train with AMP support
        train_result = predictor.train(
            nodes,
            train_edges,
            epochs=epochs,
            lr=lr,
            neg_samples=neg_samples,
            use_amp=self.use_amp,
            scaler=self.scaler
        )
        
        # Evaluate on validation set
        val_metrics = None
        if val_edges:
            # Create validation pairs
            val_pairs = [(e.get("source_id", ""), e.get("target_id", "")) for e in val_edges]
            val_predictions = predictor.predict_links(nodes, train_edges, candidate_pairs=val_pairs, top_k=len(val_pairs))
            
            if "error" not in val_predictions:
                # Extract true and predicted labels
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
                
                # Add false negatives (edges in val_edges but not in predictions)
                predicted_pairs = {(p["source_id"], p["target_id"]) for p in val_predictions.get("predictions", [])}
                for pair in val_pairs:
                    if pair not in predicted_pairs:
                        y_true.append(True)
                        y_pred.append(False)
                        y_proba.append(0.0)
                
                if y_true:
                    val_metrics = self.evaluator.evaluate_link_prediction(y_true, y_pred, y_proba)
        
        # Save model
        model_path = None
        if save_model:
            model_name = model_name or f"link_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = os.path.join(self.output_dir, model_name)
            predictor.save_model(model_path)
            logger.info(f"Saved link predictor to {model_path}")
        
        result = {
            "status": "success",
            "training_accuracy": train_result.get("accuracy", 0.0),
            "validation_metrics": val_metrics,
            "num_train_edges": len(train_edges),
            "num_val_edges": len(val_edges),
            "model_path": model_path
        }
        
        logger.info(f"Link predictor training complete: accuracy={train_result.get('accuracy', 0.0):.4f}")
        
        return result
    
    def train_anomaly_detector(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        epochs: int = 100,
        lr: float = 0.01,
        save_model: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train anomaly detector.
        
        Args:
            nodes: Training nodes (normal patterns)
            edges: Training edges (normal patterns)
            epochs: Number of training epochs
            lr: Learning rate
            save_model: Whether to save trained model
            model_name: Model name for saving
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training anomaly detector...")
        
        # Initialize detector
        detector = GNNAnomalyDetector(device=self.device)
        
        # Train
        train_result = detector.train(
            nodes,
            edges,
            epochs=epochs,
            lr=lr
        )
        
        # Save model
        model_path = None
        if save_model:
            model_name = model_name or f"anomaly_detector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = os.path.join(self.output_dir, model_name)
            detector.save_model(model_path)
            logger.info(f"Saved anomaly detector to {model_path}")
        
        result = {
            "status": "success",
            "final_loss": train_result.get("final_loss", 0.0),
            "num_nodes": train_result.get("num_nodes", 0),
            "num_edges": train_result.get("num_edges", 0),
            "model_path": model_path
        }
        
        logger.info(f"Anomaly detector training complete: loss={train_result.get('final_loss', 0.0):.4f}")
        
        return result
    
    def train_schema_matcher(
        self,
        schema_pairs: List[Dict[str, Any]],
        labels: List[float],
        epochs: int = 100,
        lr: float = 0.01,
        validation_split: float = 0.2,
        save_model: bool = True,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train schema matcher.
        
        Args:
            schema_pairs: List of schema pairs, each with 'schema1' and 'schema2'
            labels: Similarity labels (0.0 to 1.0)
            epochs: Number of training epochs
            lr: Learning rate
            validation_split: Fraction of data for validation
            save_model: Whether to save trained model
            model_name: Model name for saving
        
        Returns:
            Dictionary with training results
        """
        logger.info("Training schema matcher...")
        
        # Split data
        num_pairs = len(schema_pairs)
        num_val = int(num_pairs * validation_split)
        train_pairs = schema_pairs[:-num_val] if num_val > 0 else schema_pairs
        train_labels = labels[:-num_val] if num_val > 0 else labels
        val_pairs = schema_pairs[-num_val:] if num_val > 0 else []
        val_labels = labels[-num_val:] if num_val > 0 else []
        
        # Initialize matcher
        matcher = GNNSchemaMatcher(device=self.device)
        
        # Train
        train_result = matcher.train(
            train_pairs,
            train_labels,
            epochs=epochs,
            lr=lr
        )
        
        # Evaluate on validation set
        val_metrics = None
        if val_pairs and val_labels:
            similarities = []
            for pair in val_pairs:
                result = matcher.match_schemas(
                    pair["schema1"].get("nodes", []),
                    pair["schema1"].get("edges", []),
                    pair["schema2"].get("nodes", []),
                    pair["schema2"].get("edges", [])
                )
                if "error" not in result:
                    similarities.append(result["similarity"])
            
            if similarities:
                threshold = 0.5
                y_pred = [s >= threshold for s in similarities]
                y_proba = similarities
                val_metrics = self.evaluator.evaluate_schema_matching(similarities, val_labels, threshold)
        
        # Save model
        model_path = None
        if save_model:
            model_name = model_name or f"schema_matcher_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = os.path.join(self.output_dir, model_name)
            matcher.save_model(model_path)
            logger.info(f"Saved schema matcher to {model_path}")
        
        result = {
            "status": "success",
            "final_loss": train_result.get("final_loss", 0.0),
            "validation_metrics": val_metrics,
            "num_train_pairs": len(train_pairs),
            "num_val_pairs": len(val_pairs),
            "model_path": model_path
        }
        
        logger.info(f"Schema matcher training complete: loss={train_result.get('final_loss', 0.0):.4f}")
        
        return result
    
    def load_trained_models(
        self,
        model_paths: Dict[str, str]
    ) -> Dict[str, Any]:
        """Load trained models.
        
        Args:
            model_paths: Dictionary mapping model type to path:
                {
                    "classifier": "path/to/classifier.pt",
                    "link_predictor": "path/to/predictor.pt",
                    "anomaly_detector": "path/to/detector.pt",
                    "schema_matcher": "path/to/matcher.pt"
                }
        
        Returns:
            Dictionary with loaded models
        """
        models = {}
        
        if "classifier" in model_paths:
            try:
                classifier = GNNNodeClassifier(device=self.device)
                classifier.load_model(model_paths["classifier"])
                models["classifier"] = classifier
                logger.info(f"Loaded node classifier from {model_paths['classifier']}")
            except Exception as e:
                logger.error(f"Failed to load classifier: {e}")
        
        if "link_predictor" in model_paths:
            try:
                predictor = GNNLinkPredictor(device=self.device)
                predictor.load_model(model_paths["link_predictor"])
                models["link_predictor"] = predictor
                logger.info(f"Loaded link predictor from {model_paths['link_predictor']}")
            except Exception as e:
                logger.error(f"Failed to load link predictor: {e}")
        
        if "anomaly_detector" in model_paths:
            try:
                detector = GNNAnomalyDetector(device=self.device)
                detector.load_model(model_paths["anomaly_detector"])
                models["anomaly_detector"] = detector
                logger.info(f"Loaded anomaly detector from {model_paths['anomaly_detector']}")
            except Exception as e:
                logger.error(f"Failed to load anomaly detector: {e}")
        
        if "schema_matcher" in model_paths:
            try:
                matcher = GNNSchemaMatcher(device=self.device)
                matcher.load_model(model_paths["schema_matcher"])
                models["schema_matcher"] = matcher
                logger.info(f"Loaded schema matcher from {model_paths['schema_matcher']}")
            except Exception as e:
                logger.error(f"Failed to load schema matcher: {e}")
        
        return models

