"""Transfer Learning for GNN Models.

This module provides transfer learning capabilities: pre-training on large graphs,
fine-tuning for specific domains, and model sharing across projects.
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class GNNTransferLearner:
    """Transfer learning for GNN models."""
    
    def __init__(
        self,
        model_registry_dir: Optional[str] = None,
        device: Optional[str] = None
    ):
        """Initialize transfer learner.
        
        Args:
            model_registry_dir: Directory for model registry
            device: Device to use
        """
        self.model_registry_dir = model_registry_dir or os.getenv(
            "GNN_MODEL_REGISTRY",
            "./gnn_model_registry"
        )
        os.makedirs(self.model_registry_dir, exist_ok=True)
        
        self.device = device or os.getenv("GNN_DEVICE", "auto")
        if self.device == "auto":
            self.device = None
    
    def pretrain_on_large_graph(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        model_type: str = "classifier",  # "classifier", "link_predictor", "embedder"
        model_name: Optional[str] = None,
        epochs: int = 200,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Pre-train model on large graph.
        
        Args:
            nodes: Large graph nodes
            edges: Large graph edges
            model_type: Type of model to pre-train
            model_name: Name for saved model
            epochs: Number of pre-training epochs
            lr: Learning rate
        
        Returns:
            Dictionary with pre-training results
        """
        logger.info(f"Pre-training {model_type} on large graph ({len(nodes)} nodes, {len(edges)} edges)...")
        
        if model_type == "classifier":
            from .gnn_node_classifier import GNNNodeClassifier
            
            # Extract labels from nodes (use type as label)
            labels = {
                node.get("id", ""): node.get("type", "unknown")
                for node in nodes
            }
            
            classifier = GNNNodeClassifier(device=self.device)
            result = classifier.train(
                nodes, edges,
                labels=labels,
                epochs=epochs,
                lr=lr
            )
            
            if "error" not in result:
                # Save pre-trained model
                model_name = model_name or f"pretrained_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                model_path = os.path.join(self.model_registry_dir, model_name)
                classifier.save_model(model_path)
                
                # Save metadata
                metadata = {
                    "model_type": "classifier",
                    "pretrained_on": {
                        "num_nodes": len(nodes),
                        "num_edges": len(edges),
                        "num_classes": result.get("num_classes", 0)
                    },
                    "training_accuracy": result.get("accuracy", 0.0),
                    "created_at": datetime.now().isoformat()
                }
                metadata_path = model_path.replace(".pt", "_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                return {
                    "status": "success",
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                    "training_result": result
                }
            else:
                return result
        
        elif model_type == "link_predictor":
            from .gnn_link_predictor import GNNLinkPredictor
            
            predictor = GNNLinkPredictor(device=self.device)
            result = predictor.train(
                nodes, edges,
                epochs=epochs,
                lr=lr
            )
            
            if "error" not in result:
                model_name = model_name or f"pretrained_link_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                model_path = os.path.join(self.model_registry_dir, model_name)
                predictor.save_model(model_path)
                
                metadata = {
                    "model_type": "link_predictor",
                    "pretrained_on": {
                        "num_nodes": len(nodes),
                        "num_edges": len(edges)
                    },
                    "training_accuracy": result.get("accuracy", 0.0),
                    "created_at": datetime.now().isoformat()
                }
                metadata_path = model_path.replace(".pt", "_metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                
                return {
                    "status": "success",
                    "model_path": model_path,
                    "metadata_path": metadata_path,
                    "training_result": result
                }
            else:
                return result
        
        elif model_type == "embedder":
            from .gnn_embeddings import GNNEmbedder
            
            # For embedder, we just save the model architecture
            embedder = GNNEmbedder(device=self.device)
            
            model_name = model_name or f"pretrained_embedder_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
            model_path = os.path.join(self.model_registry_dir, model_name)
            embedder.save_model(model_path)
            
            metadata = {
                "model_type": "embedder",
                "pretrained_on": {
                    "num_nodes": len(nodes),
                    "num_edges": len(edges)
                },
                "created_at": datetime.now().isoformat()
            }
            metadata_path = model_path.replace(".pt", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return {
                "status": "success",
                "model_path": model_path,
                "metadata_path": metadata_path
            }
        
        else:
            return {"error": f"Unknown model type: {model_type}"}
    
    def fine_tune_for_domain(
        self,
        pretrained_model_path: str,
        domain_nodes: List[Dict[str, Any]],
        domain_edges: List[Dict[str, Any]],
        domain_labels: Optional[Dict[str, str]] = None,
        fine_tune_epochs: int = 50,
        lr: float = 0.001,  # Lower learning rate for fine-tuning
        freeze_backbone: bool = False
    ) -> Dict[str, Any]:
        """Fine-tune pre-trained model for specific domain.
        
        Args:
            pretrained_model_path: Path to pre-trained model
            domain_nodes: Domain-specific nodes
            domain_edges: Domain-specific edges
            domain_labels: Domain-specific labels
            fine_tune_epochs: Number of fine-tuning epochs
            lr: Learning rate for fine-tuning
            freeze_backbone: Whether to freeze backbone layers
        
        Returns:
            Dictionary with fine-tuning results
        """
        logger.info(f"Fine-tuning model from {pretrained_model_path} for domain...")
        
        # Load pre-trained model
        from .gnn_node_classifier import GNNNodeClassifier
        
        model = GNNNodeClassifier(device=self.device)
        model.load_model(pretrained_model_path)
        
        # Freeze backbone if requested
        if freeze_backbone and HAS_TORCH and model.model is not None:
            # Freeze all layers except classifier head
            for param in model.model.parameters():
                param.requires_grad = False
            # Unfreeze classifier head
            if hasattr(model.model, "classifier"):
                for param in model.model.classifier.parameters():
                    param.requires_grad = True
        
        # Fine-tune
        if domain_labels:
            result = model.train(
                domain_nodes,
                domain_edges,
                labels=domain_labels,
                epochs=fine_tune_epochs,
                lr=lr
            )
        else:
            # Extract labels from nodes
            labels = {
                node.get("id", ""): node.get("type", "unknown")
                for node in domain_nodes
            }
            result = model.train(
                domain_nodes,
                domain_edges,
                labels=labels,
                epochs=fine_tune_epochs,
                lr=lr
            )
        
        if "error" not in result:
            # Save fine-tuned model
            model_name = f"finetuned_{os.path.basename(pretrained_model_path)}"
            model_path = os.path.join(self.model_registry_dir, model_name)
            model.save_model(model_path)
            
            return {
                "status": "success",
                "model_path": model_path,
                "fine_tuning_result": result,
                "pretrained_model": pretrained_model_path
            }
        else:
            return result
    
    def list_available_models(
        self,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available pre-trained models.
        
        Args:
            model_type: Filter by model type (optional)
        
        Returns:
            List of available models with metadata
        """
        models = []
        
        for file in os.listdir(self.model_registry_dir):
            if file.endswith("_metadata.json"):
                try:
                    metadata_path = os.path.join(self.model_registry_dir, file)
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    
                    if model_type is None or metadata.get("model_type") == model_type:
                        model_path = metadata_path.replace("_metadata.json", ".pt")
                        if os.path.exists(model_path):
                            models.append({
                                "model_path": model_path,
                                "metadata": metadata
                            })
                except Exception as e:
                    logger.warning(f"Failed to load metadata from {file}: {e}")
        
        return models
    
    def share_model(
        self,
        model_path: str,
        target_dir: str,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Share/copy model to another location.
        
        Args:
            model_path: Path to model file
            target_dir: Target directory
            include_metadata: Whether to include metadata
        
        Returns:
            Dictionary with sharing results
        """
        import shutil
        
        os.makedirs(target_dir, exist_ok=True)
        
        try:
            # Copy model
            target_path = os.path.join(target_dir, os.path.basename(model_path))
            shutil.copy2(model_path, target_path)
            
            result = {"status": "success", "target_path": target_path}
            
            # Copy metadata if available
            if include_metadata:
                metadata_path = model_path.replace(".pt", "_metadata.json")
                if os.path.exists(metadata_path):
                    target_metadata = os.path.join(target_dir, os.path.basename(metadata_path))
                    shutil.copy2(metadata_path, target_metadata)
                    result["metadata_path"] = target_metadata
            
            return result
        except Exception as e:
            return {"error": str(e)}

