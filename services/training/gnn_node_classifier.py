"""GNN-based Node Classification.

This module implements node classification using Graph Neural Networks to classify
nodes (tables, columns) by type, domain, and quality.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from .gnn_features import extract_node_features as _shared_extract_node_features

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    # Provide a minimal nn.Module stub so class definitions don't fail at import time
    class _NNModuleStub:
        pass
    class _NNStub:
        Module = _NNModuleStub
    nn = _NNStub()
    Data = Any
    Batch = Any

logger = logging.getLogger(__name__)


class NodeClassifier(nn.Module):
    """GNN-based node classifier.
    
    Uses GraphSAGE or GCN to classify nodes by type, domain, and quality.
    """
    
    def __init__(
        self,
        num_node_features: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False,
        use_sage: bool = True
    ):
        """Initialize node classifier.
        
        Args:
            num_node_features: Number of input node features
            num_classes: Number of classification classes
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network
            use_sage: Whether to use GraphSAGE (default, more flexible)
        """
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_gat = use_gat
        self.use_sage = use_sage
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = num_node_features
            else:
                in_dim = hidden_dim
            
            out_dim = hidden_dim if i < num_layers - 1 else hidden_dim
            
            if use_sage:
                self.convs.append(GraphSAGE(in_dim, out_dim, num_layers=1))
            elif use_gat:
                self.convs.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout))
            else:
                self.convs.append(GCNConv(in_dim, out_dim))
        
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector for graph-level pooling (optional)
        
        Returns:
            Log probabilities for each class [num_nodes, num_classes]
        """
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Classification
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


class GNNNodeClassifier:
    """GNN-based node classifier for knowledge graphs.
    
    Classifies nodes (tables, columns) by:
    - Type (table, column, view, etc.)
    - Domain (finance, risk, regulatory, etc.)
    - Quality (high, medium, low)
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False,
        use_sage: bool = True,
        device: Optional[str] = None
    ):
        """Initialize GNN node classifier.
        
        Args:
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network
            use_sage: Whether to use GraphSAGE (default)
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN node classification")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        self.use_sage = use_sage
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.num_node_features = None
        self.num_classes = None
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        logger.info(f"Initialized GNN Node Classifier (device: {self.device})")
    
    def _extract_node_features(
        self,
        node: Dict[str, Any],
        props: Dict[str, Any]
    ) -> List[float]:
        return _shared_extract_node_features(node, props)
    
    def convert_graph_to_pyg_data(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Optional[Data]:
        """Convert knowledge graph to PyG Data format.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            PyG Data object or None if conversion fails
        """
        if not HAS_PYG:
            return None
        
        try:
            # Create node ID mapping
            node_id_to_idx = {}
            node_features = []
            
            for idx, node in enumerate(nodes):
                node_id = node.get("id", node.get("key", {}).get("id", str(idx)))
                node_id_to_idx[node_id] = idx
                
                # Extract features
                props = node.get("properties", {})
                if isinstance(props, str):
                    try:
                        import json
                        props = json.loads(props)
                    except:
                        props = {}
                
                features = self._extract_node_features(node, props)
                node_features.append(features)
            
            # Create edge index
            edge_index = []
            
            for edge in edges:
                source_id = edge.get("source_id", edge.get("source", ""))
                target_id = edge.get("target_id", edge.get("target", ""))
                
                if source_id in node_id_to_idx and target_id in node_id_to_idx:
                    source_idx = node_id_to_idx[source_id]
                    target_idx = node_id_to_idx[target_id]
                    edge_index.append([source_idx, target_idx])
            
            if not edge_index:
                logger.warning("No valid edges found for GNN conversion")
                return None
            
            # Convert to tensors
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            data = Data(
                x=node_features_tensor,
                edge_index=edge_index_tensor
            )
            
            logger.info(f"Converted graph to PyG format: {len(nodes)} nodes, {len(edge_index)} edges")
            return data
            
        except Exception as e:
            logger.error(f"Failed to convert graph to PyG format: {e}")
            return None
    
    def prepare_training_data(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        label_key: str = "type"
    ) -> Tuple[Optional[Data], Optional[torch.Tensor]]:
        """Prepare training data with labels.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            labels: Optional dictionary mapping node IDs to labels
            label_key: Key to extract label from node if labels not provided
        
        Returns:
            Tuple of (PyG Data object, label tensor)
        """
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return None, None
        
        # Extract labels
        if labels is None:
            labels = {}
            for node in nodes:
                node_id = node.get("id", node.get("key", {}).get("id", ""))
                label = node.get(label_key, "unknown")
                labels[node_id] = label
        
        # Create class mapping
        unique_labels = sorted(set(labels.values()))
        if not self.class_to_idx:
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_labels)}
            self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        # Create label tensor
        node_labels = []
        for node in nodes:
            node_id = node.get("id", node.get("key", {}).get("id", ""))
            label = labels.get(node_id, "unknown")
            class_idx = self.class_to_idx.get(label, 0)
            node_labels.append(class_idx)
        
        label_tensor = torch.tensor(node_labels, dtype=torch.long)
        
        return data, label_tensor
    
    def train(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        labels: Optional[Dict[str, str]] = None,
        label_key: str = "type",
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train the node classifier.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            labels: Optional dictionary mapping node IDs to labels
            label_key: Key to extract label from node if labels not provided
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size (None for single graph)
        
        Returns:
            Dictionary with training results
        """
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Prepare data
        data, labels_tensor = self.prepare_training_data(nodes, edges, labels, label_key)
        if data is None or labels_tensor is None:
            return {"error": "Failed to prepare training data"}
        
        # Initialize model if needed
        if self.model is None:
            self.num_node_features = data.x.shape[1]
            self.num_classes = len(self.class_to_idx)
            self.model = NodeClassifier(
                num_node_features=self.num_node_features,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_gat=self.use_gat,
                use_sage=self.use_sage
            ).to(self.device)
        
        # Move data to device
        data = data.to(self.device)
        labels_tensor = labels_tensor.to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.NLLLoss()
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            loss = criterion(out, labels_tensor)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            accuracy = (pred == labels_tensor).float().mean().item()
        
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "final_loss": train_losses[-1] if train_losses else None,
            "num_classes": self.num_classes,
            "class_mapping": self.idx_to_class
        }
    
    def classify_nodes(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Classify nodes in a graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            Dictionary with classifications:
            - classifications: List of (node_id, predicted_class, confidence)
            - class_mapping: Mapping from class index to class name
        """
        if self.model is None:
            return {"error": "Model not trained. Call train() first."}
        
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Convert graph
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return {"error": "Failed to convert graph to PyG format"}
        
        # Move to device
        data = data.to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            probs = torch.exp(out)
            pred = out.argmax(dim=1)
            confidence = probs.max(dim=1)[0]
        
        # Format results
        classifications = []
        for i, node in enumerate(nodes):
            node_id = node.get("id", node.get("key", {}).get("id", str(i)))
            class_idx = pred[i].item()
            class_name = self.idx_to_class.get(class_idx, "unknown")
            conf = confidence[i].item()
            
            classifications.append({
                "node_id": node_id,
                "predicted_class": class_name,
                "class_index": class_idx,
                "confidence": conf
            })
        
        logger.info(f"Classified {len(classifications)} nodes")
        
        return {
            "classifications": classifications,
            "class_mapping": self.idx_to_class,
            "num_nodes": len(nodes)
        }
    
    def save_model(self, path: str):
        """Save trained model.
        
        Args:
            path: Path to save model
        """
        if self.model is None:
            logger.warning("No model to save")
            return
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "num_node_features": self.num_node_features,
            "num_classes": self.num_classes,
            "class_to_idx": self.class_to_idx,
            "idx_to_class": self.idx_to_class,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_gat": self.use_gat,
            "use_sage": self.use_sage
        }
        
        torch.save(state, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load trained model.
        
        Args:
            path: Path to load model from
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required")
        
        state = torch.load(path, map_location=self.device)
        
        self.num_node_features = state["num_node_features"]
        self.num_classes = state["num_classes"]
        self.class_to_idx = state["class_to_idx"]
        self.idx_to_class = state["idx_to_class"]
        self.hidden_dim = state["hidden_dim"]
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self.use_gat = state["use_gat"]
        self.use_sage = state["use_sage"]
        
        self.model = NodeClassifier(
            num_node_features=self.num_node_features,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gat=self.use_gat,
            use_sage=self.use_sage
        ).to(self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Loaded model from {path}")

