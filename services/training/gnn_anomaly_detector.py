"""GNN-based Anomaly Detection.

This module detects structural anomalies in graph patterns using Graph Autoencoders.
"""

import logging
import os
from typing import Dict, List, Optional, Any
import numpy as np
from .gnn_features import extract_node_features as _shared_extract_node_features

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    nn = None
    Data = Any
    Batch = Any

logger = logging.getLogger(__name__)


class GraphAutoencoder(nn.Module):
    """Graph Autoencoder for anomaly detection.
    
    Encodes and reconstructs graphs to detect anomalies based on reconstruction error.
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """Initialize graph autoencoder.
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension
            embedding_dim: Embedding dimension (bottleneck)
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Encoder
        self.encoder_convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = num_node_features
            else:
                in_dim = hidden_dim
            
            out_dim = embedding_dim if i == num_layers - 1 else hidden_dim
            self.encoder_convs.append(GCNConv(in_dim, out_dim))
        
        # Decoder
        self.decoder_convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                in_dim = embedding_dim
            else:
                in_dim = hidden_dim
            
            out_dim = num_node_features if i == num_layers - 1 else hidden_dim
            self.decoder_convs.append(GCNConv(in_dim, out_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(self, x, edge_index):
        """Encode nodes to embeddings.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
        
        Returns:
            Node embeddings
        """
        z = x
        for i, conv in enumerate(self.encoder_convs):
            z = conv(z, edge_index)
            if i < len(self.encoder_convs) - 1:
                z = F.relu(z)
                z = self.dropout(z)
        return z
    
    def decode(self, z, edge_index):
        """Decode embeddings back to features.
        
        Args:
            z: Node embeddings
            edge_index: Edge connectivity
        
        Returns:
            Reconstructed node features
        """
        x_recon = z
        for i, conv in enumerate(self.decoder_convs):
            x_recon = conv(x_recon, edge_index)
            if i < len(self.decoder_convs) - 1:
                x_recon = F.relu(x_recon)
                x_recon = self.dropout(x_recon)
        return x_recon
    
    def forward(self, x, edge_index):
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
        
        Returns:
            Reconstructed node features
        """
        z = self.encode(x, edge_index)
        x_recon = self.decode(z, edge_index)
        return x_recon


class GNNAnomalyDetector:
    """GNN-based anomaly detector for knowledge graphs.
    
    Detects structural anomalies in graph patterns.
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        embedding_dim: int = 16,
        num_layers: int = 2,
        dropout: float = 0.1,
        anomaly_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """Initialize GNN anomaly detector.
        
        Args:
            hidden_dim: Hidden dimension
            embedding_dim: Embedding dimension (bottleneck)
            num_layers: Number of GNN layers
            dropout: Dropout rate
            anomaly_threshold: Threshold for anomaly detection (reconstruction error)
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN anomaly detection")
        
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.anomaly_threshold = anomaly_threshold
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.num_node_features = None
        
        logger.info(f"Initialized GNN Anomaly Detector (device: {self.device})")
    
    def _extract_node_features(
        self,
        node: Dict[str, Any],
        props: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector from a node."""
        return _shared_extract_node_features(node, props)
    
    def convert_graph_to_pyg_data(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Optional[Data]:
        """Convert knowledge graph to PyG Data format."""
        if not HAS_PYG:
            return None
        
        try:
            # Create node ID mapping
            node_id_to_idx = {}
            node_features = []
            
            for idx, node in enumerate(nodes):
                node_id = node.get("id", node.get("key", {}).get("id", str(idx)))
                node_id_to_idx[node_id] = idx
                
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
                logger.warning("No valid edges found")
                return None
            
            # Convert to tensors
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            data = Data(
                x=node_features_tensor,
                edge_index=edge_index_tensor
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to convert graph: {e}")
            return None
    
    def train(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        epochs: int = 100,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Train the anomaly detector on normal patterns.
        
        Args:
            nodes: List of graph nodes (normal patterns)
            edges: List of graph edges (normal patterns)
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Dictionary with training results
        """
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Convert graph
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return {"error": "Failed to convert graph"}
        
        # Initialize model
        if self.model is None:
            self.num_node_features = data.x.shape[1]
            self.model = GraphAutoencoder(
                num_node_features=self.num_node_features,
                hidden_dim=self.hidden_dim,
                embedding_dim=self.embedding_dim,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)
        
        # Move to device
        data = data.to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            x_recon = self.model(data.x, data.edge_index)
            loss = criterion(x_recon, data.x)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        logger.info(f"Training complete. Final loss: {train_losses[-1]:.4f}")
        
        return {
            "final_loss": train_losses[-1] if train_losses else None,
            "num_nodes": len(nodes),
            "num_edges": len(edges)
        }
    
    def detect_anomalies(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Detect anomalies in a graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            Dictionary with anomaly detection results
        """
        if self.model is None:
            return {"error": "Model not trained. Call train() first."}
        
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Convert graph
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return {"error": "Failed to convert graph"}
        
        # Move to device
        data = data.to(self.device)
        
        # Reconstruct
        self.model.eval()
        with torch.no_grad():
            x_recon = self.model(data.x, data.edge_index)
            
            # Calculate reconstruction error per node
            reconstruction_errors = F.mse_loss(x_recon, data.x, reduction='none').mean(dim=1)
            reconstruction_errors = reconstruction_errors.cpu().numpy()
        
        # Identify anomalies
        anomalies = []
        for idx, node in enumerate(nodes):
            node_id = node.get("id", node.get("key", {}).get("id", str(idx)))
            error = float(reconstruction_errors[idx])
            is_anomaly = error > self.anomaly_threshold
            
            if is_anomaly:
                anomalies.append({
                    "node_id": node_id,
                    "reconstruction_error": error,
                    "anomaly_score": error,
                    "is_anomaly": True
                })
        
        # Calculate graph-level anomaly score
        graph_anomaly_score = float(np.mean(reconstruction_errors))
        is_graph_anomaly = graph_anomaly_score > self.anomaly_threshold
        
        logger.info(f"Detected {len(anomalies)} anomalous nodes out of {len(nodes)}")
        
        return {
            "anomalies": anomalies,
            "num_anomalies": len(anomalies),
            "graph_anomaly_score": graph_anomaly_score,
            "is_graph_anomaly": is_graph_anomaly,
            "threshold": self.anomaly_threshold,
            "num_nodes": len(nodes)
        }
    
    def set_threshold(self, threshold: float):
        """Set anomaly detection threshold.
        
        Args:
            threshold: New threshold value
        """
        self.anomaly_threshold = threshold
        logger.info(f"Set anomaly threshold to {threshold}")
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "num_node_features": self.num_node_features,
            "hidden_dim": self.hidden_dim,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "anomaly_threshold": self.anomaly_threshold
        }
        
        torch.save(state, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required")
        
        state = torch.load(path, map_location=self.device)
        
        self.num_node_features = state["num_node_features"]
        self.hidden_dim = state["hidden_dim"]
        self.embedding_dim = state["embedding_dim"]
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self.anomaly_threshold = state.get("anomaly_threshold", 0.5)
        
        self.model = GraphAutoencoder(
            num_node_features=self.num_node_features,
            hidden_dim=self.hidden_dim,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Loaded model from {path}")

