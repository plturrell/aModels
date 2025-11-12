"""GNN-based Link Prediction.

This module implements link prediction using Graph Neural Networks to predict
missing relationships or suggest new mappings.
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
    from torch.cuda.amp import autocast, GradScaler
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    from torch_geometric.nn.models import GAE, VGAE
    HAS_PYG = True
    HAS_AMP = True
except ImportError:
    HAS_PYG = False
    HAS_AMP = False
    autocast = None
    GradScaler = None
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


class LinkPredictorEncoder(nn.Module):
    """Encoder for link prediction.
    
    Uses GCN or GAT to encode nodes for link prediction.
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False
    ):
        """Initialize encoder.
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network
        """
        super().__init__()
        
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = num_node_features
            else:
                in_dim = hidden_dim
            
            out_dim = hidden_dim
            
            if use_gat:
                self.convs.append(GATConv(in_dim, out_dim, heads=1, dropout=dropout))
            else:
                self.convs.append(GCNConv(in_dim, out_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
        
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x


class LinkPredictor(nn.Module):
    """Link prediction model.
    
    Predicts edge probabilities between node pairs.
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False
    ):
        """Initialize link predictor.
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network
        """
        super().__init__()
        
        self.encoder = LinkPredictorEncoder(
            num_node_features, hidden_dim, num_layers, dropout, use_gat
        )
        
        # Decoder for link prediction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_label_index=None):
        """Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge connectivity for encoding
            edge_label_index: Edge pairs to predict (if None, uses edge_index)
        
        Returns:
            Edge probabilities
        """
        # Encode nodes
        z = self.encoder(x, edge_index)
        
        # Get edge pairs to predict
        if edge_label_index is None:
            edge_label_index = edge_index
        
        # Get embeddings for source and target nodes
        source_emb = z[edge_label_index[0]]
        target_emb = z[edge_label_index[1]]
        
        # Concatenate and decode
        edge_emb = torch.cat([source_emb, target_emb], dim=1)
        edge_probs = self.decoder(edge_emb).squeeze()
        
        return edge_probs


class GNNLinkPredictor:
    """GNN-based link predictor for knowledge graphs.
    
    Predicts missing relationships or suggests new mappings.
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False,
        device: Optional[str] = None
    ):
        """Initialize GNN link predictor.
        
        Args:
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN link prediction")
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.num_node_features = None
        
        logger.info(f"Initialized GNN Link Predictor (device: {self.device})")
    
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
        lr: float = 0.01,
        neg_samples: int = 1,
        use_amp: bool = False,
        scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Train the link predictor.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges (positive examples)
            epochs: Number of training epochs
            lr: Learning rate
            neg_samples: Number of negative samples per positive edge
        
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
            self.model = LinkPredictor(
                num_node_features=self.num_node_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_gat=self.use_gat
            ).to(self.device)
        
        # Move to device
        data = data.to(self.device)
        
        # Create negative samples
        num_nodes = data.x.shape[0]
        num_pos_edges = data.edge_index.shape[1]
        num_neg_edges = num_pos_edges * neg_samples
        
        neg_edge_index = torch.randint(0, num_nodes, (2, num_neg_edges), device=self.device)
        
        # Combine positive and negative edges
        pos_labels = torch.ones(num_pos_edges, device=self.device)
        neg_labels = torch.zeros(num_neg_edges, device=self.device)
        
        edge_label_index = torch.cat([data.edge_index, neg_edge_index], dim=1)
        edge_labels = torch.cat([pos_labels, neg_labels])
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        # Training loop with AMP support
        self.model.train()
        train_losses = []
        use_amp_training = use_amp and HAS_AMP and scaler is not None
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            if use_amp_training:
                # Use AMP autocast for forward pass
                with autocast():
                    pred = self.model(data.x, data.edge_index, edge_label_index)
                    loss = criterion(pred, edge_labels)
                
                # Scale loss and backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training without AMP
                pred = self.model(data.x, data.edge_index, edge_label_index)
                loss = criterion(pred, edge_labels)
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            
            if (epoch + 1) % 20 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")
        
        # Evaluate
        self.model.eval()
        with torch.no_grad():
            pred = self.model(data.x, data.edge_index, edge_label_index)
            accuracy = ((pred > 0.5).float() == edge_labels).float().mean().item()
        
        logger.info(f"Training complete. Accuracy: {accuracy:.4f}")
        
        return {
            "accuracy": accuracy,
            "final_loss": train_losses[-1] if train_losses else None
        }
    
    def predict_links(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        candidate_pairs: Optional[List[Tuple[str, str]]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """Predict missing links.
        
        Args:
            nodes: List of graph nodes
            edges: List of existing edges
            candidate_pairs: Optional list of (source_id, target_id) pairs to evaluate
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with link predictions
        """
        if self.model is None:
            return {"error": "Model not trained. Call train() first."}
        
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Convert graph
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return {"error": "Failed to convert graph"}
        
        # Create node ID to index mapping
        node_id_to_idx = {}
        for idx, node in enumerate(nodes):
            node_id = node.get("id", node.get("key", {}).get("id", str(idx)))
            node_id_to_idx[node_id] = idx
        
        # Get candidate pairs
        if candidate_pairs is None:
            # Generate all possible pairs (excluding existing edges)
            num_nodes = len(nodes)
            existing_edges = set()
            for edge in edges:
                source_id = edge.get("source_id", edge.get("source", ""))
                target_id = edge.get("target_id", edge.get("target", ""))
                if source_id in node_id_to_idx and target_id in node_id_to_idx:
                    source_idx = node_id_to_idx[source_id]
                    target_idx = node_id_to_idx[target_id]
                    existing_edges.add((source_idx, target_idx))
            
            # Generate candidate pairs
            candidate_pairs = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j and (i, j) not in existing_edges:
                        candidate_pairs.append((i, j))
        else:
            # Convert node IDs to indices
            candidate_indices = []
            for source_id, target_id in candidate_pairs:
                if source_id in node_id_to_idx and target_id in node_id_to_idx:
                    candidate_indices.append((
                        node_id_to_idx[source_id],
                        node_id_to_idx[target_id]
                    ))
            candidate_pairs = candidate_indices
        
        if not candidate_pairs:
            return {"error": "No candidate pairs to evaluate"}
        
        # Move to device
        data = data.to(self.device)
        edge_label_index = torch.tensor(candidate_pairs, dtype=torch.long).t().to(self.device)
        
        # Predict
        self.model.eval()
        with torch.no_grad():
            probs = self.model(data.x, data.edge_index, edge_label_index)
        
        # Get top-k predictions
        top_k_indices = torch.topk(probs, min(top_k, len(probs))).indices.cpu().numpy()
        
        # Format results
        predictions = []
        for idx in top_k_indices:
            source_idx, target_idx = candidate_pairs[idx]
            source_id = nodes[source_idx].get("id", nodes[source_idx].get("key", {}).get("id", str(source_idx)))
            target_id = nodes[target_idx].get("id", nodes[target_idx].get("key", {}).get("id", str(target_idx)))
            prob = probs[idx].item()
            
            predictions.append({
                "source_id": source_id,
                "target_id": target_id,
                "probability": prob,
                "source_idx": int(source_idx),
                "target_idx": int(target_idx)
            })
        
        logger.info(f"Predicted {len(predictions)} links")
        
        return {
            "predictions": predictions,
            "num_candidates": len(candidate_pairs)
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "num_node_features": self.num_node_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_gat": self.use_gat
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
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self.use_gat = state["use_gat"]
        
        self.model = LinkPredictor(
            num_node_features=self.num_node_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gat=self.use_gat
        ).to(self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Loaded model from {path}")

