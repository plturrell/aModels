"""GNN-based Schema Matching.

This module uses GNNs for cross-system schema alignment and matching.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    from torch_geometric.nn.pool import global_mean_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    nn = None
    Data = Any
    Batch = Any

logger = logging.getLogger(__name__)


class SchemaMatcher(nn.Module):
    """Schema matching model using Graph Attention Networks.
    
    Matches schemas between different systems.
    """
    
    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = True
    ):
        """Initialize schema matcher.
        
        Args:
            num_node_features: Number of input node features
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network (default)
        """
        super().__init__()
        
        self.use_gat = use_gat
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = num_node_features
            else:
                in_dim = hidden_dim
            
            out_dim = hidden_dim
            
            if use_gat:
                self.convs.append(GATConv(in_dim, out_dim, heads=4, dropout=dropout, concat=False))
            else:
                self.convs.append(GCNConv(in_dim, out_dim))
        
        # Matching head
        self.match_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x1, edge_index1, x2, edge_index2):
        """Forward pass for schema matching.
        
        Args:
            x1: Node features of first schema
            edge_index1: Edge connectivity of first schema
            x2: Node features of second schema
            edge_index2: Edge connectivity of second schema
        
        Returns:
            Schema-level similarity score
        """
        # Encode both schemas
        z1 = x1
        for i, conv in enumerate(self.convs):
            z1 = conv(z1, edge_index1)
            if i < len(self.convs) - 1:
                z1 = F.relu(z1)
                z1 = self.dropout(z1)
        
        z2 = x2
        for i, conv in enumerate(self.convs):
            z2 = conv(z2, edge_index2)
            if i < len(self.convs) - 1:
                z2 = F.relu(z2)
                z2 = self.dropout(z2)
        
        # Pool to graph-level embeddings
        # For single graphs, use mean pooling
        z1_pooled = z1.mean(dim=0)  # [hidden_dim]
        z2_pooled = z2.mean(dim=0)  # [hidden_dim]
        
        # Compute similarity
        z_combined = torch.cat([z1_pooled, z2_pooled], dim=0)  # [hidden_dim * 2]
        similarity = self.match_head(z_combined).squeeze()
        
        return similarity
    
    def encode_schema(self, x, edge_index):
        """Encode a schema to embedding.
        
        Args:
            x: Node features
            edge_index: Edge connectivity
        
        Returns:
            Graph-level embedding
        """
        z = x
        for i, conv in enumerate(self.convs):
            z = conv(z, edge_index)
            if i < len(self.convs) - 1:
                z = F.relu(z)
                z = self.dropout(z)
        
        # Pool to graph-level
        z_pooled = z.mean(dim=0)
        return z_pooled


class GNNSchemaMatcher:
    """GNN-based schema matcher for cross-system alignment.
    
    Matches schemas between different systems (e.g., Murex and SAP).
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = True,
        device: Optional[str] = None
    ):
        """Initialize GNN schema matcher.
        
        Args:
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network (default)
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN schema matching")
        
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
        
        logger.info(f"Initialized GNN Schema Matcher (device: {self.device})")
    
    def _extract_node_features(
        self,
        node: Dict[str, Any],
        props: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector from a node."""
        features = []
        
        # Node type encoding
        node_type = node.get("type", node.get("label", "unknown"))
        type_features = [0.0] * 10
        type_map = {
            "table": 0, "column": 1, "view": 2, "database": 3, "schema": 4,
            "sql": 5, "control-m": 6, "project": 7, "system": 8, "information-system": 9
        }
        if node_type in type_map:
            type_idx = type_map[node_type]
            if type_idx < len(type_features):
                type_features[type_idx] = 1.0
        features.extend(type_features)
        
        # Property-based features
        if isinstance(props, dict):
            features.append(float(props.get("column_count", 0)))
            features.append(float(props.get("row_count", 0)))
            features.append(float(props.get("data_type_entropy", 0)))
            features.append(float(props.get("nullable_ratio", 0)))
            # Add semantic features if available
            features.append(float(props.get("name_similarity", 0)))
        else:
            features.extend([0.0] * 5)
        
        # Pad to fixed size
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
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
        schema_pairs: List[Dict[str, Any]],
        labels: List[float],
        epochs: int = 100,
        lr: float = 0.01
    ) -> Dict[str, Any]:
        """Train the schema matcher.
        
        Args:
            schema_pairs: List of schema pairs, each with 'schema1' and 'schema2' (nodes/edges)
            labels: List of similarity labels (0.0 to 1.0)
            epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Dictionary with training results
        """
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        if len(schema_pairs) != len(labels):
            return {"error": "Number of schema pairs must match number of labels"}
        
        # Convert first pair to get feature dimension
        if not schema_pairs:
            return {"error": "No schema pairs provided"}
        
        first_pair = schema_pairs[0]
        temp_data = self.convert_graph_to_pyg_data(
            first_pair["schema1"].get("nodes", []),
            first_pair["schema1"].get("edges", [])
        )
        if temp_data is None:
            return {"error": "Failed to convert schema"}
        
        # Initialize model
        if self.model is None:
            self.num_node_features = temp_data.x.shape[1]
            self.model = SchemaMatcher(
                num_node_features=self.num_node_features,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_gat=self.use_gat
            ).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        self.model.train()
        train_losses = []
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for pair, label in zip(schema_pairs, labels):
                # Convert schemas
                data1 = self.convert_graph_to_pyg_data(
                    pair["schema1"].get("nodes", []),
                    pair["schema1"].get("edges", [])
                )
                data2 = self.convert_graph_to_pyg_data(
                    pair["schema2"].get("nodes", []),
                    pair["schema2"].get("edges", [])
                )
                
                if data1 is None or data2 is None:
                    continue
                
                # Move to device
                data1 = data1.to(self.device)
                data2 = data2.to(self.device)
                label_tensor = torch.tensor([label], dtype=torch.float32, device=self.device)
                
                # Forward pass
                optimizer.zero_grad()
                pred = self.model(data1.x, data1.edge_index, data2.x, data2.edge_index)
                loss = criterion(pred, label_tensor)
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                train_losses.append(avg_loss)
                
                if (epoch + 1) % 20 == 0:
                    logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Training complete. Final loss: {train_losses[-1]:.4f if train_losses else 0.0:.4f}")
        
        return {
            "final_loss": train_losses[-1] if train_losses else None,
            "num_pairs": len(schema_pairs)
        }
    
    def match_schemas(
        self,
        schema1_nodes: List[Dict[str, Any]],
        schema1_edges: List[Dict[str, Any]],
        schema2_nodes: List[Dict[str, Any]],
        schema2_edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Match two schemas.
        
        Args:
            schema1_nodes: Nodes of first schema
            schema1_edges: Edges of first schema
            schema2_nodes: Nodes of second schema
            schema2_edges: Edges of second schema
        
        Returns:
            Dictionary with matching results
        """
        if self.model is None:
            return {"error": "Model not trained. Call train() first."}
        
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Convert schemas
        data1 = self.convert_graph_to_pyg_data(schema1_nodes, schema1_edges)
        data2 = self.convert_graph_to_pyg_data(schema2_nodes, schema2_edges)
        
        if data1 is None or data2 is None:
            return {"error": "Failed to convert schemas"}
        
        # Move to device
        data1 = data1.to(self.device)
        data2 = data2.to(self.device)
        
        # Match
        self.model.eval()
        with torch.no_grad():
            similarity = self.model(data1.x, data1.edge_index, data2.x, data2.edge_index)
            similarity_score = similarity.item()
        
        logger.info(f"Schema similarity: {similarity_score:.4f}")
        
        return {
            "similarity": similarity_score,
            "schema1_nodes": len(schema1_nodes),
            "schema1_edges": len(schema1_edges),
            "schema2_nodes": len(schema2_nodes),
            "schema2_edges": len(schema2_edges)
        }
    
    def find_best_matches(
        self,
        query_nodes: List[Dict[str, Any]],
        query_edges: List[Dict[str, Any]],
        candidate_schemas: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Find best matching schemas.
        
        Args:
            query_nodes: Nodes of query schema
            query_edges: Edges of query schema
            candidate_schemas: List of candidate schemas, each with 'nodes' and 'edges'
            top_k: Number of top matches to return
        
        Returns:
            Dictionary with best matches
        """
        matches = []
        
        for i, candidate in enumerate(candidate_schemas):
            result = self.match_schemas(
                query_nodes,
                query_edges,
                candidate.get("nodes", []),
                candidate.get("edges", [])
            )
            
            if "error" not in result:
                matches.append({
                    "index": i,
                    "similarity": result["similarity"],
                    "schema": candidate
                })
        
        # Sort by similarity
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        top_matches = matches[:top_k]
        
        return {
            "query_schema": {"nodes": len(query_nodes), "edges": len(query_edges)},
            "matches": top_matches,
            "num_candidates": len(candidate_schemas)
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
        
        self.model = SchemaMatcher(
            num_node_features=self.num_node_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_gat=self.use_gat
        ).to(self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Loaded model from {path}")

