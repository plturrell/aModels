"""GNN-based Graph Embeddings.

This module generates embeddings for similarity search and pattern matching.
"""

import logging
import os
from typing import Dict, List, Optional, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    nn = None
    Data = Any
    Batch = Any

logger = logging.getLogger(__name__)


class GraphEmbedder(nn.Module):
    """Graph-level embedder.
    
    Generates embeddings for entire graphs or subgraphs.
    """
    
    def __init__(
        self,
        num_node_features: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_sage: bool = True,
        pool_type: str = "mean"
    ):
        """Initialize graph embedder.
        
        Args:
            num_node_features: Number of input node features
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_sage: Whether to use GraphSAGE (default)
            pool_type: Pooling type ('mean', 'max', 'add')
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.use_sage = use_sage
        self.pool_type = pool_type
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_dim = num_node_features
            else:
                in_dim = hidden_dim
            
            out_dim = hidden_dim if i < num_layers - 1 else embedding_dim
            
            if use_sage:
                self.convs.append(GraphSAGE(in_dim, out_dim, num_layers=1))
            else:
                self.convs.append(GCNConv(in_dim, out_dim))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, batch=None):
        """Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch vector for graph-level pooling [num_nodes]
        
        Returns:
            Graph embeddings [num_graphs, embedding_dim] or node embeddings [num_nodes, embedding_dim]
        """
        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        
        # Pool to graph-level if batch provided
        if batch is not None:
            if self.pool_type == "mean":
                return global_mean_pool(x, batch)
            elif self.pool_type == "max":
                return global_max_pool(x, batch)
            elif self.pool_type == "add":
                return global_add_pool(x, batch)
            else:
                return global_mean_pool(x, batch)
        
        return x


class GNNEmbedder:
    """GNN-based embedder for knowledge graphs.
    
    Generates embeddings for similarity search and pattern matching.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_sage: bool = True,
        pool_type: str = "mean",
        device: Optional[str] = None
    ):
        """Initialize GNN embedder.
        
        Args:
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_sage: Whether to use GraphSAGE
            pool_type: Pooling type ('mean', 'max', 'add')
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GNN embeddings")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_sage = use_sage
        self.pool_type = pool_type
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.num_node_features = None
        
        logger.info(f"Initialized GNN Embedder (device: {self.device}, embedding_dim: {embedding_dim})")
    
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
            features.append(float(props.get("metadata_entropy", 0)))
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
    
    def generate_embeddings(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        graph_level: bool = True
    ) -> Dict[str, Any]:
        """Generate embeddings for a graph.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            graph_level: If True, return graph-level embedding; if False, return node embeddings
        
        Returns:
            Dictionary with embeddings
        """
        if not HAS_PYG:
            return {"error": "PyTorch Geometric not available"}
        
        # Initialize model if needed
        if self.model is None:
            # Convert once to get feature dimension
            temp_data = self.convert_graph_to_pyg_data(nodes, edges)
            if temp_data is None:
                return {"error": "Failed to convert graph"}
            
            self.num_node_features = temp_data.x.shape[1]
            self.model = GraphEmbedder(
                num_node_features=self.num_node_features,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                dropout=self.dropout,
                use_sage=self.use_sage,
                pool_type=self.pool_type
            ).to(self.device)
            self.model.eval()
        
        # Convert graph
        data = self.convert_graph_to_pyg_data(nodes, edges)
        if data is None:
            return {"error": "Failed to convert graph"}
        
        # Move to device
        data = data.to(self.device)
        
        # Generate embeddings
        with torch.no_grad():
            if graph_level:
                # Create batch vector (single graph)
                batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
                embeddings = self.model(data.x, data.edge_index, batch)
                embeddings = embeddings.cpu().numpy()
                
                return {
                    "graph_embedding": embeddings[0].tolist(),
                    "embedding_dim": self.embedding_dim,
                    "num_nodes": len(nodes),
                    "num_edges": len(edges)
                }
            else:
                # Node-level embeddings
                embeddings = self.model(data.x, data.edge_index)
                embeddings = embeddings.cpu().numpy()
                
                # Map to node IDs
                node_embeddings = {}
                for idx, node in enumerate(nodes):
                    node_id = node.get("id", node.get("key", {}).get("id", str(idx)))
                    node_embeddings[node_id] = embeddings[idx].tolist()
                
                return {
                    "node_embeddings": node_embeddings,
                    "embedding_dim": self.embedding_dim,
                    "num_nodes": len(nodes)
                }
    
    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Cosine similarity score
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_similar_graphs(
        self,
        query_nodes: List[Dict[str, Any]],
        query_edges: List[Dict[str, Any]],
        candidate_graphs: List[Dict[str, Any]],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Find similar graphs using embeddings.
        
        Args:
            query_nodes: Nodes of query graph
            query_edges: Edges of query graph
            candidate_graphs: List of candidate graphs, each with 'nodes' and 'edges'
            top_k: Number of top similar graphs to return
        
        Returns:
            Dictionary with similar graphs and similarity scores
        """
        # Generate query embedding
        query_result = self.generate_embeddings(query_nodes, query_edges, graph_level=True)
        if "error" in query_result:
            return query_result
        
        query_embedding = query_result["graph_embedding"]
        
        # Generate embeddings for candidates
        similarities = []
        for i, candidate in enumerate(candidate_graphs):
            candidate_result = self.generate_embeddings(
                candidate.get("nodes", []),
                candidate.get("edges", []),
                graph_level=True
            )
            
            if "error" not in candidate_result:
                candidate_embedding = candidate_result["graph_embedding"]
                similarity = self.compute_similarity(query_embedding, candidate_embedding)
                similarities.append({
                    "index": i,
                    "similarity": similarity,
                    "graph": candidate
                })
        
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k
        top_similar = similarities[:top_k]
        
        return {
            "query_embedding": query_embedding,
            "similar_graphs": top_similar,
            "num_candidates": len(candidate_graphs)
        }
    
    def save_model(self, path: str):
        """Save model."""
        if self.model is None:
            logger.warning("No model to save")
            return
        
        state = {
            "model_state_dict": self.model.state_dict(),
            "num_node_features": self.num_node_features,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_sage": self.use_sage,
            "pool_type": self.pool_type
        }
        
        torch.save(state, path)
        logger.info(f"Saved model to {path}")
    
    def load_model(self, path: str):
        """Load model."""
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required")
        
        state = torch.load(path, map_location=self.device)
        
        self.num_node_features = state["num_node_features"]
        self.embedding_dim = state["embedding_dim"]
        self.hidden_dim = state["hidden_dim"]
        self.num_layers = state["num_layers"]
        self.dropout = state["dropout"]
        self.use_sage = state["use_sage"]
        self.pool_type = state["pool_type"]
        
        self.model = GraphEmbedder(
            num_node_features=self.num_node_features,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            use_sage=self.use_sage,
            pool_type=self.pool_type
        ).to(self.device)
        
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()
        
        logger.info(f"Loaded model from {path}")

