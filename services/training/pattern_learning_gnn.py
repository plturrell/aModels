"""Graph Neural Network (GNN) for relationship pattern learning.

This module implements GNN-based pattern learning for knowledge graph structures,
learning embeddings for nodes (tables, columns) and edges (relationships).
"""

import logging
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    # Fallback for when PyTorch Geometric is not available
    try:
        import torch
        import torch.nn as nn
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

logger = logging.getLogger(__name__)


class GNNRelationshipPatternLearner:
    """GNN-based learner for relationship patterns in knowledge graphs."""
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False
    ):
        """Initialize GNN pattern learner.
        
        Args:
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network (GAT) instead of GCN
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        
        if HAS_PYG:
            self._build_model()
        else:
            logger.warning("PyTorch Geometric not available. GNN features will be limited.")
    
    def _build_model(self):
        """Build the GNN model."""
        if not HAS_PYG:
            return
        
        # Node feature dimension (will be determined from input)
        # For now, use a placeholder
        input_dim = 32  # Will be adjusted based on actual node features
        
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                in_dim = input_dim
            else:
                in_dim = self.hidden_dim
            
            out_dim = self.hidden_dim if i < self.num_layers - 1 else self.hidden_dim
            
            if self.use_gat:
                layer = GATConv(in_dim, out_dim, heads=1, dropout=self.dropout)
            else:
                layer = GCNConv(in_dim, out_dim)
            layers.append(layer)
        
        self.model = nn.Sequential(*layers).to(self.device)
        logger.info(f"Built GNN model with {self.num_layers} layers, hidden_dim={self.hidden_dim}")
    
    def convert_graph_to_pyg_data(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Optional[Data]:
        """Convert knowledge graph nodes/edges to PyG Data format.
        
        Args:
            nodes: List of graph nodes with properties
            edges: List of graph edges with source/target IDs
        
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
                
                # Extract features from node
                props = node.get("properties", {})
                if isinstance(props, str):
                    try:
                        props = json.loads(props)
                    except:
                        props = {}
                
                # Create feature vector (simplified - can be enhanced)
                features = self._extract_node_features(node, props)
                node_features.append(features)
            
            # Create edge index
            edge_index = []
            edge_attr = []
            
            for edge in edges:
                source_id = edge.get("source_id", edge.get("source", ""))
                target_id = edge.get("target_id", edge.get("target", ""))
                
                if source_id in node_id_to_idx and target_id in node_id_to_idx:
                    source_idx = node_id_to_idx[source_id]
                    target_idx = node_id_to_idx[target_id]
                    edge_index.append([source_idx, target_idx])
                    
                    # Extract edge attributes
                    edge_props = edge.get("properties", {})
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except:
                            edge_props = {}
                    
                    edge_features = self._extract_edge_features(edge, edge_props)
                    edge_attr.append(edge_features)
            
            if not edge_index:
                logger.warning("No valid edges found for GNN conversion")
                return None
            
            # Convert to tensors
            node_features_tensor = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            
            if edge_attr:
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)
            else:
                edge_attr_tensor = None
            
            data = Data(
                x=node_features_tensor,
                edge_index=edge_index_tensor,
                edge_attr=edge_attr_tensor
            )
            
            logger.info(f"Converted graph to PyG format: {len(nodes)} nodes, {len(edge_index)} edges")
            return data
            
        except Exception as e:
            logger.error(f"Failed to convert graph to PyG format: {e}")
            return None
    
    def _extract_node_features(self, node: Dict[str, Any], props: Dict[str, Any]) -> List[float]:
        """Extract feature vector from a node.
        
        Args:
            node: Node dictionary
            props: Node properties
        
        Returns:
            Feature vector as list of floats
        """
        features = []
        
        # Node type encoding (one-hot-like)
        node_type = node.get("type", node.get("label", "unknown"))
        type_features = [0.0] * 10  # Support up to 10 types
        type_map = {
            "table": 0, "column": 1, "sql": 2, "control-m": 3, "project": 4,
            "system": 5, "information-system": 6, "petri-net": 7
        }
        if node_type in type_map:
            type_idx = type_map[node_type]
            if type_idx < len(type_features):
                type_features[type_idx] = 1.0
        features.extend(type_features)
        
        # Property-based features
        if isinstance(props, dict):
            # Numeric properties
            features.append(float(props.get("column_count", 0)))
            features.append(float(props.get("row_count", 0)))
            features.append(float(props.get("metadata_entropy", 0)))
            features.append(float(props.get("kl_divergence", 0)))
        else:
            features.extend([0.0] * 4)
        
        # Add more features if needed
        features.extend([0.0] * (32 - len(features)))  # Pad to 32 dimensions
        return features[:32]  # Ensure exactly 32 dimensions
    
    def _extract_edge_features(self, edge: Dict[str, Any], props: Dict[str, Any]) -> List[float]:
        """Extract feature vector from an edge.
        
        Args:
            edge: Edge dictionary
            props: Edge properties
        
        Returns:
            Feature vector as list of floats
        """
        features = []
        
        # Edge type encoding
        edge_type = edge.get("type", edge.get("label", "unknown"))
        type_features = [0.0] * 10  # Support up to 10 edge types
        type_map = {
            "HAS_COLUMN": 0, "DATA_FLOW": 1, "HAS_SQL": 2, "DEPENDS_ON": 3,
            "PROCESSES_BEFORE": 4, "HAS_PETRI_NET": 5
        }
        if edge_type in type_map:
            type_idx = type_map[edge_type]
            if type_idx < len(type_features):
                type_features[type_idx] = 1.0
        features.extend(type_features)
        
        # Property-based features
        if isinstance(props, dict):
            features.append(float(props.get("sequence_order", 0)))
            features.append(float(props.get("confidence", 0)))
        else:
            features.extend([0.0] * 2)
        
        # Pad to fixed size
        features.extend([0.0] * (12 - len(features)))
        return features[:12]
    
    def learn_patterns(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn relationship patterns using GNN.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            Dictionary with learned patterns:
            - node_embeddings: Learned node embeddings
            - edge_predictions: Predicted edge types
            - relationship_patterns: Learned relationship patterns
        """
        if not HAS_PYG:
            logger.warning("GNN learning skipped: PyTorch Geometric not available")
            return {
                "node_embeddings": [],
                "edge_predictions": [],
                "relationship_patterns": {},
                "error": "PyTorch Geometric not available"
            }
        
        try:
            # Convert graph to PyG format
            data = self.convert_graph_to_pyg_data(nodes, edges)
            if data is None:
                return {
                    "node_embeddings": [],
                    "edge_predictions": [],
                    "relationship_patterns": {},
                    "error": "Failed to convert graph to PyG format"
                }
            
            # Move to device
            data = data.to(self.device)
            
            # Forward pass through GNN
            self.model.eval()
            with torch.no_grad():
                x = data.x
                for i, layer in enumerate(self.model):
                    x = layer(x, data.edge_index)
                    if i < len(self.model) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=False)
                
                node_embeddings = x.cpu().numpy()
            
            # Extract relationship patterns
            relationship_patterns = self._extract_relationship_patterns(nodes, edges, node_embeddings)
            
            # Predict edge types (simplified)
            edge_predictions = self._predict_edge_types(data, node_embeddings)
            
            logger.info(
                f"GNN learning complete: {len(node_embeddings)} node embeddings, "
                f"{len(relationship_patterns)} relationship patterns"
            )
            
            return {
                "node_embeddings": node_embeddings.tolist(),
                "edge_predictions": edge_predictions,
                "relationship_patterns": relationship_patterns,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "embedding_dim": self.hidden_dim
            }
            
        except Exception as e:
            logger.error(f"GNN learning failed: {e}", exc_info=True)
            return {
                "node_embeddings": [],
                "edge_predictions": [],
                "relationship_patterns": {},
                "error": str(e)
            }
    
    def _extract_relationship_patterns(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Extract relationship patterns from learned embeddings.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            embeddings: Learned node embeddings
        
        Returns:
            Dictionary with relationship patterns
        """
        patterns = {}
        
        # Group edges by type
        edges_by_type = {}
        for edge in edges:
            edge_type = edge.get("type", edge.get("label", "unknown"))
            if edge_type not in edges_by_type:
                edges_by_type[edge_type] = []
            edges_by_type[edge_type].append(edge)
        
        # Analyze patterns for each edge type
        for edge_type, edge_list in edges_by_type.items():
            # Calculate average embedding similarity for this edge type
            similarities = []
            for edge in edge_list:
                source_id = edge.get("source_id", edge.get("source", ""))
                target_id = edge.get("target_id", edge.get("target", ""))
                
                # Find node indices (simplified - would need proper mapping)
                source_idx = None
                target_idx = None
                for i, node in enumerate(nodes):
                    node_id = node.get("id", node.get("key", {}).get("id", ""))
                    if node_id == source_id:
                        source_idx = i
                    if node_id == target_id:
                        target_idx = i
                
                if source_idx is not None and target_idx is not None:
                    if source_idx < len(embeddings) and target_idx < len(embeddings):
                        source_emb = embeddings[source_idx]
                        target_emb = embeddings[target_idx]
                        # Cosine similarity
                        similarity = np.dot(source_emb, target_emb) / (
                            np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
                        )
                        similarities.append(float(similarity))
            
            if similarities:
                patterns[edge_type] = {
                    "count": len(edge_list),
                    "avg_similarity": float(np.mean(similarities)),
                    "std_similarity": float(np.std(similarities)),
                }
        
        return patterns
    
    def _predict_edge_types(
        self,
        data: Data,
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Predict edge types from learned embeddings.
        
        Args:
            data: PyG Data object
            embeddings: Learned node embeddings
        
        Returns:
            List of edge predictions
        """
        predictions = []
        
        if data.edge_index is not None:
            edge_index = data.edge_index.cpu().numpy()
            for i in range(edge_index.shape[1]):
                source_idx = edge_index[0, i]
                target_idx = edge_index[1, i]
                
                if source_idx < len(embeddings) and target_idx < len(embeddings):
                    source_emb = embeddings[source_idx]
                    target_emb = embeddings[target_idx]
                    
                    # Simple similarity-based prediction
                    similarity = np.dot(source_emb, target_emb) / (
                        np.linalg.norm(source_emb) * np.linalg.norm(target_emb) + 1e-8
                    )
                    
                    predictions.append({
                        "source": int(source_idx),
                        "target": int(target_idx),
                        "similarity": float(similarity),
                        "predicted_type": "DATA_FLOW" if similarity > 0.5 else "RELATED"
                    })
        
        return predictions
    
    def predict_relationship_type(
        self,
        source_node: Dict[str, Any],
        target_node: Dict[str, Any],
        embeddings: Optional[np.ndarray] = None
    ) -> str:
        """Predict relationship type between two nodes.
        
        Args:
            source_node: Source node
            target_node: Target node
            embeddings: Optional pre-computed embeddings
        
        Returns:
            Predicted relationship type
        """
        # This would use the learned model to predict relationship type
        # For now, return a default
        return "RELATED"

