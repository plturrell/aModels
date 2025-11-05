"""Graph Neural Network (GNN) for relationship pattern learning.

This module implements GNN-based pattern learning for knowledge graph structures,
learning embeddings for nodes (tables, columns) and edges (relationships).

Domain-aware enhancements:
- Domain-specific GNN models per domain
- Domain-conditioned node/edge features
- Domain-aware pattern extraction
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import httpx

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
    """GNN-based learner for relationship patterns in knowledge graphs.
    
    Domain-aware enhancements:
    - Stores domain-specific GNN models
    - Uses domain configs for feature extraction
    - Learns patterns per domain
    """
    
    def __init__(
        self,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_gat: bool = False,
        localai_url: Optional[str] = None,
        gpu_strategy: str = "single",
        gpu_orchestrator_url: Optional[str] = None,
        device_ids: Optional[List[int]] = None
    ):
        """Initialize GNN pattern learner.
        
        Args:
            hidden_dim: Hidden dimension for GNN layers
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_gat: Whether to use Graph Attention Network (GAT) instead of GCN
            localai_url: LocalAI URL for domain config fetching (optional)
            gpu_strategy: GPU strategy - "single", "data_parallel", or "distributed"
            gpu_orchestrator_url: URL to GPU orchestrator service (optional)
            device_ids: List of GPU device IDs to use (optional, auto-detect if not provided)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat = use_gat
        self.model = None
        self.gpu_strategy = gpu_strategy
        self.gpu_orchestrator_url = gpu_orchestrator_url or os.getenv("GPU_ORCHESTRATOR_URL")
        self.device_ids = device_ids
        self.allocation_id = None
        
        # Initialize GPU allocation if orchestrator is available
        if self.gpu_orchestrator_url and gpu_strategy != "single":
            self._allocate_gpus()
        
        # Set device(s)
        if HAS_TORCH and torch.cuda.is_available():
            if self.device_ids:
                self.device = torch.device(f"cuda:{self.device_ids[0]}")
            else:
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") if HAS_TORCH else None
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_models = {}  # domain_id -> GNN model
        self.domain_configs = {}  # domain_id -> domain config
        self.domain_embeddings_cache = {}  # domain_id -> domain embedding
        
        if HAS_PYG:
            self._build_model()
        else:
            logger.warning("PyTorch Geometric not available. GNN features will be limited.")
    
    def _allocate_gpus(self):
        """Allocate GPUs from GPU orchestrator."""
        if not self.gpu_orchestrator_url:
            return
        
        try:
            import httpx
            request_data = {
                "service_name": "training-gnn",
                "workload_type": "training",
                "workload_data": {
                    "model_size": "medium",
                    "multi_gpu": self.gpu_strategy != "single"
                }
            }
            
            response = httpx.post(
                f"{self.gpu_orchestrator_url}/gpu/allocate",
                json=request_data,
                timeout=10.0
            )
            
            if response.status_code == 200:
                allocation = response.json()
                self.allocation_id = allocation.get("id")
                self.device_ids = allocation.get("gpu_ids", [])
                logger.info(f"Allocated GPUs {self.device_ids} from orchestrator (allocation ID: {self.allocation_id})")
            else:
                logger.warning(f"Failed to allocate GPUs: {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to allocate GPUs from orchestrator: {e}")
    
    def __del__(self):
        """Cleanup GPU allocation on destruction."""
        if self.allocation_id and self.gpu_orchestrator_url:
            try:
                import httpx
                httpx.post(
                    f"{self.gpu_orchestrator_url}/gpu/release",
                    json={"allocation_id": self.allocation_id},
                    timeout=5.0
                )
                logger.info(f"Released GPU allocation {self.allocation_id}")
            except Exception as e:
                logger.warning(f"Failed to release GPU allocation: {e}")
    
    def release_gpus(self):
        """Manually release GPU allocation."""
        if self.allocation_id and self.gpu_orchestrator_url:
            try:
                import httpx
                httpx.post(
                    f"{self.gpu_orchestrator_url}/gpu/release",
                    json={"allocation_id": self.allocation_id},
                    timeout=5.0
                )
                self.allocation_id = None
                logger.info("Released GPU allocation")
            except Exception as e:
                logger.warning(f"Failed to release GPU allocation: {e}")
    
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
        
        # Wrap model with DataParallel or DistributedDataParallel if multi-GPU
        if self.gpu_strategy == "data_parallel" and torch.cuda.device_count() > 1:
            if self.device_ids:
                self.model = nn.DataParallel(self.model, device_ids=self.device_ids)
            else:
                self.model = nn.DataParallel(self.model)
            logger.info(f"Wrapped model with DataParallel on {torch.cuda.device_count()} GPUs")
        elif self.gpu_strategy == "distributed":
            # DistributedDataParallel requires process group initialization
            # This should be done in the training script, not here
            logger.info("DistributedDataParallel mode - ensure process group is initialized")
        
        logger.info(f"Built GNN model with {self.num_layers} layers, hidden_dim={self.hidden_dim}, strategy={self.gpu_strategy}")
    
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
                # Note: For domain-aware learning, use convert_graph_to_pyg_data_with_domain
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
    
    def _extract_node_features(
        self, 
        node: Dict[str, Any], 
        props: Dict[str, Any],
        domain_config: Optional[Dict[str, Any]] = None
    ) -> List[float]:
        """Extract feature vector from a node.
        
        Args:
            node: Node dictionary
            props: Node properties
            domain_config: Optional domain configuration for domain-aware features
        
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
        
        # Domain-aware features (if domain config provided)
        if domain_config:
            domain_features = self._extract_domain_features(node, props, domain_config)
            features.extend(domain_features)
        else:
            # Pad for domain features
            features.extend([0.0] * 8)
        
        # Add more features if needed
        features.extend([0.0] * (40 - len(features)))  # Pad to 40 dimensions (32 + 8 domain)
        return features[:40]  # Ensure exactly 40 dimensions
    
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
    
    def _extract_domain_features(
        self,
        node: Dict[str, Any],
        props: Dict[str, Any],
        domain_config: Dict[str, Any]
    ) -> List[float]:
        """Extract domain-specific features from a node.
        
        Args:
            node: Node dictionary
            props: Node properties
            domain_config: Domain configuration
        
        Returns:
            List of domain feature values
        """
        features = []
        
        # Check if node matches domain keywords
        node_text = str(node.get("id", "") + " " + str(props)).lower()
        domain_keywords = domain_config.get("keywords", [])
        
        keyword_matches = sum(1 for kw in domain_keywords if kw.lower() in node_text)
        keyword_match_ratio = keyword_matches / len(domain_keywords) if domain_keywords else 0.0
        features.append(keyword_match_ratio)
        
        # Domain tags matching
        domain_tags = domain_config.get("tags", [])
        tag_matches = sum(1 for tag in domain_tags if tag.lower() in node_text)
        tag_match_ratio = tag_matches / len(domain_tags) if domain_tags else 0.0
        features.append(tag_match_ratio)
        
        # Domain layer/team encoding
        layer = domain_config.get("layer", "unknown")
        team = domain_config.get("team", "unknown")
        layer_encoding = [0.0] * 3  # Support up to 3 layers
        team_encoding = [0.0] * 3   # Support up to 3 teams
        
        layer_map = {"data": 0, "application": 1, "business": 2}
        if layer in layer_map:
            layer_encoding[layer_map[layer]] = 1.0
        features.extend(layer_encoding)
        features.extend(team_encoding)
        
        # Pad to 8 features
        while len(features) < 8:
            features.append(0.0)
        
        return features[:8]
    
    def _load_domain_config(self, domain_id: str) -> Optional[Dict[str, Any]]:
        """Load domain configuration from LocalAI.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Domain configuration or None if not found
        """
        if domain_id in self.domain_configs:
            return self.domain_configs[domain_id]
        
        try:
            # Fetch from LocalAI
            response = httpx.get(
                f"{self.localai_url}/v1/domains",
                timeout=5.0
            )
            if response.status_code == 200:
                domains_data = response.json()
                domains = domains_data.get("domains", {})
                
                if domain_id in domains:
                    domain_info = domains[domain_id]
                    config = domain_info.get("config", domain_info)
                    self.domain_configs[domain_id] = config
                    return config
        except Exception as e:
            logger.warning(f"Failed to load domain config for {domain_id}: {e}")
        
        return None
    
    def _create_domain_model(self, domain_id: str) -> Optional[Any]:
        """Create a domain-specific GNN model.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            GNN model or None if PyG not available
        """
        if not HAS_PYG:
            return None
        
        # Use same architecture as base model but domain-specific
        input_dim = 40  # 32 base + 8 domain features
        
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
        
        model = nn.Sequential(*layers).to(self.device)
        logger.info(f"Created domain-specific GNN model for {domain_id}")
        
        return model
    
    def learn_domain_patterns(
        self,
        domain_id: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Learn patterns specific to a domain.
        
        Args:
            domain_id: Domain identifier
            nodes: List of graph nodes
            edges: List of graph edges
        
        Returns:
            Dictionary with learned patterns for the domain
        """
        logger.info(f"Learning domain-specific patterns for domain: {domain_id}")
        
        # Load domain configuration
        domain_config = self._load_domain_config(domain_id)
        
        # Get or create domain-specific model
        if domain_id not in self.domain_models:
            self.domain_models[domain_id] = self._create_domain_model(domain_id)
        
        domain_model = self.domain_models[domain_id]
        if domain_model is None:
            # Fallback to generic learning
            logger.warning(f"Domain model not available for {domain_id}, using generic model")
            return self.learn_patterns(nodes, edges)
        
        # Convert graph with domain context
        data = self.convert_graph_to_pyg_data_with_domain(nodes, edges, domain_config)
        if data is None:
            return {
                "node_embeddings": [],
                "edge_predictions": [],
                "relationship_patterns": {},
                "error": "Failed to convert graph to PyG format",
                "domain_id": domain_id
            }
        
        # Learn patterns with domain model
        try:
            data = data.to(self.device)
            domain_model.eval()
            with torch.no_grad():
                x = data.x
                for i, layer in enumerate(domain_model):
                    x = layer(x, data.edge_index)
                    if i < len(domain_model) - 1:
                        x = F.relu(x)
                        x = F.dropout(x, p=self.dropout, training=False)
                
                node_embeddings = x.cpu().numpy()
            
            # Extract domain-specific patterns
            relationship_patterns = self._extract_relationship_patterns(
                nodes, edges, node_embeddings, domain_config
            )
            
            edge_predictions = self._predict_edge_types(data, node_embeddings)
            
            logger.info(
                f"Domain-specific GNN learning complete for {domain_id}: "
                f"{len(node_embeddings)} node embeddings, "
                f"{len(relationship_patterns)} relationship patterns"
            )
            
            return {
                "node_embeddings": node_embeddings.tolist(),
                "edge_predictions": edge_predictions,
                "relationship_patterns": relationship_patterns,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "embedding_dim": self.hidden_dim,
                "domain_id": domain_id
            }
            
        except Exception as e:
            logger.error(f"Domain-specific GNN learning failed for {domain_id}: {e}", exc_info=True)
            return {
                "node_embeddings": [],
                "edge_predictions": [],
                "relationship_patterns": {},
                "error": str(e),
                "domain_id": domain_id
            }
    
    def convert_graph_to_pyg_data_with_domain(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        domain_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Data]:
        """Convert graph to PyG format with domain context.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            domain_config: Optional domain configuration
        
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
                
                # Create feature vector with domain context
                features = self._extract_node_features(node, props, domain_config)
                node_features.append(features)
            
            # Create edge index (same as before)
            edge_index = []
            edge_attr = []
            
            for edge in edges:
                source_id = edge.get("source_id", edge.get("source", ""))
                target_id = edge.get("target_id", edge.get("target", ""))
                
                if source_id in node_id_to_idx and target_id in node_id_to_idx:
                    source_idx = node_id_to_idx[source_id]
                    target_idx = node_id_to_idx[target_id]
                    edge_index.append([source_idx, target_idx])
                    
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
            
            logger.info(
                f"Converted graph to PyG format with domain context: "
                f"{len(nodes)} nodes, {len(edge_index)} edges"
            )
            return data
            
        except Exception as e:
            logger.error(f"Failed to convert graph to PyG format: {e}")
            return None
    
    def learn_patterns(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Learn relationship patterns using GNN.
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            domain_id: Optional domain identifier for domain-specific learning
        
        Returns:
            Dictionary with learned patterns:
            - node_embeddings: Learned node embeddings
            - edge_predictions: Predicted edge types
            - relationship_patterns: Learned relationship patterns
        """
        # Use domain-specific learning if domain_id provided
        if domain_id:
            return self.learn_domain_patterns(domain_id, nodes, edges)
        
        # Generic learning (no domain specified)
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
        embeddings: np.ndarray,
        domain_config: Optional[Dict[str, Any]] = None
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
                pattern_info = {
                    "count": len(edge_list),
                    "avg_similarity": float(np.mean(similarities)),
                    "std_similarity": float(np.std(similarities)),
                }
                
                # Add domain-specific metadata if available
                if domain_config:
                    pattern_info["domain_id"] = domain_config.get("name", "unknown")
                    pattern_info["domain_layer"] = domain_config.get("layer", "unknown")
                
                patterns[edge_type] = pattern_info
        
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

