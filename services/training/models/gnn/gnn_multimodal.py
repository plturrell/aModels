"""Multi-Modal Learning for GNN.

This module combines GNN embeddings with semantic embeddings (SAP RPT),
temporal patterns, and domain configurations for enhanced intelligence.
"""

import logging
import os
import subprocess
import json
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class MultiModalFusion(nn.Module):
    """Fusion layer for combining multiple embedding modalities.
    
    Combines:
    - GNN embeddings (graph structure)
    - Semantic embeddings (SAP RPT)
    - Temporal patterns
    - Domain configurations
    """
    
    def __init__(
        self,
        gnn_dim: int = 128,
        semantic_dim: int = 384,  # SAP RPT default
        temporal_dim: int = 64,
        domain_dim: int = 64,
        output_dim: int = 256,
        fusion_method: str = "attention"  # "attention", "concat", "weighted"
    ):
        """Initialize multi-modal fusion layer.
        
        Args:
            gnn_dim: Dimension of GNN embeddings
            semantic_dim: Dimension of semantic embeddings (SAP RPT)
            temporal_dim: Dimension of temporal pattern features
            domain_dim: Dimension of domain configuration features
            output_dim: Output dimension after fusion
            fusion_method: Fusion method ("attention", "concat", "weighted")
        """
        super().__init__()
        
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for MultiModalFusion")
        
        self.gnn_dim = gnn_dim
        self.semantic_dim = semantic_dim
        self.temporal_dim = temporal_dim
        self.domain_dim = domain_dim
        self.output_dim = output_dim
        self.fusion_method = fusion_method
        
        if fusion_method == "attention":
            # Attention-based fusion
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                batch_first=True
            )
            self.gnn_proj = nn.Linear(gnn_dim, output_dim)
            self.semantic_proj = nn.Linear(semantic_dim, output_dim)
            self.temporal_proj = nn.Linear(temporal_dim, output_dim) if temporal_dim > 0 else None
            self.domain_proj = nn.Linear(domain_dim, output_dim) if domain_dim > 0 else None
            self.output_norm = nn.LayerNorm(output_dim)
        elif fusion_method == "concat":
            # Concatenation + projection
            total_dim = gnn_dim + semantic_dim
            if temporal_dim > 0:
                total_dim += temporal_dim
            if domain_dim > 0:
                total_dim += domain_dim
            self.fusion_proj = nn.Linear(total_dim, output_dim)
            self.output_norm = nn.LayerNorm(output_dim)
        elif fusion_method == "weighted":
            # Weighted combination
            self.gnn_proj = nn.Linear(gnn_dim, output_dim)
            self.semantic_proj = nn.Linear(semantic_dim, output_dim)
            self.temporal_proj = nn.Linear(temporal_dim, output_dim) if temporal_dim > 0 else None
            self.domain_proj = nn.Linear(domain_dim, output_dim) if domain_dim > 0 else None
            self.weights = nn.Parameter(torch.ones(4) / 4)  # Learnable weights
            self.output_norm = nn.LayerNorm(output_dim)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(
        self,
        gnn_emb: torch.Tensor,
        semantic_emb: torch.Tensor,
        temporal_emb: Optional[torch.Tensor] = None,
        domain_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Fuse multiple modalities.
        
        Args:
            gnn_emb: GNN embeddings [batch, gnn_dim]
            semantic_emb: Semantic embeddings [batch, semantic_dim]
            temporal_emb: Temporal pattern features [batch, temporal_dim] (optional)
            domain_emb: Domain configuration features [batch, domain_dim] (optional)
        
        Returns:
            Fused embeddings [batch, output_dim]
        """
        if self.fusion_method == "attention":
            # Project all modalities to same dimension
            gnn_proj = self.gnn_proj(gnn_emb)
            semantic_proj = self.semantic_proj(semantic_emb)
            
            # Stack as sequence for attention
            modalities = [gnn_proj.unsqueeze(1), semantic_proj.unsqueeze(1)]
            
            if temporal_emb is not None and self.temporal_proj is not None:
                temporal_proj = self.temporal_proj(temporal_emb)
                modalities.append(temporal_proj.unsqueeze(1))
            
            if domain_emb is not None and self.domain_proj is not None:
                domain_proj = self.domain_proj(domain_emb)
                modalities.append(domain_proj.unsqueeze(1))
            
            # Concatenate modalities
            query_key_value = torch.cat(modalities, dim=1)  # [batch, num_modalities, output_dim]
            
            # Self-attention
            fused, _ = self.attention(query_key_value, query_key_value, query_key_value)
            
            # Average pooling over modalities
            fused = fused.mean(dim=1)  # [batch, output_dim]
            
            return self.output_norm(fused)
        
        elif self.fusion_method == "concat":
            # Concatenate all modalities
            fused = [gnn_emb, semantic_emb]
            if temporal_emb is not None:
                fused.append(temporal_emb)
            if domain_emb is not None:
                fused.append(domain_emb)
            
            fused = torch.cat(fused, dim=-1)
            fused = self.fusion_proj(fused)
            return self.output_norm(fused)
        
        elif self.fusion_method == "weighted":
            # Weighted combination
            gnn_proj = self.gnn_proj(gnn_emb)
            semantic_proj = self.semantic_proj(semantic_emb)
            
            # Normalize weights
            weights = torch.softmax(self.weights, dim=0)
            
            fused = weights[0] * gnn_proj + weights[1] * semantic_proj
            
            if temporal_emb is not None and self.temporal_proj is not None:
                temporal_proj = self.temporal_proj(temporal_emb)
                fused = fused + weights[2] * temporal_proj
            
            if domain_emb is not None and self.domain_proj is not None:
                domain_proj = self.domain_proj(domain_emb)
                fused = fused + weights[3] * domain_proj
            
            return self.output_norm(fused)


class MultiModalGNN:
    """Multi-modal GNN that combines graph structure with semantic and temporal information."""
    
    def __init__(
        self,
        gnn_embedder=None,
        fusion_method: str = "attention",
        output_dim: int = 256,
        device: Optional[str] = None,
        sap_rpt_path: Optional[str] = None
    ):
        """Initialize multi-modal GNN.
        
        Args:
            gnn_embedder: GNN embedder instance
            fusion_method: Fusion method ("attention", "concat", "weighted")
            output_dim: Output embedding dimension
            device: Device to use ("cuda" or "cpu")
            sap_rpt_path: Path to SAP RPT embedding script
        """
        self.gnn_embedder = gnn_embedder
        self.fusion_method = fusion_method
        self.output_dim = output_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sap_rpt_path = sap_rpt_path or os.getenv("SAP_RPT_PATH", "./scripts/embed_sap_rpt.py")
        
        # Initialize fusion layer
        if HAS_TORCH:
            self.fusion_layer = MultiModalFusion(
                gnn_dim=128,  # Default GNN embedding dim
                semantic_dim=384,  # SAP RPT default
                temporal_dim=64,
                domain_dim=64,
                output_dim=output_dim,
                fusion_method=fusion_method
            ).to(self.device)
        else:
            self.fusion_layer = None
            logger.warning("PyTorch not available, multi-modal fusion disabled")
    
    def _get_semantic_embedding(
        self,
        text: str,
        artifact_type: str = "text"
    ) -> Optional[np.ndarray]:
        """Get semantic embedding using SAP RPT.
        
        Args:
            text: Text to embed
            artifact_type: Type of artifact ("text", "table", "column")
        
        Returns:
            Semantic embedding vector or None
        """
        try:
            cmd = [
                "python3",
                self.sap_rpt_path,
                "--artifact-type", artifact_type,
                "--text", text
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10.0
            )
            
            if result.returncode == 0:
                embedding_data = json.loads(result.stdout)
                if isinstance(embedding_data, list):
                    return np.array(embedding_data, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to get semantic embedding: {e}")
        
        return None
    
    def _extract_temporal_features(
        self,
        temporal_patterns: Optional[Dict[str, Any]],
        node_id: str
    ) -> Optional[np.ndarray]:
        """Extract temporal features for a node.
        
        Args:
            temporal_patterns: Temporal pattern data
            node_id: Node ID
        
        Returns:
            Temporal feature vector or None
        """
        if not temporal_patterns:
            return None
        
        try:
            # Extract evolution patterns
            evolution = temporal_patterns.get("evolution_patterns", {})
            temporal_metrics = temporal_patterns.get("temporal_metrics", {})
            
            # Create feature vector
            features = []
            
            # Evolution features
            if node_id in evolution:
                node_evolution = evolution[node_id]
                features.extend([
                    node_evolution.get("change_frequency", 0.0),
                    node_evolution.get("stability_score", 0.0),
                    len(node_evolution.get("change_history", []))
                ])
            else:
                features.extend([0.0, 1.0, 0.0])  # Default: stable, no changes
            
            # Temporal metrics
            if node_id in temporal_metrics:
                node_metrics = temporal_metrics[node_id]
                features.extend([
                    node_metrics.get("avg_change_rate", 0.0),
                    node_metrics.get("volatility", 0.0),
                    node_metrics.get("trend", 0.0)
                ])
            else:
                features.extend([0.0, 0.0, 0.0])
            
            # Pad or truncate to 64 dimensions
            while len(features) < 64:
                features.append(0.0)
            features = features[:64]
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract temporal features: {e}")
            return None
    
    def _extract_domain_features(
        self,
        domain_config: Optional[Dict[str, Any]],
        node: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Extract domain configuration features.
        
        Args:
            domain_config: Domain configuration
            node: Node data
        
        Returns:
            Domain feature vector or None
        """
        if not domain_config:
            return None
        
        try:
            features = []
            
            # Domain keywords
            keywords = domain_config.get("keywords", [])
            features.append(len(keywords))
            
            # Domain rules
            rules = domain_config.get("rules", {})
            features.append(len(rules))
            
            # Node matches domain
            node_text = node.get("label", node.get("id", ""))
            matches = sum(1 for kw in keywords if kw.lower() in node_text.lower())
            features.append(matches)
            
            # Pad to 64 dimensions
            while len(features) < 64:
                features.append(0.0)
            features = features[:64]
            
            return np.array(features, dtype=np.float32)
        except Exception as e:
            logger.warning(f"Failed to extract domain features: {e}")
            return None
    
    def generate_multimodal_embeddings(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        semantic_embeddings: Optional[Dict[str, np.ndarray]] = None,
        temporal_patterns: Optional[Dict[str, Any]] = None,
        domain_configs: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate multi-modal embeddings combining GNN, semantic, temporal, and domain features.
        
        Args:
            nodes: Graph nodes
            edges: Graph edges
            semantic_embeddings: Pre-computed semantic embeddings (optional)
            temporal_patterns: Temporal pattern data (optional)
            domain_configs: Domain configurations (optional)
        
        Returns:
            Dictionary with multi-modal embeddings
        """
        if not HAS_TORCH or self.fusion_layer is None:
            return {"error": "PyTorch or fusion layer not available"}
        
        try:
            # Step 1: Get GNN embeddings
            if self.gnn_embedder:
                gnn_result = self.gnn_embedder.generate_embeddings(
                    nodes, edges, graph_level=False
                )
                if "error" in gnn_result:
                    return {"error": f"GNN embedding failed: {gnn_result['error']}"}
                gnn_embeddings = gnn_result.get("node_embeddings", {})
            else:
                return {"error": "GNN embedder not available"}
            
            # Step 2: Get or compute semantic embeddings
            if semantic_embeddings is None:
                semantic_embeddings = {}
                for node in nodes:
                    node_id = node.get("id", "")
                    if node_id not in semantic_embeddings:
                        # Generate semantic embedding
                        node_text = node.get("label", node.get("id", ""))
                        node_type = node.get("type", "text")
                        artifact_type = "table" if node_type == "table" else "column" if node_type == "column" else "text"
                        
                        sem_emb = self._get_semantic_embedding(node_text, artifact_type)
                        if sem_emb is not None:
                            semantic_embeddings[node_id] = sem_emb
            
            # Step 3: Extract temporal features
            temporal_features = {}
            if temporal_patterns:
                for node in nodes:
                    node_id = node.get("id", "")
                    temp_feat = self._extract_temporal_features(temporal_patterns, node_id)
                    if temp_feat is not None:
                        temporal_features[node_id] = temp_feat
            
            # Step 4: Extract domain features
            domain_features = {}
            if domain_configs:
                for node in nodes:
                    node_id = node.get("id", "")
                    # Find matching domain config
                    for domain_id, domain_config in domain_configs.items():
                        domain_feat = self._extract_domain_features(domain_config, node)
                        if domain_feat is not None:
                            domain_features[node_id] = domain_feat
                            break
            
            # Step 5: Fuse modalities
            multimodal_embeddings = {}
            self.fusion_layer.eval()
            
            with torch.no_grad():
                for node in nodes:
                    node_id = node.get("id", "")
                    
                    # Get GNN embedding
                    if node_id not in gnn_embeddings:
                        continue
                    gnn_emb = torch.tensor(gnn_embeddings[node_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get semantic embedding
                    if node_id not in semantic_embeddings:
                        continue
                    sem_emb = torch.tensor(semantic_embeddings[node_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get temporal features (optional)
                    temp_emb = None
                    if node_id in temporal_features:
                        temp_emb = torch.tensor(temporal_features[node_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Get domain features (optional)
                    dom_emb = None
                    if node_id in domain_features:
                        dom_emb = torch.tensor(domain_features[node_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Fuse
                    fused = self.fusion_layer(gnn_emb, sem_emb, temp_emb, dom_emb)
                    multimodal_embeddings[node_id] = fused.squeeze(0).cpu().numpy().tolist()
            
            return {
                "multimodal_embeddings": multimodal_embeddings,
                "num_nodes": len(multimodal_embeddings),
                "fusion_method": self.fusion_method,
                "output_dim": self.output_dim
            }
            
        except Exception as e:
            logger.error(f"Multi-modal embedding generation failed: {e}")
            return {"error": str(e)}

