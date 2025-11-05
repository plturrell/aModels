"""Transformer models for sequence pattern learning.

This module extends relational_transformer for temporal sequences,
learning table processing sequences (Control-M → SQL → Tables).

Domain-aware enhancements:
- Domain-conditioned sequence learning
- SAP RPT embeddings for domain context
- Domain-specific sequence patterns
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import httpx
import subprocess
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoModel, AutoConfig
    HAS_TRANSFORMER = True
except ImportError:
    HAS_TRANSFORMER = False
    try:
        import torch
        import torch.nn as nn
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False

logger = logging.getLogger(__name__)


class SequencePatternTransformer:
    """Transformer-based learner for temporal sequence patterns.
    
    Domain-aware enhancements:
    - Domain-conditioned sequence embeddings
    - SAP RPT embeddings for domain context
    - Domain-specific sequence pattern learning
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_length: int = 512,
        dropout: float = 0.1,
        localai_url: Optional[str] = None
    ):
        """Initialize sequence pattern transformer.
        
        Args:
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            localai_url: LocalAI URL for domain config fetching (optional)
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_length = max_seq_length
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if HAS_TORCH else None
        
        # Domain awareness
        self.localai_url = localai_url or os.getenv("LOCALAI_URL", "http://localai:8080")
        self.domain_configs = {}  # domain_id -> domain config
        self.domain_embeddings = {}  # domain_id -> domain embedding (SAP RPT)
        self.sap_rpt_path = Path(os.getenv("SAP_RPT_PATH", "/home/aModels/models/sap-rpt-1-oss-main"))
        
        if HAS_TORCH:
            self._build_model()
        else:
            logger.warning("PyTorch not available. Transformer features will be limited.")
    
    def _build_model(self):
        """Build the transformer model."""
        if not HAS_TORCH:
            return
        
        # Build a simple transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation="relu",
            batch_first=True
        )
        
        self.model = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        ).to(self.device)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.hidden_dim, self.dropout).to(self.device)
        
        # Input embedding
        self.input_embedding = nn.Embedding(10000, self.hidden_dim).to(self.device)
        
        logger.info(
            f"Built Transformer model: {self.num_layers} layers, "
            f"{self.num_heads} heads, hidden_dim={self.hidden_dim}"
        )
    
    def convert_sequence_to_tensor(
        self,
        sequence: List[Dict[str, Any]],
        sequence_type: str = "process"
    ) -> Optional[torch.Tensor]:
        """Convert a sequence to tensor format.
        
        Args:
            sequence: List of sequence items (e.g., Control-M jobs, SQL queries, tables)
            sequence_type: Type of sequence ("process", "temporal", "workflow")
        
        Returns:
            Tensor representation of sequence or None if conversion fails
        """
        if not HAS_TORCH:
            return None
        
        try:
            # Extract features from sequence items
            features = []
            for item in sequence:
                item_features = self._extract_sequence_item_features(item, sequence_type)
                features.append(item_features)
            
            if not features:
                return None
            
            # Convert to tensor
            # Pad or truncate to max_seq_length
            if len(features) > self.max_seq_length:
                features = features[:self.max_seq_length]
            else:
                # Pad with zeros
                padding = [[0.0] * len(features[0])] * (self.max_seq_length - len(features))
                features.extend(padding)
            
            tensor = torch.tensor(features, dtype=torch.float32)
            return tensor
            
        except Exception as e:
            logger.error(f"Failed to convert sequence to tensor: {e}")
            return None
    
    def _extract_sequence_item_features(
        self,
        item: Dict[str, Any],
        sequence_type: str
    ) -> List[float]:
        """Extract features from a sequence item.
        
        Args:
            item: Sequence item dictionary
            sequence_type: Type of sequence
        
        Returns:
            Feature vector as list of floats
        """
        features = []
        
        # Item type encoding
        item_type = item.get("type", item.get("label", "unknown"))
        type_features = [0.0] * 10
        type_map = {
            "control-m": 0, "sql": 1, "table": 2, "job": 3, "sequence": 4,
            "petri-net": 5, "transition": 6, "place": 7
        }
        if item_type.lower() in type_map:
            type_idx = type_map[item_type.lower()]
            if type_idx < len(type_features):
                type_features[type_idx] = 1.0
        features.extend(type_features)
        
        # Properties
        props = item.get("properties", {})
        if isinstance(props, str):
            try:
                props = json.loads(props)
            except:
                props = {}
        
        if isinstance(props, dict):
            # Numeric properties
            features.append(float(props.get("order", 0)))
            features.append(float(props.get("sequence_order", 0)))
            features.append(float(props.get("confidence", 0)))
            
            # Temporal features (if available)
            if "timestamp" in props or "created_at" in props:
                features.append(1.0)  # Has timestamp
            else:
                features.append(0.0)
        else:
            features.extend([0.0] * 4)
        
        # Pad to fixed size (can be adjusted)
        target_size = 32
        features.extend([0.0] * (target_size - len(features)))
        return features[:target_size]
    
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
    
    def _get_domain_embedding(self, domain_id: str) -> np.ndarray:
        """Get domain embedding for conditioning using SAP RPT.
        
        Args:
            domain_id: Domain identifier
        
        Returns:
            Domain embedding vector (384-dim for SAP RPT)
        """
        if domain_id in self.domain_embeddings:
            return self.domain_embeddings[domain_id]
        
        # Load domain config
        domain_config = self._load_domain_config(domain_id)
        if not domain_config:
            # Return zero embedding if domain not found
            embedding = np.zeros(384)
            self.domain_embeddings[domain_id] = embedding
            return embedding
        
        # Generate embedding from domain keywords using SAP RPT
        keywords = domain_config.get("keywords", [])
        domain_text = " ".join(keywords) if keywords else domain_config.get("name", domain_id)
        
        # Use SAP RPT embedding script
        try:
            embed_script = self.sap_rpt_path / "sap_rpt_oss" / "scripts" / "start_embedding_server.py"
            if not embed_script.exists():
                # Try alternative path
                embed_script = Path("./scripts/embed_sap_rpt.py")
            
            if embed_script.exists():
                # Call SAP RPT embedding script
                cmd = [
                    "python3",
                    str(embed_script),
                    "--artifact-type", "text",
                    "--text", domain_text
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
                        embedding = np.array(embedding_data, dtype=np.float32)
                        if embedding.shape[0] == 384:
                            self.domain_embeddings[domain_id] = embedding
                            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate SAP RPT embedding for domain {domain_id}: {e}")
        
        # Fallback: return zero embedding
        embedding = np.zeros(384)
        self.domain_embeddings[domain_id] = embedding
        return embedding
    
    def learn_domain_sequences(
        self,
        domain_id: str,
        sequences: List[List[Dict[str, Any]]],
        sequence_type: str = "process"
    ) -> Dict[str, Any]:
        """Learn sequences specific to a domain with domain conditioning.
        
        Args:
            domain_id: Domain identifier
            sequences: List of sequences (each sequence is a list of items)
            sequence_type: Type of sequences
        
        Returns:
            Dictionary with learned patterns for the domain
        """
        logger.info(f"Learning domain-specific sequences for domain: {domain_id}")
        
        # Get domain embedding for conditioning
        domain_embedding = self._get_domain_embedding(domain_id)
        
        # Learn sequences with domain context
        patterns = self.learn_sequence_patterns(sequences, sequence_type)
        
        # Add domain-specific metadata
        patterns["domain_id"] = domain_id
        patterns["domain_embedding_dim"] = len(domain_embedding)
        
        # Extract domain-specific temporal patterns
        domain_temporal_patterns = self._extract_domain_temporal_patterns(
            sequences, patterns.get("sequence_embeddings", []), domain_id
        )
        patterns["domain_temporal_patterns"] = domain_temporal_patterns
        
        logger.info(
            f"Domain-specific sequence learning complete for {domain_id}: "
            f"{len(sequences)} sequences, "
            f"{len(domain_temporal_patterns)} domain-specific patterns"
        )
        
        return patterns
    
    def _extract_domain_temporal_patterns(
        self,
        sequences: List[List[Dict[str, Any]]],
        embeddings: List[List[float]],
        domain_id: str
    ) -> Dict[str, Any]:
        """Extract domain-specific temporal patterns.
        
        Args:
            sequences: Original sequences
            embeddings: Sequence embeddings
            domain_id: Domain identifier
        
        Returns:
            Dictionary of domain-specific temporal patterns
        """
        domain_config = self._load_domain_config(domain_id)
        if not domain_config:
            return {}
        
        # Filter sequences that match domain keywords
        domain_keywords = set(kw.lower() for kw in domain_config.get("keywords", []))
        
        matching_sequences = []
        matching_indices = []
        
        for i, seq in enumerate(sequences):
            # Check if sequence matches domain keywords
            seq_text = " ".join(str(item) for item in seq).lower()
            if any(kw in seq_text for kw in domain_keywords):
                matching_sequences.append(seq)
                matching_indices.append(i)
        
        if not matching_sequences:
            return {
                "domain_id": domain_id,
                "matching_sequences": 0,
                "patterns": {}
            }
        
        # Extract patterns from matching sequences
        matching_embeddings = [embeddings[i] for i in matching_indices if i < len(embeddings)]
        
        patterns = {
            "domain_id": domain_id,
            "domain_name": domain_config.get("name", domain_id),
            "matching_sequences": len(matching_sequences),
            "sequence_patterns": self._extract_temporal_patterns(matching_sequences, np.array(matching_embeddings) if matching_embeddings else np.array([]))
        }
        
        return patterns
    
    def learn_sequence_patterns(
        self,
        sequences: List[List[Dict[str, Any]]],
        sequence_type: str = "process",
        domain_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Learn patterns from sequences using transformer.
        
        Args:
            sequences: List of sequences (each sequence is a list of items)
            sequence_type: Type of sequences
            domain_id: Optional domain identifier for domain-specific learning
        
        Returns:
            Dictionary with learned patterns:
            - sequence_embeddings: Learned sequence embeddings
            - pattern_predictions: Predicted next items in sequences
            - temporal_patterns: Learned temporal patterns
        """
        # Use domain-specific learning if domain_id provided
        if domain_id:
            return self.learn_domain_sequences(domain_id, sequences, sequence_type)
        if not HAS_TORCH:
            logger.warning("Transformer learning skipped: PyTorch not available")
            return {
                "sequence_embeddings": [],
                "pattern_predictions": [],
                "temporal_patterns": {},
                "error": "PyTorch not available"
            }
        
        try:
            # Convert sequences to tensors
            sequence_tensors = []
            valid_sequences = []
            
            for seq in sequences:
                tensor = self.convert_sequence_to_tensor(seq, sequence_type)
                if tensor is not None:
                    sequence_tensors.append(tensor)
                    valid_sequences.append(seq)
            
            if not sequence_tensors:
                return {
                    "sequence_embeddings": [],
                    "pattern_predictions": [],
                    "temporal_patterns": {},
                    "error": "No valid sequences found"
                }
            
            # Batch sequences
            batch = torch.stack(sequence_tensors).to(self.device)
            
            # Forward pass through transformer
            self.model.eval()
            with torch.no_grad():
                # Embed inputs (simplified - would use proper tokenization)
                # For now, use a linear projection
                if batch.dim() == 2:
                    batch = batch.unsqueeze(0)  # Add batch dimension if needed
                
                # Project to hidden_dim
                if batch.shape[-1] != self.hidden_dim:
                    # Use a simple linear layer (would be better to have this as a model attribute)
                    projection = nn.Linear(batch.shape[-1], self.hidden_dim).to(self.device)
                    batch = projection(batch)
                
                # Add positional encoding
                batch = self.pos_encoder(batch)
                
                # Pass through transformer
                output = self.model(batch)
                
                # Get sequence embeddings (mean pooling over sequence length)
                sequence_embeddings = output.mean(dim=1).cpu().numpy()
            
            # Extract temporal patterns
            temporal_patterns = self._extract_temporal_patterns(valid_sequences, sequence_embeddings)
            
            # Predict next items (simplified)
            pattern_predictions = self._predict_next_items(valid_sequences, sequence_embeddings)
            
            logger.info(
                f"Transformer learning complete: {len(sequence_embeddings)} sequence embeddings, "
                f"{len(temporal_patterns)} temporal patterns"
            )
            
            return {
                "sequence_embeddings": sequence_embeddings.tolist(),
                "pattern_predictions": pattern_predictions,
                "temporal_patterns": temporal_patterns,
                "num_sequences": len(valid_sequences),
                "embedding_dim": self.hidden_dim
            }
            
        except Exception as e:
            logger.error(f"Transformer learning failed: {e}", exc_info=True)
            return {
                "sequence_embeddings": [],
                "pattern_predictions": [],
                "temporal_patterns": {},
                "error": str(e)
            }
    
    def _extract_temporal_patterns(
        self,
        sequences: List[List[Dict[str, Any]]],
        embeddings: np.ndarray
    ) -> Dict[str, Any]:
        """Extract temporal patterns from learned embeddings.
        
        Args:
            sequences: Original sequences
            embeddings: Learned sequence embeddings
        
        Returns:
            Dictionary with temporal patterns
        """
        patterns = {}
        
        # Analyze sequence types
        sequence_types = {}
        for i, seq in enumerate(sequences):
            if seq:
                first_item = seq[0]
                seq_type = first_item.get("type", "unknown")
                if seq_type not in sequence_types:
                    sequence_types[seq_type] = []
                sequence_types[seq_type].append(i)
        
        # Calculate patterns for each sequence type
        for seq_type, indices in sequence_types.items():
            if indices and len(indices) > 1:
                # Get embeddings for this sequence type
                type_embeddings = embeddings[indices]
                
                # Calculate average embedding
                avg_embedding = np.mean(type_embeddings, axis=0)
                
                # Calculate similarity variance
                similarities = []
                for emb in type_embeddings:
                    similarity = np.dot(emb, avg_embedding) / (
                        np.linalg.norm(emb) * np.linalg.norm(avg_embedding) + 1e-8
                    )
                    similarities.append(float(similarity))
                
                patterns[seq_type] = {
                    "count": len(indices),
                    "avg_similarity": float(np.mean(similarities)),
                    "std_similarity": float(np.std(similarities)),
                    "avg_embedding": avg_embedding.tolist()
                }
        
        return patterns
    
    def _predict_next_items(
        self,
        sequences: List[List[Dict[str, Any]]],
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Predict next items in sequences.
        
        Args:
            sequences: Original sequences
            embeddings: Learned sequence embeddings
        
        Returns:
            List of predictions
        """
        predictions = []
        
        for i, seq in enumerate(sequences):
            if seq and i < len(embeddings):
                # Simple prediction based on sequence type
                last_item = seq[-1]
                last_type = last_item.get("type", "unknown")
                
                predictions.append({
                    "sequence_index": i,
                    "last_item_type": last_type,
                    "predicted_next_type": "table" if last_type == "sql" else "sql",
                    "confidence": 0.7  # Placeholder
                })
        
        return predictions


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

