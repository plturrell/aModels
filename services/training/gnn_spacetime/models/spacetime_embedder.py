"""Spacetime-aware embedder that wraps existing GNN embedder with temporal capabilities.

Extends gnn_embeddings.GNNEmbedder with temporal state tracking and spacetime message passing.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

# Import from parent directory
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))
try:
    from gnn_embeddings import GNNEmbedder
except ImportError:
    # Fallback: try absolute import
    from services.training.gnn_embeddings import GNNEmbedder
from ..data.temporal_graph import TemporalGraph
from ..core.temporal_models import GRUStateUpdater
from ..core.message_passing import SpacetimeMessagePassing

logger = logging.getLogger(__name__)


class SpacetimeEmbedder:
    """Spacetime-aware embedder that wraps GNNEmbedder with temporal state tracking.
    
    Adds temporal state tracking on top of spatial GNN embeddings.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_sage: bool = True,
        temporal_updater_type: str = "gru",  # "rnn", "lstm", "gru"
        device: Optional[str] = None
    ):
        """Initialize spacetime embedder.
        
        Args:
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            use_sage: Whether to use GraphSAGE
            temporal_updater_type: Type of temporal updater ("rnn", "lstm", "gru")
            device: Device to run on ('cuda' or 'cpu')
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for SpacetimeEmbedder")
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_sage = use_sage
        self.temporal_updater_type = temporal_updater_type
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize base GNN embedder
        self.base_embedder = GNNEmbedder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_sage=use_sage,
            device=str(self.device)
        )
        
        # Initialize temporal updater
        from ..core.temporal_models import RNNStateUpdater, LSTMStateUpdater, GRUStateUpdater
        
        if temporal_updater_type == "rnn":
            self.temporal_updater = RNNStateUpdater(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=1
            ).to(self.device)
        elif temporal_updater_type == "lstm":
            self.temporal_updater = LSTMStateUpdater(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=1
            ).to(self.device)
        else:  # gru (default)
            self.temporal_updater = GRUStateUpdater(
                input_dim=embedding_dim,
                hidden_dim=hidden_dim,
                num_layers=1
            ).to(self.device)
        
        # Initialize spacetime message passing
        self.message_passing = SpacetimeMessagePassing(
            node_dim=embedding_dim,
            message_dim=embedding_dim,
            hidden_dim=hidden_dim,
            temporal_updater=self.temporal_updater,
            aggregation="mean"
        ).to(self.device)
        
        logger.info(
            f"Initialized SpacetimeEmbedder "
            f"(embedding_dim={embedding_dim}, temporal_updater={temporal_updater_type}, "
            f"device={self.device})"
        )
    
    def generate_embeddings(
        self,
        temporal_graph: TemporalGraph,
        time_t: float,
        graph_level: bool = True,
        update_states: bool = True
    ) -> Dict[str, Any]:
        """Generate embeddings for temporal graph at time t.
        
        Args:
            temporal_graph: Temporal graph
            time_t: Time point to query
            graph_level: If True, return graph-level embedding; if False, node embeddings
            update_states: If True, update node states after embedding generation
            
        Returns:
            Dictionary with embeddings:
            - graph_embedding: Graph-level embedding (if graph_level=True)
            - node_embeddings: Dict of node_id -> embedding (if graph_level=False)
        """
        # Get graph snapshot at time t
        nodes, edges = temporal_graph.get_graph_at_time(time_t)
        
        if not nodes:
            return {
                "graph_embedding": None if graph_level else {},
                "node_embeddings": {} if not graph_level else None
            }
        
        # Convert to PyG format
        pyg_data = temporal_graph.to_pyg_data_at_time(time_t, include_node_states=True)
        
        if pyg_data is None:
            logger.warning(f"Failed to convert temporal graph to PyG format at time {time_t}")
            return {
                "graph_embedding": None if graph_level else {},
                "node_embeddings": {} if not graph_level else None
            }
        
        # Move to device
        pyg_data = pyg_data.to(self.device)
        
        # Generate base embeddings using wrapped embedder
        # First, ensure base embedder has a model
        if self.base_embedder.model is None:
            # Initialize model with current graph
            num_node_features = pyg_data.x.shape[1]
            self.base_embedder._initialize_model(num_node_features)
        
        # Get base embeddings
        with torch.no_grad():
            base_embeddings = self.base_embedder.model(pyg_data.x, pyg_data.edge_index)
        
        # Apply temporal state updates if requested
        if update_states:
            # Update node states using spacetime message passing
            updated_embeddings = []
            
            for i, node in enumerate(nodes):
                node_id = node.node_id
                
                # Get base embedding for this node
                base_emb = base_embeddings[i]
                
                # Get previous state
                previous_state = node.get_state_at_time(time_t - 1.0)  # Previous time step
                if previous_state is None:
                    previous_state = node.static_embedding
                if previous_state is not None and not isinstance(previous_state, torch.Tensor):
                    previous_state = torch.tensor(previous_state, dtype=torch.float)
                if previous_state is not None:
                    previous_state = previous_state.to(self.device)
                
                # Get temporal neighbors (past states)
                past_states = [
                    (t, s) for t, s in node.state_history
                    if t < time_t
                ]
                if past_states:
                    # Get most recent past state
                    past_time, past_state = past_states[-1]
                    if not isinstance(past_state, torch.Tensor):
                        past_state = torch.tensor(past_state, dtype=torch.float)
                    past_state = past_state.to(self.device)
                    past_states = [(past_time, past_state)]
                
                # Get spatial neighbors
                neighbors = temporal_graph.get_temporal_neighbors(
                    node_id, time_t, k=10, include_past=False, include_future=False
                )
                
                # Build neighbor list with nodes and edges
                neighbor_list = []
                for neighbor_id, weight, _ in neighbors:
                    if neighbor_id in temporal_graph.nodes:
                        neighbor_node = temporal_graph.nodes[neighbor_id]
                        # Find edge
                        edge = None
                        for e in edges:
                            if (e.source_id == node_id and e.target_id == neighbor_id) or \
                               (e.target_id == node_id and e.source_id == neighbor_id):
                                edge = e
                                break
                        if edge is None:
                            # Create dummy edge
                            from ..data.temporal_edge import TemporalEdge
                            edge = TemporalEdge(node_id, neighbor_id, "relates_to")
                        neighbor_list.append((neighbor_id, neighbor_node, edge, weight))
                
                # Apply spacetime message passing
                if previous_state is not None or past_states:
                    updated_emb = self.message_passing(
                        node, neighbor_list, past_states, time_t, previous_state
                    )
                else:
                    updated_emb = base_emb
                
                updated_embeddings.append(updated_emb)
                
                # Update node state
                if update_states:
                    node.add_state(time_t, updated_emb.detach().cpu())
            
            embeddings = torch.stack(updated_embeddings) if updated_embeddings else base_embeddings
        else:
            embeddings = base_embeddings
        
        # Return appropriate format
        if graph_level:
            # Graph-level embedding: mean pooling
            graph_emb = embeddings.mean(dim=0)
            return {
                "graph_embedding": graph_emb.cpu().numpy().tolist(),
                "node_embeddings": None
            }
        else:
            # Node-level embeddings
            node_embeddings = {
                node.node_id: emb.cpu().numpy().tolist()
                for node, emb in zip(nodes, embeddings)
            }
            return {
                "graph_embedding": None,
                "node_embeddings": node_embeddings
            }
    
    def update_states(
        self,
        temporal_graph: TemporalGraph,
        time_t: float
    ):
        """Update node states in temporal graph at time t.
        
        Args:
            temporal_graph: Temporal graph
            time_t: Time point
        """
        # Generate embeddings with state update
        self.generate_embeddings(
            temporal_graph, time_t, graph_level=False, update_states=True
        )
    
    def train(
        self,
        temporal_graph: TemporalGraph,
        time_points: List[float],
        epochs: int = 10,
        learning_rate: float = 0.001
    ):
        """Train spacetime embedder on temporal graph.
        
        Args:
            temporal_graph: Temporal graph to train on
            time_points: List of time points to train on
            epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # This is a placeholder for training logic
        # In practice, would need loss function and training loop
        logger.info(
            f"Training SpacetimeEmbedder on {len(time_points)} time points "
            f"for {epochs} epochs"
        )
        
        # For now, just update states at each time point
        for epoch in range(epochs):
            for time_t in time_points:
                self.update_states(temporal_graph, time_t)
        
        logger.info("Training complete")

