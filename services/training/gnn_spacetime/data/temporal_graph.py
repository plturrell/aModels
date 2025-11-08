"""Temporal Graph data structure for managing temporal graph snapshots.

A TemporalGraph manages a collection of temporal nodes and edges,
allowing queries at specific time points and temporal neighborhood sampling.
"""

import logging
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict

try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    Data = Any

from .temporal_node import TemporalNode
from .temporal_edge import TemporalEdge

logger = logging.getLogger(__name__)


class TemporalGraph:
    """A graph that evolves over time with temporal nodes and edges.
    
    Manages temporal graph data and provides utilities for:
    - Extracting static graph snapshots at specific times
    - Temporal neighborhood sampling
    - Adding temporal snapshots
    """
    
    def __init__(
        self,
        nodes: Optional[List[TemporalNode]] = None,
        edges: Optional[List[TemporalEdge]] = None,
        snapshots: Optional[Dict[float, Tuple[List[TemporalNode], List[TemporalEdge]]]] = None
    ):
        """Initialize temporal graph.
        
        Args:
            nodes: List of temporal nodes
            edges: List of temporal edges
            snapshots: Pre-computed snapshots at specific times
        """
        self.nodes = {node.node_id: node for node in (nodes or [])}
        self.edges = edges or []
        self.snapshots = snapshots or {}
        
        logger.info(
            f"Created TemporalGraph with {len(self.nodes)} nodes, "
            f"{len(self.edges)} edges, {len(self.snapshots)} snapshots"
        )
    
    def add_node(self, node: TemporalNode):
        """Add a temporal node to the graph.
        
        Args:
            node: TemporalNode to add
        """
        self.nodes[node.node_id] = node
        logger.debug(f"Added node {node.node_id} to temporal graph")
    
    def add_edge(self, edge: TemporalEdge):
        """Add a temporal edge to the graph.
        
        Args:
            edge: TemporalEdge to add
        """
        self.edges.append(edge)
        logger.debug(
            f"Added edge {edge.source_id}->{edge.target_id} to temporal graph"
        )
    
    def get_graph_at_time(
        self,
        t: float,
        include_inactive: bool = False
    ) -> Tuple[List[TemporalNode], List[TemporalEdge]]:
        """Extract static graph snapshot at time t.
        
        Args:
            t: Time point to query
            include_inactive: If True, include nodes/edges that are not active at t
            
        Returns:
            Tuple of (active_nodes, active_edges) at time t
        """
        # Check if we have a pre-computed snapshot
        if t in self.snapshots:
            return self.snapshots[t]
        
        # Filter nodes active at time t
        active_nodes = [
            node for node in self.nodes.values()
            if include_inactive or node.is_valid_at_time(t)
        ]
        
        # Filter edges active at time t
        active_edges = [
            edge for edge in self.edges
            if include_inactive or edge.is_active_at_time(t)
        ]
        
        logger.debug(
            f"Extracted graph at time {t}: {len(active_nodes)} nodes, "
            f"{len(active_edges)} edges"
        )
        
        return active_nodes, active_edges
    
    def get_temporal_neighbors(
        self,
        node_id: str,
        t: float,
        k: Optional[int] = None,
        include_past: bool = True,
        include_future: bool = False
    ) -> List[Tuple[str, float, float]]:
        """Get temporal neighbors of a node at time t.
        
        Args:
            node_id: Node ID to find neighbors for
            t: Time point
            k: Maximum number of neighbors to return (None = all)
            include_past: Include neighbors from past time points
            include_future: Include neighbors from future time points
            
        Returns:
            List of (neighbor_id, edge_weight, time_delta) tuples
        """
        if node_id not in self.nodes:
            return []
        
        neighbors = []
        
        # Find neighbors through edges
        for edge in self.edges:
            if edge.source_id == node_id:
                neighbor_id = edge.target_id
                if edge.is_active_at_time(t):
                    weight = edge.get_weight_at_time(t)
                    neighbors.append((neighbor_id, weight, 0.0))
            elif edge.target_id == node_id:
                neighbor_id = edge.source_id
                if edge.is_active_at_time(t):
                    weight = edge.get_weight_at_time(t)
                    neighbors.append((neighbor_id, weight, 0.0))
        
        # If including past/future, look at temporal snapshots
        if include_past or include_future:
            # Get node's state history
            node = self.nodes[node_id]
            for state_time, _ in node.state_history:
                if include_past and state_time < t:
                    # Get neighbors at past time
                    past_nodes, past_edges = self.get_graph_at_time(state_time)
                    for edge in past_edges:
                        if edge.source_id == node_id:
                            neighbor_id = edge.target_id
                            weight = edge.get_weight_at_time(state_time)
                            time_delta = t - state_time
                            neighbors.append((neighbor_id, weight, time_delta))
                        elif edge.target_id == node_id:
                            neighbor_id = edge.source_id
                            weight = edge.get_weight_at_time(state_time)
                            time_delta = t - state_time
                            neighbors.append((neighbor_id, weight, time_delta))
                
                if include_future and state_time > t:
                    # Get neighbors at future time
                    future_nodes, future_edges = self.get_graph_at_time(state_time)
                    for edge in future_edges:
                        if edge.source_id == node_id:
                            neighbor_id = edge.target_id
                            weight = edge.get_weight_at_time(state_time)
                            time_delta = state_time - t
                            neighbors.append((neighbor_id, weight, time_delta))
                        elif edge.target_id == node_id:
                            neighbor_id = edge.source_id
                            weight = edge.get_weight_at_time(state_time)
                            time_delta = state_time - t
                            neighbors.append((neighbor_id, weight, time_delta))
        
        # Sort by weight (descending) and limit to k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        if k is not None:
            neighbors = neighbors[:k]
        
        return neighbors
    
    def add_temporal_snapshot(
        self,
        t: float,
        nodes: List[TemporalNode],
        edges: List[TemporalEdge]
    ):
        """Add a pre-computed graph snapshot at time t.
        
        Args:
            t: Time point
            nodes: Nodes active at time t
            edges: Edges active at time t
        """
        self.snapshots[t] = (nodes, edges)
        logger.debug(f"Added snapshot at time {t}")
    
    def to_pyg_data_at_time(
        self,
        t: float,
        include_node_states: bool = True
    ) -> Optional[Data]:
        """Convert temporal graph to PyTorch Geometric Data at time t.
        
        Args:
            t: Time point
            include_node_states: If True, include temporal state in node features
            
        Returns:
            PyG Data object or None if conversion fails
        """
        if not HAS_PYG:
            logger.warning("PyTorch Geometric not available for conversion")
            return None
        
        nodes, edges = self.get_graph_at_time(t)
        
        if not nodes:
            return None
        
        # Build node feature matrix
        node_features = []
        node_id_to_idx = {}
        
        for idx, node in enumerate(nodes):
            node_id_to_idx[node.node_id] = idx
            
            # Start with static features
            if node.static_embedding is not None:
                if isinstance(node.static_embedding, torch.Tensor):
                    features = node.static_embedding.clone()
                else:
                    features = torch.tensor(node.static_embedding, dtype=torch.float)
            else:
                # Use node features if available
                if node.features:
                    # Convert features dict to tensor
                    feature_list = []
                    for key in sorted(node.features.keys()):
                        val = node.features[key]
                        if isinstance(val, (int, float)):
                            feature_list.append(float(val))
                        elif isinstance(val, list):
                            feature_list.extend([float(x) for x in val])
                    features = torch.tensor(feature_list, dtype=torch.float)
                else:
                    # Default: one-hot for node type
                    features = torch.zeros(1, dtype=torch.float)
            
            # Add temporal state if requested
            if include_node_states:
                state = node.get_state_at_time(t)
                if state is not None:
                    if isinstance(state, torch.Tensor):
                        state_tensor = state
                    else:
                        state_tensor = torch.tensor(state, dtype=torch.float)
                    features = torch.cat([features, state_tensor])
            
            node_features.append(features)
        
        # Pad or truncate to same dimension
        max_dim = max(f.shape[0] if len(f.shape) > 0 else 1 for f in node_features)
        node_feature_matrix = []
        for f in node_features:
            if f.shape[0] < max_dim:
                f = torch.cat([f, torch.zeros(max_dim - f.shape[0])])
            elif f.shape[0] > max_dim:
                f = f[:max_dim]
            node_feature_matrix.append(f)
        
        x = torch.stack(node_feature_matrix)
        
        # Build edge index
        edge_index = []
        edge_attr = []
        
        for edge in edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                src_idx = node_id_to_idx[edge.source_id]
                tgt_idx = node_id_to_idx[edge.target_id]
                edge_index.append([src_idx, tgt_idx])
                
                # Edge attributes: weight and relation embedding
                weight = edge.get_weight_at_time(t)
                edge_attrs = [weight]
                
                if edge.relation_embedding is not None:
                    if isinstance(edge.relation_embedding, torch.Tensor):
                        rel_emb = edge.relation_embedding
                    else:
                        rel_emb = torch.tensor(edge.relation_embedding, dtype=torch.float)
                    edge_attrs.extend(rel_emb.tolist() if len(rel_emb.shape) > 0 else [rel_emb.item()])
                
                edge_attr.append(edge_attrs)
        
        if not edge_index:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        if edge_attr:
            # Pad edge attributes to same dimension
            max_edge_dim = max(len(attrs) for attrs in edge_attr)
            edge_attr_padded = []
            for attrs in edge_attr:
                if len(attrs) < max_edge_dim:
                    attrs = attrs + [0.0] * (max_edge_dim - len(attrs))
                elif len(attrs) > max_edge_dim:
                    attrs = attrs[:max_edge_dim]
                edge_attr_padded.append(attrs)
            edge_attr_tensor = torch.tensor(edge_attr_padded, dtype=torch.float)
        else:
            edge_attr_tensor = torch.empty((0, 1), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr_tensor)
    
    def get_time_range(self) -> Tuple[float, float]:
        """Get the time range covered by the graph.
        
        Returns:
            Tuple of (min_time, max_time)
        """
        times = []
        
        # Collect times from node lifespans
        for node in self.nodes.values():
            start, end = node.lifespan
            times.append(start)
            if end is not None:
                times.append(end)
        
        # Collect times from edge scopes
        for edge in self.edges:
            if edge.temporal_scope:
                start, end = edge.temporal_scope
                times.append(start)
                if end is not None:
                    times.append(end)
        
        # Collect times from state history
        for node in self.nodes.values():
            for t, _ in node.state_history:
                times.append(t)
        
        # Collect times from snapshots
        times.extend(self.snapshots.keys())
        
        if not times:
            return (0.0, 0.0)
        
        return (min(times), max(times))

