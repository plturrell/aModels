"""Data utilities for converting between temporal and static graph formats.

Provides functions to:
- Convert existing graph format to temporal format
- Extract temporal features from temporal analysis
- Convert temporal graphs to PyG format
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

try:
    import torch
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    torch = None
    Data = Any

from ..data.temporal_node import TemporalNode
from ..data.temporal_edge import TemporalEdge
from ..data.temporal_graph import TemporalGraph

logger = logging.getLogger(__name__)


def convert_to_temporal_graph(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    temporal_patterns: Optional[Dict[str, Any]] = None,
    default_start_time: float = 0.0
) -> TemporalGraph:
    """Convert standard graph format to temporal graph.
    
    Args:
        nodes: List of node dicts with 'id', 'type', 'properties', etc.
        edges: List of edge dicts with 'source_id', 'target_id', 'label', etc.
        temporal_patterns: Optional temporal pattern data from temporal_analysis
        default_start_time: Default start time for nodes/edges without temporal info
        
    Returns:
        TemporalGraph instance
    """
    temporal_nodes = []
    temporal_edges = []
    
    # Convert nodes
    for node_data in nodes:
        node_id = node_data.get("id") or node_data.get("key", {}).get("id")
        if not node_id:
            logger.warning(f"Skipping node without ID: {node_data}")
            continue
        
        node_type = node_data.get("type") or node_data.get("label", "unknown")
        properties = node_data.get("properties", {})
        
        # Extract static embedding if available
        static_emb = None
        if "embedding" in properties:
            static_emb = properties["embedding"]
        elif "static_embedding" in node_data:
            static_emb = node_data["static_embedding"]
        
        # Extract lifespan from temporal patterns or properties
        lifespan = None
        if temporal_patterns:
            evolution = temporal_patterns.get("evolution_patterns", {})
            if node_id in evolution:
                node_evolution = evolution[node_id]
                # Try to extract creation and deletion times
                change_history = node_evolution.get("change_history", [])
                if change_history:
                    first_change = change_history[0]
                    last_change = change_history[-1]
                    # Use first change as start, last as end (if deleted)
                    start_time = first_change.get("timestamp", default_start_time)
                    end_time = None
                    if node_evolution.get("deleted", False):
                        end_time = last_change.get("timestamp")
                    lifespan = (start_time, end_time)
        
        if lifespan is None:
            # Default: exists from start_time onwards
            lifespan = (default_start_time, None)
        
        # Extract state history from temporal patterns
        state_history = []
        if temporal_patterns:
            evolution = temporal_patterns.get("evolution_patterns", {})
            if node_id in evolution:
                node_evolution = evolution[node_id]
                change_history = node_evolution.get("change_history", [])
                for change in change_history:
                    timestamp = change.get("timestamp", default_start_time)
                    # Create state vector from change data
                    state_vector = _extract_state_from_change(change)
                    if state_vector is not None:
                        state_history.append((timestamp, state_vector))
        
        temporal_node = TemporalNode(
            node_id=node_id,
            node_type=node_type,
            features=properties,
            static_embedding=static_emb,
            lifespan=lifespan,
            state_history=state_history,
            properties=properties
        )
        temporal_nodes.append(temporal_node)
    
    # Convert edges
    for edge_data in edges:
        source_id = edge_data.get("source_id") or edge_data.get("source")
        target_id = edge_data.get("target_id") or edge_data.get("target")
        
        if not source_id or not target_id:
            logger.warning(f"Skipping edge without source/target: {edge_data}")
            continue
        
        relation_type = edge_data.get("label") or edge_data.get("relation_type", "relates_to")
        properties = edge_data.get("properties", {})
        
        # Extract relation embedding if available
        relation_emb = None
        if "embedding" in properties:
            relation_emb = properties["embedding"]
        elif "relation_embedding" in edge_data:
            relation_emb = edge_data["relation_embedding"]
        
        # Extract temporal scope
        temporal_scope = None
        if "temporal_scope" in edge_data:
            temporal_scope = tuple(edge_data["temporal_scope"])
        elif "start_time" in properties or "end_time" in properties:
            start_time = properties.get("start_time", default_start_time)
            end_time = properties.get("end_time")
            temporal_scope = (start_time, end_time)
        
        # Extract weight function info
        weight_function = None
        if "weight_function" in edge_data:
            # Try to reconstruct weight function
            weight_func_info = edge_data["weight_function"]
            if isinstance(weight_func_info, dict):
                func_type = weight_func_info.get("type")
                if func_type == "exponential_decay":
                    decay_rate = weight_func_info.get("decay_rate", 0.1)
                    from ..data.temporal_edge import exponential_decay
                    weight_function = exponential_decay(decay_rate)
                elif func_type == "linear_decay":
                    slope = weight_func_info.get("slope", 0.01)
                    from ..data.temporal_edge import linear_decay
                    weight_function = linear_decay(slope)
        
        base_weight = edge_data.get("weight", properties.get("weight", 1.0))
        
        temporal_edge = TemporalEdge(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            relation_embedding=relation_emb,
            temporal_scope=temporal_scope,
            weight_function=weight_function,
            base_weight=base_weight,
            properties=properties
        )
        temporal_edges.append(temporal_edge)
    
    return TemporalGraph(nodes=temporal_nodes, edges=temporal_edges)


def _extract_state_from_change(change: Dict[str, Any]) -> Optional[Any]:
    """Extract state vector from a change record.
    
    Args:
        change: Change record dict
        
    Returns:
        State vector (torch.Tensor or np.ndarray) or None
    """
    # Try to extract state from change data
    if "state" in change:
        state = change["state"]
        if isinstance(state, (list, tuple)):
            if HAS_PYG:
                return torch.tensor(state, dtype=torch.float)
            else:
                import numpy as np
                return np.array(state, dtype=np.float32)
        elif isinstance(state, torch.Tensor):
            return state
    
    # Create state from change metadata
    state_features = []
    if "change_type" in change:
        # One-hot encode change type
        change_types = ["create", "update", "delete", "modify"]
        change_type_idx = change_types.index(change["change_type"]) if change["change_type"] in change_types else 0
        state_features.extend([1.0 if i == change_type_idx else 0.0 for i in range(len(change_types))])
    
    if "magnitude" in change:
        state_features.append(float(change["magnitude"]))
    
    if state_features:
        if HAS_PYG:
            return torch.tensor(state_features, dtype=torch.float)
        else:
            import numpy as np
            return np.array(state_features, dtype=np.float32)
    
    return None


def extract_temporal_features(
    temporal_graph: TemporalGraph,
    node_id: str,
    time_t: float
) -> Optional[Dict[str, Any]]:
    """Extract temporal features for a node at a specific time.
    
    Args:
        temporal_graph: Temporal graph
        time_t: Time point
        
    Returns:
        Dictionary with temporal features or None
    """
    if node_id not in temporal_graph.nodes:
        return None
    
    node = temporal_graph.nodes[node_id]
    
    features = {
        "node_id": node_id,
        "time": time_t,
        "is_valid": node.is_valid_at_time(time_t),
        "state": node.get_state_at_time(time_t),
        "lifespan": node.lifespan,
        "num_state_history": len(node.state_history),
    }
    
    # Get temporal neighbors
    neighbors = temporal_graph.get_temporal_neighbors(node_id, time_t)
    features["num_neighbors"] = len(neighbors)
    features["neighbor_weights"] = [w for _, w, _ in neighbors]
    
    return features


def temporal_graph_to_pyg_data(
    temporal_graph: TemporalGraph,
    time_t: float,
    include_node_states: bool = True
) -> Optional[Data]:
    """Convert temporal graph to PyG Data at specific time.
    
    Convenience wrapper around TemporalGraph.to_pyg_data_at_time.
    
    Args:
        temporal_graph: Temporal graph
        time_t: Time point
        include_node_states: Include temporal states in features
        
    Returns:
        PyG Data object or None
    """
    return temporal_graph.to_pyg_data_at_time(time_t, include_node_states)

