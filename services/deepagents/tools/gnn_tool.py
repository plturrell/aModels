"""Tools for querying GNN Structural Reasoner service."""

import os
from typing import Optional, List, Dict, Any
import httpx
from langchain_core.tools import tool
import json


TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8080")
_client = httpx.Client(timeout=60.0)  # Longer timeout for GNN processing


@tool
def query_gnn_embeddings(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    graph_level: bool = True,
) -> str:
    """Generate GNN embeddings for nodes or entire graph.
    
    This tool generates graph-level or node-level embeddings using Graph Neural Networks.
    Use embeddings for similarity search, pattern matching, or as features for downstream tasks.
    
    Args:
        nodes: List of graph nodes, each with 'id', 'type', and optionally 'properties'
               Example: [{"id": "table1", "type": "table", "properties": {...}}, ...]
        edges: List of graph edges, each with 'source_id', 'target_id', and optionally 'label'
               Example: [{"source_id": "table1", "target_id": "col1", "label": "HAS_COLUMN"}, ...]
        graph_level: If True, returns graph-level embedding. If False, returns node-level embeddings.
    
    Returns:
        Formatted string with embedding information including:
        - Graph embedding vector (if graph_level=True)
        - Node embeddings dictionary (if graph_level=False)
        - Embedding dimensions and statistics
    
    Example:
        Use this tool when you need to:
        - Find similar tables or columns across systems
        - Generate features for machine learning
        - Perform semantic search on graph structures
        - Compare graph structures
    """
    try:
        endpoint = f"{TRAINING_SERVICE_URL}/gnn/embeddings"
        
        payload = {
            "nodes": nodes,
            "edges": edges,
            "graph_level": graph_level
        }
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") != "success":
            return f"GNN embeddings generation failed: {result.get('detail', 'Unknown error')}"
        
        embeddings = result.get("embeddings", {})
        
        # Format results for readability
        lines = []
        
        if graph_level:
            graph_emb = embeddings.get("graph_embedding", [])
            if graph_emb:
                lines.append(f"Graph-level embedding generated:")
                lines.append(f"  Dimension: {len(graph_emb)}")
                lines.append(f"  Sample values: {graph_emb[:5]}...")
        else:
            node_embs = embeddings.get("node_embeddings", {})
            if node_embs:
                lines.append(f"Node-level embeddings generated:")
                lines.append(f"  Number of nodes: {len(node_embs)}")
                if node_embs:
                    first_node = list(node_embs.keys())[0]
                    first_emb = node_embs[first_node]
                    lines.append(f"  Embedding dimension: {len(first_emb)}")
                    lines.append(f"  Example node '{first_node}': {first_emb[:5]}...")
        
        if not lines:
            return "No embeddings generated. Check that nodes and edges are valid."
        
        return "\n".join(lines)
    
    except httpx.HTTPStatusError as e:
        return f"Error querying GNN embeddings: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying GNN embeddings: {str(e)}"


@tool
def query_gnn_structural_insights(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    insight_type: str = "all",
    threshold: float = 0.5,
) -> str:
    """Get structural insights from graph using GNN.
    
    This tool provides structural insights including anomaly detection, pattern recognition,
    and structural analysis of the knowledge graph.
    
    Args:
        nodes: List of graph nodes
        edges: List of graph edges
        insight_type: Type of insights to retrieve. Options: 'anomalies', 'patterns', 'all'
        threshold: Threshold for anomaly detection (0.0 to 1.0)
    
    Returns:
        Formatted string with structural insights including:
        - Anomalous nodes (if insight_type includes 'anomalies')
        - Pattern information (if insight_type includes 'patterns')
        - Node type classifications
    
    Example:
        Use this tool when you need to:
        - Detect structural anomalies in the graph
        - Identify unusual patterns
        - Understand graph structure characteristics
        - Find potential data quality issues
    """
    try:
        endpoint = f"{TRAINING_SERVICE_URL}/gnn/structural-insights"
        
        payload = {
            "nodes": nodes,
            "edges": edges,
            "insight_type": insight_type,
            "threshold": threshold
        }
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") != "success":
            return f"GNN structural insights failed: {result.get('detail', 'Unknown error')}"
        
        insights = result.get("insights", {})
        
        # Format results
        lines = []
        lines.append("Structural Insights:")
        lines.append("")
        
        if "anomalies" in insights:
            anomalies = insights["anomalies"]
            if "error" not in anomalies:
                anomalous_nodes = anomalies.get("anomalous_nodes", [])
                anomaly_scores = anomalies.get("anomaly_scores", {})
                
                lines.append("Anomalies Detected:")
                if anomalous_nodes:
                    lines.append(f"  Number of anomalous nodes: {len(anomalous_nodes)}")
                    for node_id in anomalous_nodes[:5]:  # Show first 5
                        score = anomaly_scores.get(node_id, "N/A")
                        lines.append(f"  - {node_id}: anomaly_score={score}")
                    if len(anomalous_nodes) > 5:
                        lines.append(f"  ... and {len(anomalous_nodes) - 5} more")
                else:
                    lines.append("  No anomalies detected")
                lines.append("")
            else:
                lines.append(f"  Anomaly detection error: {anomalies.get('error')}")
                lines.append("")
        
        if "patterns" in insights:
            patterns = insights["patterns"]
            if "error" not in patterns:
                lines.append("Pattern Analysis:")
                lines.append(f"  Graph embedding dimension: {patterns.get('graph_embedding_dim', 'N/A')}")
                lines.append(f"  Node embeddings available: {patterns.get('num_node_embeddings', 0)}")
                lines.append("")
            else:
                lines.append(f"  Pattern analysis error: {patterns.get('error')}")
                lines.append("")
        
        if "node_types" in insights:
            node_types = insights["node_types"]
            lines.append("Node Classification:")
            lines.append(f"  Nodes classified: {node_types.get('num_classified', 0)}")
            lines.append(f"  Number of classes: {node_types.get('num_classes', 0)}")
            lines.append("")
        
        if not lines or len(lines) == 1:
            return "No structural insights available. Ensure GNN modules are enabled."
        
        return "\n".join(lines)
    
    except httpx.HTTPStatusError as e:
        return f"Error querying GNN structural insights: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying GNN structural insights: {str(e)}"


@tool
def predict_relationships_gnn(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    candidate_pairs: Optional[List[List[str]]] = None,
    top_k: int = 10,
) -> str:
    """Predict missing relationships or suggest new mappings using GNN.
    
    This tool predicts missing links between nodes or suggests new relationships
    that might exist based on graph structure patterns.
    
    Args:
        nodes: List of graph nodes
        edges: List of existing edges
        candidate_pairs: Optional list of [source_id, target_id] pairs to evaluate.
                        If None, predicts for all possible pairs.
        top_k: Number of top predictions to return
    
    Returns:
        Formatted string with predicted relationships including:
        - Source and target node IDs
        - Probability/confidence scores
        - Confidence level (high/medium/low)
    
    Example:
        Use this tool when you need to:
        - Discover missing relationships in the graph
        - Suggest new mappings between systems
        - Complete incomplete lineage information
        - Find potential connections between entities
    """
    try:
        endpoint = f"{TRAINING_SERVICE_URL}/gnn/predict-links"
        
        payload = {
            "nodes": nodes,
            "edges": edges,
            "top_k": top_k
        }
        
        if candidate_pairs:
            payload["candidate_pairs"] = candidate_pairs
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") != "success":
            return f"GNN link prediction failed: {result.get('detail', 'Unknown error')}"
        
        predictions_data = result.get("predictions", {})
        predictions = predictions_data.get("predictions", [])
        
        if not predictions:
            return "No link predictions generated. The model may need training first."
        
        # Format results
        lines = []
        lines.append(f"Top {len(predictions)} Predicted Relationships:")
        lines.append("")
        
        for i, pred in enumerate(predictions[:top_k], 1):
            source_id = pred.get("source_id", "unknown")
            target_id = pred.get("target_id", "unknown")
            probability = pred.get("probability", 0.0)
            confidence = pred.get("confidence", "medium")
            
            lines.append(f"{i}. {source_id} → {target_id}")
            lines.append(f"   Probability: {probability:.3f} ({confidence} confidence)")
            lines.append("")
        
        return "\n".join(lines)
    
    except httpx.HTTPStatusError as e:
        return f"Error predicting relationships: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error predicting relationships: {str(e)}"


@tool
def classify_nodes_gnn(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> str:
    """Classify nodes by type, domain, or quality using GNN.
    
    This tool classifies graph nodes using Graph Neural Networks to predict
    node types, domains, or quality characteristics.
    
    Args:
        nodes: List of graph nodes to classify
        edges: List of graph edges
        top_k: Optional number of top predictions per node to return
    
    Returns:
        Formatted string with node classifications including:
        - Node ID
        - Predicted class
        - Confidence score
        - Class probabilities (if top_k specified)
    
    Example:
        Use this tool when you need to:
        - Automatically classify nodes by type
        - Identify domain of nodes
        - Predict data quality characteristics
        - Organize schema elements
    """
    try:
        endpoint = f"{TRAINING_SERVICE_URL}/gnn/classify"
        
        payload = {
            "nodes": nodes,
            "edges": edges
        }
        
        if top_k:
            payload["top_k"] = top_k
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") != "success":
            return f"GNN node classification failed: {result.get('detail', 'Unknown error')}"
        
        classifications_data = result.get("classifications", {})
        classifications = classifications_data.get("classifications", [])
        class_mapping = classifications_data.get("class_mapping", {})
        num_classes = classifications_data.get("num_classes", 0)
        
        if not classifications:
            return "No classifications generated. The model may need training first."
        
        # Format results
        lines = []
        lines.append(f"Node Classifications ({num_classes} classes):")
        lines.append("")
        
        for i, cls in enumerate(classifications[:20], 1):  # Limit to 20 for readability
            node_id = cls.get("node_id", "unknown")
            predicted_class = cls.get("predicted_class", "unknown")
            confidence = cls.get("confidence", 0.0)
            probabilities = cls.get("probabilities", [])
            
            lines.append(f"{i}. Node: {node_id}")
            lines.append(f"   Predicted class: {predicted_class}")
            lines.append(f"   Confidence: {confidence:.3f}")
            
            if probabilities and top_k:
                lines.append(f"   Top probabilities:")
                for j, prob in enumerate(probabilities[:top_k]):
                    class_name = class_mapping.get(str(j), f"class_{j}")
                    lines.append(f"     - {class_name}: {prob:.3f}")
            
            lines.append("")
        
        if len(classifications) > 20:
            lines.append(f"... and {len(classifications) - 20} more classifications")
        
        return "\n".join(lines)
    
    except httpx.HTTPStatusError as e:
        return f"Error classifying nodes: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error classifying nodes: {str(e)}"


@tool
def query_domain_gnn(
    domain_id: str,
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    query_type: str = "embeddings",
    query_params: Optional[Dict[str, Any]] = None,
) -> str:
    """Query domain-specific GNN model.
    
    This tool routes queries to domain-specific GNN models for specialized
    analysis within a specific business domain (e.g., finance, supply chain).
    
    Args:
        domain_id: Domain identifier (e.g., 'finance', 'supply_chain')
        nodes: List of graph nodes
        edges: List of graph edges
        query_type: Type of query. Options: 'embeddings', 'classify', 'predict-links', 'insights'
        query_params: Optional additional parameters for the query
    
    Returns:
        Formatted string with domain-specific results
    
    Example:
        Use this tool when you need to:
        - Get domain-specific embeddings
        - Classify nodes within a specific domain context
        - Predict relationships relevant to a domain
        - Get domain-specific structural insights
    """
    try:
        endpoint = f"{TRAINING_SERVICE_URL}/gnn/domains/{domain_id}/query"
        
        payload = {
            "nodes": nodes,
            "edges": edges,
            "query_type": query_type
        }
        
        if query_params:
            payload["query_params"] = query_params
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        if result.get("status") != "success":
            return f"Domain GNN query failed: {result.get('detail', 'Unknown error')}"
        
        query_result = result.get("result", {})
        
        # Format based on query type
        lines = []
        lines.append(f"Domain '{domain_id}' Query Results ({query_type}):")
        lines.append("")
        
        if query_type == "embeddings":
            if "graph_embedding" in query_result:
                graph_emb = query_result["graph_embedding"]
                lines.append(f"Graph embedding dimension: {len(graph_emb)}")
            if "node_embeddings" in query_result:
                node_embs = query_result["node_embeddings"]
                lines.append(f"Node embeddings: {len(node_embs)} nodes")
        
        elif query_type == "classify":
            classifications = query_result.get("classifications", [])
            lines.append(f"Classifications: {len(classifications)} nodes")
            for cls in classifications[:5]:
                lines.append(f"  - {cls.get('node_id')}: {cls.get('predicted_class')}")
        
        elif query_type == "predict-links":
            predictions = query_result.get("predictions", [])
            lines.append(f"Predictions: {len(predictions)} links")
            for pred in predictions[:5]:
                lines.append(f"  - {pred.get('source_id')} → {pred.get('target_id')}: {pred.get('probability', 0.0):.3f}")
        
        elif query_type == "insights":
            insights = query_result.get("insights", {})
            lines.append("Structural insights:")
            if "anomalies" in insights:
                lines.append(f"  Anomalies: {len(insights['anomalies'].get('anomalous_nodes', []))} nodes")
            if "patterns" in insights:
                lines.append(f"  Patterns: Available")
        
        return "\n".join(lines) if lines else str(query_result)
    
    except httpx.HTTPStatusError as e:
        return f"Error querying domain GNN: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error querying domain GNN: {str(e)}"

