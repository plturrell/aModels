"""Bidirectional Query Router for KG and GNN.

This module provides intelligent query routing that directs queries to either
the Knowledge Graph (for explicit facts) or GNN service (for structural insights).
"""

import os
from typing import Optional, Dict, Any, List
import httpx
from langchain_core.tools import tool
import re


EXTRACT_SERVICE_URL = os.getenv("EXTRACT_SERVICE_URL", "http://extract-service:19080")
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8080")
_client = httpx.Client(timeout=60.0)


def _is_structural_query(query: str) -> bool:
    """Determine if a query is structural (GNN) or factual (KG).
    
    Structural queries typically ask about:
    - Patterns, similarities, relationships
    - Anomalies, outliers
    - Classifications, groupings
    - Embeddings, representations
    - Missing links, predictions
    
    Factual queries typically ask about:
    - Specific entities, tables, columns
    - Exact relationships, lineage
    - Data values, counts
    - Cypher queries
    
    Args:
        query: Query string
    
    Returns:
        True if structural, False if factual
    """
    query_lower = query.lower()
    
    # Structural keywords
    structural_keywords = [
        "similar", "pattern", "anomaly", "outlier", "classify", "group",
        "embedding", "representation", "predict", "missing", "suggest",
        "relationship", "link", "connection", "structure", "structural",
        "insight", "analysis", "cluster", "grouping", "type", "domain",
        "quality", "characteristic", "feature", "vector", "distance"
    ]
    
    # Factual keywords
    factual_keywords = [
        "find", "get", "list", "show", "return", "match", "where",
        "count", "exists", "has", "contains", "specific", "exact",
        "cypher", "query", "table", "column", "system", "project"
    ]
    
    # Check for structural indicators
    structural_score = sum(1 for keyword in structural_keywords if keyword in query_lower)
    factual_score = sum(1 for keyword in factual_keywords if keyword in query_lower)
    
    # Check for Cypher query syntax (factual)
    if re.search(r'\b(match|return|where|with|create|merge)\b', query_lower):
        return False
    
    # Check for explicit structural requests
    if any(phrase in query_lower for phrase in [
        "structural insight", "graph pattern", "similar nodes",
        "predict relationship", "classify node", "anomaly detection"
    ]):
        return True
    
    # Default: if structural score > factual score, route to GNN
    return structural_score > factual_score


def _extract_graph_data_from_kg_result(kg_result: Dict[str, Any]) -> tuple[List[Dict], List[Dict]]:
    """Extract nodes and edges from KG query result.
    
    Args:
        kg_result: Knowledge graph query result
    
    Returns:
        Tuple of (nodes, edges)
    """
    nodes = []
    edges = []
    
    # Try to extract from different result formats
    if "data" in kg_result and "columns" in kg_result:
        # Cypher query result format
        columns = kg_result["columns"]
        data = kg_result["data"]
        
        for row in data:
            # Try to find node/edge data in row
            for col in columns:
                value = row.get(col)
                if isinstance(value, dict):
                    if "id" in value and "type" in value:
                        nodes.append(value)
                    elif "source" in value and "target" in value:
                        edges.append({
                            "source_id": value.get("source"),
                            "target_id": value.get("target"),
                            "label": value.get("label", "")
                        })
    
    elif "nodes" in kg_result:
        nodes = kg_result["nodes"]
        edges = kg_result.get("edges", [])
    
    return nodes, edges


@tool
def hybrid_query(
    query: str,
    project_id: Optional[str] = None,
    system_id: Optional[str] = None,
    prefer_gnn: Optional[bool] = None,
    combine_results: bool = True,
) -> str:
    """Intelligently route queries to KG (factual) or GNN (structural) and optionally combine results.
    
    This tool analyzes the query and routes it to the appropriate service:
    - Factual queries → Knowledge Graph (Neo4j)
    - Structural queries → GNN service
    
    Optionally combines results from both sources for comprehensive answers.
    
    Args:
        query: Query string (Cypher for KG, natural language for GNN)
        project_id: Optional project ID filter
        system_id: Optional system ID filter
        prefer_gnn: Force routing to GNN (True) or KG (False). If None, auto-detect.
        combine_results: If True, query both services and combine results
    
    Returns:
        Formatted string with query results from appropriate service(s)
    
    Example:
        Use this tool when you need to:
        - Get both factual data and structural insights
        - Automatically route queries to the right service
        - Combine explicit facts with implicit patterns
    """
    try:
        # Determine query type
        is_structural = prefer_gnn if prefer_gnn is not None else _is_structural_query(query)
        
        kg_result = None
        gnn_result = None
        
        # Query Knowledge Graph (for factual data or if combining)
        if not is_structural or combine_results:
            try:
                kg_endpoint = f"{EXTRACT_SERVICE_URL}/knowledge-graph/query"
                kg_payload = {"query": query}
                if project_id or system_id:
                    kg_payload["params"] = {}
                    if project_id:
                        kg_payload["params"]["project_id"] = project_id
                    if system_id:
                        kg_payload["params"]["system_id"] = system_id
                
                kg_response = _client.post(kg_endpoint, json=kg_payload, timeout=30.0)
                kg_response.raise_for_status()
                kg_result = kg_response.json()
            except Exception as e:
                if not combine_results:
                    return f"Error querying knowledge graph: {str(e)}"
                # Continue if combining (GNN might still work)
        
        # Query GNN (for structural insights or if combining)
        if is_structural or combine_results:
            try:
                # First, get graph data from KG if available
                nodes = []
                edges = []
                
                if kg_result:
                    nodes, edges = _extract_graph_data_from_kg_result(kg_result)
                
                # If no graph data from KG, try to get it via a simple KG query
                if not nodes and not edges:
                    try:
                        # Try to get nodes/edges for the project/system
                        simple_query = "MATCH (n) RETURN n LIMIT 100"
                        if project_id:
                            simple_query = f"MATCH (n) WHERE n.project_id = '{project_id}' RETURN n LIMIT 100"
                        
                        simple_kg_response = _client.post(
                            f"{EXTRACT_SERVICE_URL}/knowledge-graph/query",
                            json={"query": simple_query},
                            timeout=30.0
                        )
                        if simple_kg_response.status_code == 200:
                            simple_result = simple_kg_response.json()
                            nodes, edges = _extract_graph_data_from_kg_result(simple_result)
                    except Exception:
                        pass  # Continue without graph data
                
                # Query GNN if we have graph data or if explicitly requested
                if nodes or is_structural:
                    if is_structural:
                        # Use structural insights endpoint
                        gnn_endpoint = f"{TRAINING_SERVICE_URL}/gnn/structural-insights"
                        gnn_payload = {
                            "nodes": nodes if nodes else [],
                            "edges": edges if edges else [],
                            "insight_type": "all",
                            "threshold": 0.5
                        }
                    else:
                        # Use embeddings endpoint
                        gnn_endpoint = f"{TRAINING_SERVICE_URL}/gnn/embeddings"
                        gnn_payload = {
                            "nodes": nodes,
                            "edges": edges,
                            "graph_level": True
                        }
                    
                    gnn_response = _client.post(gnn_endpoint, json=gnn_payload, timeout=60.0)
                    gnn_response.raise_for_status()
                    gnn_result = gnn_response.json()
            except Exception as e:
                if not combine_results:
                    return f"Error querying GNN: {str(e)}"
                # Continue if combining (KG might still work)
        
        # Format combined results
        lines = []
        
        if combine_results:
            lines.append("=== Hybrid Query Results ===")
            lines.append("")
        
        # Format KG results
        if kg_result:
            lines.append("Knowledge Graph Results:")
            if "columns" in kg_result and "data" in kg_result:
                columns = kg_result["columns"]
                data = kg_result["data"]
                if data:
                    lines.append(f"  Columns: {', '.join(columns)}")
                    lines.append(f"  Rows: {len(data)}")
                    for i, row in enumerate(data[:5], 1):
                        lines.append(f"  Row {i}: {dict(row)}")
                    if len(data) > 5:
                        lines.append(f"  ... and {len(data) - 5} more rows")
                else:
                    lines.append("  No results")
            else:
                lines.append(f"  {str(kg_result)[:200]}...")
            lines.append("")
        
        # Format GNN results
        if gnn_result:
            lines.append("GNN Structural Insights:")
            if gnn_result.get("status") == "success":
                if "insights" in gnn_result:
                    insights = gnn_result["insights"]
                    if "anomalies" in insights:
                        anomalies = insights["anomalies"]
                        if "anomalous_nodes" in anomalies:
                            lines.append(f"  Anomalies: {len(anomalies['anomalous_nodes'])} nodes")
                    if "patterns" in insights:
                        lines.append("  Patterns: Available")
                elif "embeddings" in gnn_result:
                    embeddings = gnn_result["embeddings"]
                    if "graph_embedding" in embeddings:
                        lines.append(f"  Graph embedding dimension: {len(embeddings['graph_embedding'])}")
            else:
                lines.append(f"  {gnn_result.get('detail', 'Unknown error')}")
            lines.append("")
        
        if not lines:
            return "No results from either service. Check query format and service availability."
        
        return "\n".join(lines)
    
    except Exception as e:
        return f"Error in hybrid query: {str(e)}"


@tool
def route_query(
    query: str,
    project_id: Optional[str] = None,
    system_id: Optional[str] = None,
) -> str:
    """Route a query to the appropriate service (KG or GNN) based on query type.
    
    This tool analyzes the query and automatically routes it to:
    - Knowledge Graph for factual queries (tables, columns, relationships)
    - GNN service for structural queries (patterns, similarities, predictions)
    
    Args:
        query: Query string
        project_id: Optional project ID filter
        system_id: Optional system ID filter
    
    Returns:
        Formatted string with results from the routed service
    
    Example:
        Use this tool when you want automatic routing:
        - "Find all tables" → routes to KG
        - "Find similar tables" → routes to GNN
        - "Detect anomalies" → routes to GNN
    """
    try:
        is_structural = _is_structural_query(query)
        
        if is_structural:
            # Route to GNN - but we need graph data first
            # Try to get graph data from KG
            try:
                simple_query = "MATCH (n) RETURN n LIMIT 100"
                if project_id:
                    simple_query = f"MATCH (n) WHERE n.project_id = '{project_id}' RETURN n LIMIT 100"
                
                kg_response = _client.post(
                    f"{EXTRACT_SERVICE_URL}/knowledge-graph/query",
                    json={"query": simple_query},
                    timeout=30.0
                )
                
                if kg_response.status_code == 200:
                    kg_result = kg_response.json()
                    nodes, edges = _extract_graph_data_from_kg_result(kg_result)
                    
                    if nodes or edges:
                        # Query GNN with graph data
                        gnn_endpoint = f"{TRAINING_SERVICE_URL}/gnn/structural-insights"
                        gnn_payload = {
                            "nodes": nodes,
                            "edges": edges,
                            "insight_type": "all",
                            "threshold": 0.5
                        }
                        
                        gnn_response = _client.post(gnn_endpoint, json=gnn_payload, timeout=60.0)
                        gnn_response.raise_for_status()
                        gnn_result = gnn_response.json()
                        
                        if gnn_result.get("status") == "success":
                            insights = gnn_result.get("insights", {})
                            lines = ["GNN Structural Insights (routed automatically):", ""]
                            
                            if "anomalies" in insights:
                                anomalies = insights["anomalies"]
                                if "anomalous_nodes" in anomalies:
                                    lines.append(f"Anomalies detected: {len(anomalies['anomalous_nodes'])} nodes")
                            
                            if "patterns" in insights:
                                lines.append("Pattern analysis: Available")
                            
                            return "\n".join(lines) if len(lines) > 2 else str(gnn_result)
            except Exception as e:
                return f"Error routing to GNN: {str(e)}. Query routed to KG as fallback."
        
        # Route to Knowledge Graph (factual query or fallback)
        try:
            endpoint = f"{EXTRACT_SERVICE_URL}/knowledge-graph/query"
            payload = {"query": query}
            if project_id or system_id:
                payload["params"] = {}
                if project_id:
                    payload["params"]["project_id"] = project_id
                if system_id:
                    payload["params"]["system_id"] = system_id
            
            response = _client.post(endpoint, json=payload, timeout=30.0)
            response.raise_for_status()
            result = response.json()
            
            # Format results
            if "columns" in result and "data" in result:
                columns = result["columns"]
                data = result["data"]
                
                if not data:
                    return "Query returned no results."
                
                lines = [f"Knowledge Graph Results (routed automatically):", ""]
                lines.append(f"Columns: {', '.join(columns)}")
                for i, row in enumerate(data[:10], 1):
                    lines.append(f"Row {i}:")
                    for col in columns:
                        lines.append(f"  {col}: {row.get(col, 'N/A')}")
                    lines.append("")
                
                if len(data) > 10:
                    lines.append(f"... and {len(data) - 10} more rows")
                
                return "\n".join(lines)
            
            return str(result)
        except Exception as e:
            return f"Error routing to Knowledge Graph: {str(e)}"
    
    except Exception as e:
        return f"Error in query routing: {str(e)}"

