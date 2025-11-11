"""Tool for running AgentFlow/LangFlow flows."""

import os
from typing import Optional, Dict, Any
import httpx
from langchain_core.tools import tool


AGENTFLOW_SERVICE_URL = os.getenv("AGENTFLOW_SERVICE_URL", "http://agentflow-service:9001")
_client = httpx.Client(timeout=120.0)


@tool
def run_agentflow_flow(
    flow_id: str,
    input_value: Optional[str] = None,
    inputs: Optional[Dict[str, Any]] = None,
    ensure: bool = False,
) -> str:
    """Run an AgentFlow/LangFlow flow and return the results.
    
    This tool allows you to execute pre-configured LangFlow flows that represent
    data pipelines, processing workflows, or agent chains.
    
    Args:
        flow_id: The ID of the flow to run (e.g., "sgmi_pipeline")
        input_value: Optional input value to pass to the flow
        inputs: Optional dictionary of input parameters
        ensure: If True, ensures the flow is synced before running
    
    Returns:
        String containing the flow execution results
    
    Examples:
        - Run a data pipeline flow: flow_id="sgmi_pipeline", inputs={"data": "..."}
        - Run a processing workflow: flow_id="data_quality_check", input_value="table_name"
    """
    try:
        endpoint = f"{AGENTFLOW_SERVICE_URL}/flows/{flow_id}/run"
        
        payload: Dict[str, Any] = {}
        if input_value:
            payload["input_value"] = input_value
        if inputs:
            payload["inputs"] = inputs
        if ensure:
            payload["ensure"] = ensure
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Format result for readability
        if isinstance(result, dict):
            if "result" in result:
                return f"Flow execution completed. Result: {result['result']}"
            elif "output" in result:
                return f"Flow execution completed. Output: {result['output']}"
            else:
                return f"Flow execution completed: {result}"
        
        return str(result)
    
    except httpx.HTTPStatusError as e:
        return f"Error running AgentFlow flow: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error running AgentFlow flow: {str(e)}"


@tool
def optimize_flow(
    flow_id: str,
    performance_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Optimize an AgentFlow/LangFlow flow based on performance metrics.
    
    This tool analyzes flow performance and suggests optimizations for better
    execution efficiency, resource usage, and throughput.
    
    Args:
        flow_id: The ID of the flow to optimize
        performance_metrics: Optional performance metrics from previous runs
    
    Returns:
        JSON string with optimization suggestions:
        - Execution order improvements
        - Node configuration optimizations
        - Resource allocation recommendations
        - Caching strategies
    """
    try:
        # Get flow specification first
        flow_endpoint = f"{AGENTFLOW_SERVICE_URL}/flows/{flow_id}"
        flow_resp = _client.get(flow_endpoint)
        flow_resp.raise_for_status()
        flow_spec = flow_resp.json()
        
        # Build optimization analysis
        analysis = {
            "flow_id": flow_id,
            "optimizations": _analyze_flow_structure(flow_spec),
            "performance_improvements": _suggest_performance_improvements(flow_spec, performance_metrics),
            "resource_optimizations": _suggest_resource_optimizations(flow_spec),
        }
        
        import json
        return json.dumps(analysis, indent=2)
    
    except httpx.HTTPStatusError as e:
        return f"Error optimizing flow: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error optimizing flow: {str(e)}"


@tool
def validate_flow(
    flow_spec: Dict[str, Any],
) -> str:
    """Validate an AgentFlow/LangFlow flow specification.
    
    This tool validates flow structure, node configurations, and connections
    to ensure the flow is properly defined and can execute successfully.
    
    Args:
        flow_spec: Flow specification dictionary
    
    Returns:
        JSON string with validation results:
        - Validation status
        - Issues found
        - Warnings
        - Recommendations
    """
    try:
        validation = {
            "is_valid": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }
        
        # Check for required fields
        if "nodes" not in flow_spec:
            validation["is_valid"] = False
            validation["issues"].append("Missing 'nodes' field in flow specification")
        
        if "edges" not in flow_spec:
            validation["is_valid"] = False
            validation["issues"].append("Missing 'edges' field in flow specification")
        
        # Validate nodes
        if "nodes" in flow_spec:
            for node in flow_spec.get("nodes", []):
                if "id" not in node:
                    validation["warnings"].append("Node missing 'id' field")
                if "data" not in node:
                    validation["warnings"].append(f"Node {node.get('id', 'unknown')} missing 'data' field")
        
        # Validate edges
        if "edges" in flow_spec:
            node_ids = {node.get("id") for node in flow_spec.get("nodes", [])}
            for edge in flow_spec.get("edges", []):
                source = edge.get("source")
                target = edge.get("target")
                if source and source not in node_ids:
                    validation["warnings"].append(f"Edge references unknown source node: {source}")
                if target and target not in node_ids:
                    validation["warnings"].append(f"Edge references unknown target node: {target}")
        
        # Generate recommendations
        if len(flow_spec.get("nodes", [])) > 20:
            validation["recommendations"].append("Consider breaking large flow into smaller sub-flows")
        
        import json
        return json.dumps(validation, indent=2)
    
    except Exception as e:
        return f"Error validating flow: {str(e)}"


@tool
def compare_flows(
    flow_id_1: str,
    flow_id_2: str,
) -> str:
    """Compare two AgentFlow/LangFlow flows.
    
    This tool compares two flows to identify differences, similarities,
    and potential merge opportunities.
    
    Args:
        flow_id_1: First flow ID to compare
        flow_id_2: Second flow ID to compare
    
    Returns:
        JSON string with comparison results:
        - Structural differences
        - Common nodes/edges
        - Merge suggestions
        - Compatibility assessment
    """
    try:
        # Get both flow specifications
        flow1_endpoint = f"{AGENTFLOW_SERVICE_URL}/flows/{flow_id_1}"
        flow2_endpoint = f"{AGENTFLOW_SERVICE_URL}/flows/{flow_id_2}"
        
        flow1_resp = _client.get(flow1_endpoint)
        flow1_resp.raise_for_status()
        flow1_spec = flow1_resp.json()
        
        flow2_resp = _client.get(flow2_endpoint)
        flow2_resp.raise_for_status()
        flow2_spec = flow2_resp.json()
        
        # Compare flows
        comparison = {
            "flow_1": flow_id_1,
            "flow_2": flow_id_2,
            "node_differences": _compare_nodes(flow1_spec, flow2_spec),
            "edge_differences": _compare_edges(flow1_spec, flow2_spec),
            "common_elements": _find_common_elements(flow1_spec, flow2_spec),
            "merge_suggestions": _suggest_merge(flow1_spec, flow2_spec),
        }
        
        import json
        return json.dumps(comparison, indent=2)
    
    except httpx.HTTPStatusError as e:
        return f"Error comparing flows: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error comparing flows: {str(e)}"


# Helper functions

def _analyze_flow_structure(flow_spec: Dict[str, Any]) -> list:
    """Analyze flow structure for optimization opportunities."""
    optimizations = []
    
    nodes = flow_spec.get("nodes", [])
    edges = flow_spec.get("edges", [])
    
    # Check for sequential chains that could be parallelized
    if len(edges) < len(nodes) - 1:
        optimizations.append("Consider parallelizing independent nodes")
    
    # Check for redundant nodes
    node_types = {}
    for node in nodes:
        node_type = node.get("data", {}).get("type", "unknown")
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    for node_type, count in node_types.items():
        if count > 5:
            optimizations.append(f"Multiple {node_type} nodes detected - consider consolidation")
    
    return optimizations


def _suggest_performance_improvements(flow_spec: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> list:
    """Suggest performance improvements."""
    improvements = []
    
    if metrics:
        if metrics.get("execution_time", 0) > 60:
            improvements.append("Long execution time - consider caching intermediate results")
        if metrics.get("memory_usage", 0) > 1024:
            improvements.append("High memory usage - optimize node configurations")
    
    return improvements


def _suggest_resource_optimizations(flow_spec: Dict[str, Any]) -> list:
    """Suggest resource optimizations."""
    return [
        "Use connection pooling for database nodes",
        "Enable result caching for expensive operations",
        "Consider batch processing for large datasets",
    ]


def _compare_nodes(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare nodes between two flows."""
    nodes1 = {node.get("id"): node for node in flow1.get("nodes", [])}
    nodes2 = {node.get("id"): node for node in flow2.get("nodes", [])}
    
    only_in_1 = set(nodes1.keys()) - set(nodes2.keys())
    only_in_2 = set(nodes2.keys()) - set(nodes1.keys())
    common = set(nodes1.keys()) & set(nodes2.keys())
    
    return {
        "only_in_flow_1": list(only_in_1),
        "only_in_flow_2": list(only_in_2),
        "common_nodes": list(common),
    }


def _compare_edges(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> Dict[str, Any]:
    """Compare edges between two flows."""
    edges1 = {(e.get("source"), e.get("target")) for e in flow1.get("edges", [])}
    edges2 = {(e.get("source"), e.get("target")) for e in flow2.get("edges", [])}
    
    only_in_1 = edges1 - edges2
    only_in_2 = edges2 - edges1
    common = edges1 & edges2
    
    return {
        "only_in_flow_1": [{"source": s, "target": t} for s, t in only_in_1],
        "only_in_flow_2": [{"source": s, "target": t} for s, t in only_in_2],
        "common_edges": [{"source": s, "target": t} for s, t in common],
    }


def _find_common_elements(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> Dict[str, Any]:
    """Find common elements between flows."""
    return {
        "common_node_types": _get_common_node_types(flow1, flow2),
        "similarity_score": _calculate_similarity(flow1, flow2),
    }


def _get_common_node_types(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> list:
    """Get common node types."""
    types1 = {node.get("data", {}).get("type") for node in flow1.get("nodes", [])}
    types2 = {node.get("data", {}).get("type") for node in flow2.get("nodes", [])}
    return list(types1 & types2)


def _calculate_similarity(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> float:
    """Calculate similarity score between flows."""
    nodes1 = len(flow1.get("nodes", []))
    nodes2 = len(flow2.get("nodes", []))
    
    if nodes1 == 0 or nodes2 == 0:
        return 0.0
    
    common_types = len(_get_common_node_types(flow1, flow2))
    max_types = max(len({n.get("data", {}).get("type") for n in flow1.get("nodes", [])}),
                    len({n.get("data", {}).get("type") for n in flow2.get("nodes", [])}))
    
    if max_types == 0:
        return 0.0
    
    return common_types / max_types


def _suggest_merge(flow1: Dict[str, Any], flow2: Dict[str, Any]) -> list:
    """Suggest merge opportunities."""
    suggestions = []
    
    similarity = _calculate_similarity(flow1, flow2)
    if similarity > 0.5:
        suggestions.append("High similarity detected - flows may be candidates for merging")
    
    return suggestions

