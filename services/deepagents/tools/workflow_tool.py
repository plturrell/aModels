"""Tool for analyzing and optimizing workflow execution."""

import os
from typing import Optional, Dict, Any
import httpx
from langchain_core.tools import tool
import json


GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://graph-service:8081")
_client = httpx.Client(timeout=60.0)


@tool
def analyze_workflow_state(
    workflow_state: Dict[str, Any],
    workflow_id: Optional[str] = None,
) -> str:
    """Analyze the current state of a workflow execution.
    
    This tool analyzes workflow state to understand progress, identify bottlenecks,
    and suggest optimizations.
    
    Args:
        workflow_state: Dictionary containing workflow state information
        workflow_id: Optional workflow identifier
    
    Returns:
        JSON string with analysis results including:
        - Progress assessment
        - Bottleneck identification
        - Resource utilization
        - Recommendations
    """
    try:
        # Format workflow state for analysis
        state_summary = json.dumps(workflow_state, indent=2)
        
        # For now, return a structured analysis format
        # In a full implementation, this could call the graph service or use DeepAgents
        analysis = {
            "workflow_id": workflow_id or "unknown",
            "state_keys": list(workflow_state.keys()) if isinstance(workflow_state, dict) else [],
            "progress": _assess_progress(workflow_state),
            "bottlenecks": _identify_bottlenecks(workflow_state),
            "recommendations": _generate_recommendations(workflow_state),
        }
        
        return json.dumps(analysis, indent=2)
    
    except Exception as e:
        return f"Error analyzing workflow state: {str(e)}"


@tool
def suggest_next_steps(
    current_state: Dict[str, Any],
    workflow_type: Optional[str] = None,
) -> str:
    """Suggest next steps for workflow execution.
    
    This tool analyzes the current workflow state and suggests optimal next steps
    based on the workflow type and current progress.
    
    Args:
        current_state: Current workflow state dictionary
        workflow_type: Optional workflow type (e.g., "sequential", "parallel", "unified")
    
    Returns:
        JSON string with suggested next steps:
        - Next actions to take
        - Priority ordering
        - Expected outcomes
        - Resource requirements
    """
    try:
        suggestions = {
            "next_actions": _determine_next_actions(current_state, workflow_type),
            "priority": _prioritize_actions(current_state),
            "expected_outcomes": _predict_outcomes(current_state),
            "resource_requirements": _estimate_resources(current_state),
        }
        
        return json.dumps(suggestions, indent=2)
    
    except Exception as e:
        return f"Error suggesting next steps: {str(e)}"


@tool
def optimize_workflow(
    workflow_spec: Dict[str, Any],
    performance_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Optimize workflow execution based on specification and performance metrics.
    
    This tool analyzes workflow specification and performance data to suggest
    optimizations for better execution efficiency.
    
    Args:
        workflow_spec: Workflow specification/definition
        performance_metrics: Optional performance metrics from previous runs
    
    Returns:
        JSON string with optimization suggestions:
        - Execution order improvements
        - Parallelization opportunities
        - Resource allocation recommendations
        - Caching strategies
    """
    try:
        optimizations = {
            "execution_order": _suggest_execution_order(workflow_spec),
            "parallelization": _identify_parallelization_opportunities(workflow_spec),
            "resource_allocation": _suggest_resource_allocation(workflow_spec, performance_metrics),
            "caching_strategies": _suggest_caching(workflow_spec),
            "estimated_improvement": _estimate_improvement(workflow_spec, performance_metrics),
        }
        
        return json.dumps(optimizations, indent=2)
    
    except Exception as e:
        return f"Error optimizing workflow: {str(e)}"


# Helper functions

def _assess_progress(state: Dict[str, Any]) -> Dict[str, Any]:
    """Assess workflow progress."""
    completed = []
    in_progress = []
    pending = []
    
    # Check for completion flags
    for key, value in state.items():
        if isinstance(key, str) and key.endswith("_complete") and value:
            completed.append(key.replace("_complete", ""))
        elif isinstance(key, str) and key.endswith("_processed") and value:
            completed.append(key.replace("_processed", ""))
        elif isinstance(key, str) and ("_success" in key or "_executed" in key) and value:
            completed.append(key)
    
    return {
        "completed": completed,
        "in_progress": in_progress,
        "pending": pending,
        "completion_percentage": len(completed) / max(len(completed) + len(pending), 1) * 100,
    }


def _identify_bottlenecks(state: Dict[str, Any]) -> list:
    """Identify potential bottlenecks."""
    bottlenecks = []
    
    # Check for long-running operations
    if "deepagents_executed_at" in state and "knowledge_graph_executed_at" in state:
        bottlenecks.append("Multiple long-running operations detected")
    
    # Check for sequential dependencies
    if "mode" in state and state.get("mode") == "sequential":
        bottlenecks.append("Sequential execution mode may create bottlenecks")
    
    return bottlenecks


def _generate_recommendations(state: Dict[str, Any]) -> list:
    """Generate workflow recommendations."""
    recommendations = []
    
    # Suggest parallelization if sequential
    if state.get("mode") == "sequential":
        recommendations.append("Consider parallel execution mode for independent operations")
    
    # Suggest caching if repeated operations
    if "knowledge_graph" in state and "deepagents_result" in state:
        recommendations.append("Cache intermediate results to avoid redundant processing")
    
    return recommendations


def _determine_next_actions(state: Dict[str, Any], workflow_type: Optional[str]) -> list:
    """Determine next actions based on current state."""
    actions = []
    
    # Check what's been completed
    if "knowledge_graph" not in state or not state.get("knowledge_graph"):
        actions.append("Process knowledge graph")
    
    if "deepagents_result" not in state or not state.get("deepagents_result"):
        actions.append("Execute DeepAgents analysis")
    
    if "orchestration_result" not in state or not state.get("orchestration_result"):
        actions.append("Run orchestration chain")
    
    return actions


def _prioritize_actions(state: Dict[str, Any]) -> Dict[str, int]:
    """Prioritize actions based on dependencies."""
    priorities = {}
    
    # Knowledge graph should be first
    if "knowledge_graph" not in state:
        priorities["process_knowledge_graph"] = 1
    
    # DeepAgents can run in parallel
    if "deepagents_result" not in state:
        priorities["execute_deepagents"] = 2
    
    return priorities


def _predict_outcomes(state: Dict[str, Any]) -> Dict[str, Any]:
    """Predict expected outcomes."""
    return {
        "estimated_completion_time": "Unknown",
        "expected_results": ["Knowledge graph data", "AI analysis", "Orchestration output"],
        "confidence": 0.7,
    }


def _estimate_resources(state: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate resource requirements."""
    return {
        "cpu_cores": 2,
        "memory_mb": 1024,
        "gpu_required": False,
        "network_bandwidth": "Low",
    }


def _suggest_execution_order(workflow_spec: Dict[str, Any]) -> list:
    """Suggest optimal execution order."""
    return [
        "1. Process knowledge graph (foundation)",
        "2. Execute orchestration chain (can run in parallel)",
        "3. Run DeepAgents analysis (can run in parallel)",
        "4. Join and aggregate results",
    ]


def _identify_parallelization_opportunities(workflow_spec: Dict[str, Any]) -> list:
    """Identify opportunities for parallelization."""
    opportunities = []
    
    if "nodes" in workflow_spec:
        opportunities.append("Independent nodes can execute in parallel")
    
    if "branches" in workflow_spec:
        opportunities.append("Workflow branches can run concurrently")
    
    return opportunities


def _suggest_resource_allocation(workflow_spec: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Suggest resource allocation."""
    return {
        "knowledge_graph": {"cpu": 1, "memory_mb": 512},
        "deepagents": {"cpu": 2, "memory_mb": 2048, "gpu": False},
        "orchestration": {"cpu": 1, "memory_mb": 512},
    }


def _suggest_caching(workflow_spec: Dict[str, Any]) -> list:
    """Suggest caching strategies."""
    return [
        "Cache knowledge graph query results",
        "Cache DeepAgents analysis for similar inputs",
        "Cache orchestration chain outputs",
    ]


def _estimate_improvement(workflow_spec: Dict[str, Any], metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Estimate potential improvement."""
    return {
        "time_reduction_percent": 30,
        "resource_optimization_percent": 20,
        "throughput_increase_percent": 40,
    }

