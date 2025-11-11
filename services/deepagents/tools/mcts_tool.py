"""MCTS tools for DeepAgents.

Provides tools for narrative path planning and what-if analysis using MCTS.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import json

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Training service URL for MCTS API
TRAINING_SERVICE_URL = os.getenv("TRAINING_SERVICE_URL", "http://training-service:8080")


@tool
def plan_narrative_path_mcts(
    current_state: Dict[str, Any],
    storyline_id: Optional[str] = None,
    num_iterations: int = 100
) -> Dict[str, Any]:
    """Plan a narrative path using Monte Carlo Tree Search (MCTS).
    
    Uses MCTS to explore possible narrative sequences and find the best path
    forward based on causal plausibility and narrative coherence.
    
    Args:
        current_state: Current narrative state (dict with 'time', 'events', etc.)
        storyline_id: Optional storyline ID to focus on
        num_iterations: Number of MCTS iterations (more = better but slower)
        
    Returns:
        Dict with 'action_sequence' (list of actions) and 'best_value' (float)
        
    Example:
        {
            "current_state": {"time": 0.0, "events": []},
            "storyline_id": "merger_story",
            "num_iterations": 50
        }
    """
    try:
        import httpx
        
        url = f"{TRAINING_SERVICE_URL}/gnn/mcts/plan-path"
        payload = {
            "current_state": current_state,
            "storyline_id": storyline_id,
            "num_iterations": num_iterations
        }
        
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        
        result = response.json()
        return {
            "action_sequence": result.get("action_sequence", []),
            "best_value": result.get("best_value", 0.0),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error planning narrative path with MCTS: {e}")
        return {
            "action_sequence": [],
            "best_value": 0.0,
            "status": "error",
            "error": str(e)
        }


@tool
def what_if_analysis_mcts(
    counterfactual_condition: Dict[str, Any],
    storyline_id: Optional[str] = None,
    num_iterations: int = 200,
    use_reflective: bool = True
) -> Dict[str, Any]:
    """Perform what-if analysis using MCTS (optionally with Reflective-MCTS).
    
    Explores counterfactual scenarios by simulating different outcomes
    and comparing them to the original scenario.
    
    Args:
        counterfactual_condition: Counterfactual change (e.g., {"remove_node": "node_id"})
        storyline_id: Optional storyline ID
        num_iterations: Number of MCTS iterations
        use_reflective: Whether to use Reflective-MCTS (multi-agent debate)
        
    Returns:
        Dict with 'original_value', 'counterfactual_value', 'difference', 'analysis'
        
    Example:
        {
            "counterfactual_condition": {"remove_node": "key_player_1"},
            "storyline_id": "merger_story",
            "num_iterations": 100,
            "use_reflective": True
        }
    """
    try:
        import httpx
        
        url = f"{TRAINING_SERVICE_URL}/gnn/mcts/what-if"
        payload = {
            "counterfactual_condition": counterfactual_condition,
            "storyline_id": storyline_id,
            "num_iterations": num_iterations,
            "use_reflective": use_reflective
        }
        
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        
        result = response.json()
        return {
            "original_value": result.get("original_value", 0.0),
            "counterfactual_value": result.get("counterfactual_value", 0.0),
            "difference": result.get("difference", 0.0),
            "analysis": result.get("analysis", ""),
            "recommended_action": result.get("recommended_action"),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error performing what-if analysis with MCTS: {e}")
        return {
            "original_value": 0.0,
            "counterfactual_value": 0.0,
            "difference": 0.0,
            "analysis": f"Error: {str(e)}",
            "status": "error",
            "error": str(e)
        }


@tool
def reflective_mcts_debate(
    state: Dict[str, Any],
    question: str,
    num_agents: int = 3
) -> Dict[str, Any]:
    """Use Reflective-MCTS with multi-agent debate to evaluate a state.
    
    Multiple agents with different perspectives debate the evaluation,
    providing more balanced and robust assessments.
    
    Args:
        state: State to evaluate
        question: Question about the state (e.g., "Is this narrative coherent?")
        num_agents: Number of agents for debate (default 3)
        
    Returns:
        Dict with 'consensus_value', 'agent_evaluations', 'debate_summary'
        
    Example:
        {
            "state": {"time": 0.0, "events": ["event1", "event2"]},
            "question": "Is this narrative path plausible?",
            "num_agents": 5
        }
    """
    try:
        import httpx
        
        url = f"{TRAINING_SERVICE_URL}/gnn/mcts/reflective-debate"
        payload = {
            "state": state,
            "question": question,
            "num_agents": num_agents
        }
        
        response = httpx.post(url, json=payload, timeout=60.0)
        response.raise_for_status()
        
        result = response.json()
        return {
            "consensus_value": result.get("consensus_value", 0.0),
            "agent_evaluations": result.get("agent_evaluations", []),
            "debate_summary": result.get("debate_summary", ""),
            "reflection_insights": result.get("reflection_insights", []),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in reflective MCTS debate: {e}")
        return {
            "consensus_value": 0.0,
            "agent_evaluations": [],
            "debate_summary": f"Error: {str(e)}",
            "status": "error",
            "error": str(e)
        }

