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

