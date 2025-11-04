"""Tool for running orchestration chains."""

import os
from typing import Optional, Dict, Any
import httpx
from langchain_core.tools import tool


GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://graph-service:8081")
_client = httpx.Client(timeout=60.0)


@tool
def run_orchestration_chain(
    chain_name: str,
    inputs: Dict[str, Any],
    knowledge_graph_query: Optional[str] = None,
) -> str:
    """Run an orchestration chain (LangChain-like) and return the results.
    
    This tool allows you to execute orchestration chains for:
    - Question answering with context
    - Summarization
    - Knowledge graph analysis
    - Data quality analysis
    - Pipeline analysis
    - SQL analysis
    
    Args:
        chain_name: Name of the chain to run. Options:
            - "llm_chain": Basic LLM chain
            - "question_answering": Q&A with context
            - "summarization": Text summarization
            - "knowledge_graph_analyzer": Analyze knowledge graphs
            - "data_quality_analyzer": Analyze data quality metrics
            - "pipeline_analyzer": Analyze data pipelines
            - "sql_analyzer": Analyze SQL queries
        inputs: Dictionary of input parameters for the chain
        knowledge_graph_query: Optional Cypher query to enrich chain context
    
    Returns:
        String containing the chain execution results
    
    Examples:
        - Analyze knowledge graph: chain_name="knowledge_graph_analyzer",
          inputs={"query": "What are the main tables?"}
        - Analyze data quality: chain_name="data_quality_analyzer",
          inputs={"quality_score": 0.75, "query": "Assess quality"}
    """
    try:
        endpoint = f"{GRAPH_SERVICE_URL}/orchestration/process"
        
        payload: Dict[str, Any] = {
            "orchestration_request": {
                "chain_name": chain_name,
                "inputs": inputs,
            }
        }
        
        if knowledge_graph_query:
            payload["knowledge_graph_query"] = knowledge_graph_query
        
        response = _client.post(endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        
        # Extract text output if available
        if isinstance(result, dict):
            if "orchestration_text" in result:
                return result["orchestration_text"]
            elif "orchestration_result" in result:
                orch_result = result["orchestration_result"]
                if isinstance(orch_result, dict) and "text" in orch_result:
                    return orch_result["text"]
                elif isinstance(orch_result, dict) and "output" in orch_result:
                    return orch_result["output"]
                else:
                    return str(orch_result)
            else:
                return f"Chain execution completed: {result}"
        
        return str(result)
    
    except httpx.HTTPStatusError as e:
        return f"Error running orchestration chain: HTTP {e.response.status_code} - {e.response.text}"
    except Exception as e:
        return f"Error running orchestration chain: {str(e)}"

