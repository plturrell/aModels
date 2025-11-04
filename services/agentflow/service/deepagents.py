"""DeepAgents integration for AgentFlow service."""

import os
from typing import Any, Dict, Optional
import httpx
from ..config import get_settings

# HTTP client for DeepAgents requests
_deepagents_client: Optional[httpx.AsyncClient] = None


def get_deepagents_client() -> Optional[httpx.AsyncClient]:
    """Get or create DeepAgents HTTP client.
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    DeepAgents is enabled by default (10/10 integration).
    Set DEEPAGENTS_ENABLED=false to disable.
    """
    global _deepagents_client
<<<<<<< HEAD

    # Enabled by default (10/10 integration)
    # Only disable if explicitly set to false
    enabled = os.getenv("DEEPAGENTS_ENABLED", "").lower() != "false"

    if not enabled:
        return None

=======
    
    # Enabled by default (10/10 integration)
    # Only disable if explicitly set to false
    enabled = os.getenv("DEEPAGENTS_ENABLED", "").lower() != "false"
    
    if not enabled:
        return None
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    if _deepagents_client is None:
        base_url = os.getenv("DEEPAGENTS_URL", "http://deepagents-service:9004")
        _deepagents_client = httpx.AsyncClient(
            base_url=base_url,
            timeout=120.0,  # Deep agents can take longer
        )
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"DeepAgents integration enabled (URL: {base_url})")
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    return _deepagents_client


async def analyze_flow_execution(
    flow_id: str,
    flow_result: Any,
    input_value: Optional[str] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Analyze flow execution using DeepAgents.
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    Args:
        flow_id: The flow ID that was executed
        flow_result: The result from flow execution
        input_value: Optional input value that was used
        inputs: Optional input dictionary that was used
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    Returns:
        Analysis result from DeepAgents, or None if disabled or failed (non-fatal)
    """
    client = get_deepagents_client()
    if client is None:
        return None
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    # Quick health check before attempting analysis
    try:
        health_response = await client.get("/healthz", timeout=5.0)
        if health_response.status_code != 200:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("DeepAgents service unavailable, skipping analysis")
            return None
    except Exception:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("DeepAgents health check failed, skipping analysis")
        return None
<<<<<<< HEAD

    max_retries = 2
    last_exception = None

=======
    
    max_retries = 2
    last_exception = None
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Exponential backoff
            import asyncio
            backoff = attempt * 1.0
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Retrying DeepAgents analysis (attempt {attempt + 1}/{max_retries + 1}) after {backoff}s")
            await asyncio.sleep(backoff)
<<<<<<< HEAD

=======
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        try:
            # Build analysis prompt
            prompt = f"""Analyze this AgentFlow/LangFlow execution:

Flow ID: {flow_id}

Execution Result:
{_format_result(flow_result)}

Input Context:
- Input Value: {input_value or 'N/A'}
- Inputs: {inputs or {}}

Please provide:
1. Execution quality assessment
2. Potential issues or errors
3. Recommendations for improvement
4. Suggestions for optimization
5. Next steps or follow-up actions

Be specific and actionable."""
<<<<<<< HEAD

=======
            
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
            request = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            }
<<<<<<< HEAD

            response = await client.post("/invoke", json=request, timeout=120.0)
            response.raise_for_status()

            result = response.json()

            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"DeepAgents analysis completed successfully (attempt {attempt + 1})")

=======
            
            response = await client.post("/invoke", json=request, timeout=120.0)
            response.raise_for_status()
            
            result = response.json()
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"DeepAgents analysis completed successfully (attempt {attempt + 1})")
            
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
            return {
                "analysis": result.get("messages", []),
                "result": result.get("result"),
            }
<<<<<<< HEAD

=======
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        except httpx.HTTPStatusError as e:
            last_exception = e
            if e.response.status_code >= 500 and attempt < max_retries:
                continue  # Retry on server errors
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DeepAgents analysis failed with status {e.response.status_code}: {e}")
            return None
<<<<<<< HEAD

=======
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_exception = e
            if attempt < max_retries:
                continue  # Retry on network/timeout errors
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DeepAgents analysis failed after {attempt + 1} attempts: {e}")
            return None
<<<<<<< HEAD

=======
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        except Exception as e:
            last_exception = e
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"DeepAgents analysis failed (non-fatal): {e}")
            return None
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    # Should not reach here, but handle gracefully
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"DeepAgents analysis failed after all retries: {last_exception}")
    return None


async def suggest_flow_improvements(
    flow_id: str,
    flow_spec: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Get suggestions for flow improvements using DeepAgents.
<<<<<<< HEAD

    Args:
        flow_id: The flow ID
        flow_spec: The flow specification/definition

=======
    
    Args:
        flow_id: The flow ID
        flow_spec: The flow specification/definition
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    Returns:
        Suggestions from DeepAgents, or None if disabled or failed (non-fatal)
    """
    client = get_deepagents_client()
    if client is None:
        return None
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    try:
        # Build analysis prompt
        prompt = f"""Analyze this AgentFlow/LangFlow flow definition and suggest improvements:

Flow ID: {flow_id}

Flow Structure:
{_format_flow_spec(flow_spec)}

Please provide:
1. Structural analysis of the flow
2. Potential bottlenecks or inefficiencies
3. Recommendations for optimization
4. Suggestions for additional nodes or connections
5. Best practices for this type of flow

Be specific and actionable."""
<<<<<<< HEAD

=======
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
<<<<<<< HEAD

        response = await client.post("/invoke", json=request, timeout=120.0)
        response.raise_for_status()

        result = response.json()

        import logging
        logger = logging.getLogger(__name__)
        logger.info("DeepAgents flow suggestions completed successfully")

=======
        
        response = await client.post("/invoke", json=request, timeout=120.0)
        response.raise_for_status()
        
        result = response.json()
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info("DeepAgents flow suggestions completed successfully")
        
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
        return {
            "suggestions": result.get("messages", []),
            "result": result.get("result"),
        }
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    except httpx.HTTPStatusError as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"DeepAgents flow analysis failed with status {e.response.status_code}: {e}")
        return None
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    except (httpx.RequestError, httpx.TimeoutException) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"DeepAgents flow analysis failed: {e}")
        return None
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"DeepAgents flow analysis failed (non-fatal): {e}")
        return None


def _format_result(result: Any) -> str:
    """Format flow result for analysis."""
    if isinstance(result, dict):
        # Try to extract key information
        if "output" in result:
            return f"Output: {result['output']}"
        elif "result" in result:
            return f"Result: {result['result']}"
        else:
            return str(result)
    elif isinstance(result, str):
        return result
    else:
        return str(result)


def _format_flow_spec(spec: Dict[str, Any]) -> str:
    """Format flow specification for analysis."""
    lines = []
<<<<<<< HEAD

=======
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    # Extract key information
    if "nodes" in spec:
        lines.append(f"Nodes: {len(spec['nodes'])}")
        node_types = {}
        for node in spec.get("nodes", []):
            node_type = node.get("data", {}).get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        lines.append(f"Node Types: {node_types}")
<<<<<<< HEAD

    if "edges" in spec:
        lines.append(f"Edges: {len(spec['edges'])}")

    if "name" in spec:
        lines.append(f"Name: {spec['name']}")

    if "description" in spec:
        lines.append(f"Description: {spec['description']}")

=======
    
    if "edges" in spec:
        lines.append(f"Edges: {len(spec['edges'])}")
    
    if "name" in spec:
        lines.append(f"Name: {spec['name']}")
    
    if "description" in spec:
        lines.append(f"Description: {spec['description']}")
    
>>>>>>> 0a025abe60b7633bd29e09340fd3b54080e7b084
    return "\n".join(lines) if lines else str(spec)


async def close_deepagents_client():
    """Close DeepAgents HTTP client."""
    global _deepagents_client
    if _deepagents_client is not None:
        await _deepagents_client.aclose()
        _deepagents_client = None

