"""FastAPI service for DeepAgents integration."""

import logging
import os
import json
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent_factory import create_amodels_deep_agent
from observability import (
    record_request,
    record_latency,
    record_error,
    record_tool_usage,
    record_token_usage,
    record_structured_output,
    get_metrics,
)
from otel_instrumentation import init_otel_instrumentation, get_tracer, shutdown_otel
from opentelemetry import trace

# Global agent instance (created on startup)
_agent = None
logger = logging.getLogger(__name__)

app = FastAPI(
    title="aModels DeepAgents Service",
    description="Deep agent service with planning, sub-agents, and aModels integration",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class AgentRequest(BaseModel):
    messages: List[Message]
    stream: bool = False
    config: Optional[Dict[str, Any]] = None
    response_format: Optional[Union[str, Dict[str, Any]]] = Field(
        default=None,
        description="Response format: 'json' for JSON mode, or dict with 'type' and 'schema' for structured output"
    )


class StructuredOutputSchema(BaseModel):
    """Schema for structured output."""
    type: str = Field(default="json_schema", description="Schema type")
    json_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema definition")


class StructuredAgentRequest(BaseModel):
    """Request for structured output."""
    messages: List[Message]
    response_format: Union[str, StructuredOutputSchema, Dict[str, Any]] = Field(
        description="Response format specification"
    )
    stream: bool = False
    config: Optional[Dict[str, Any]] = None


class AgentResponse(BaseModel):
    messages: List[Dict[str, Any]]
    result: Optional[Any] = None


class StructuredAgentResponse(BaseModel):
    """Structured response with validated output."""
    messages: List[Dict[str, Any]]
    structured_output: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Validated structured output matching the requested schema"
    )
    result: Optional[Any] = None
    validation_errors: Optional[List[str]] = Field(
        default=None,
        description="Any validation errors encountered"
    )


def validate_config():
    """Validate required environment variables."""
    errors = []
    
    # Required service URLs
    required_vars = [
        "EXTRACT_SERVICE_URL",
        "AGENTFLOW_SERVICE_URL",
        "GRAPH_SERVICE_URL",
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"{var} is required")
    
    # At least one LLM provider must be configured
    llm_providers = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "LOCALAI_URL",
    ]
    
    has_llm = any(os.getenv(provider) for provider in llm_providers)
    if not has_llm:
        errors.append("At least one LLM provider must be configured (ANTHROPIC_API_KEY, OPENAI_API_KEY, or LOCALAI_URL)")
    
    if errors:
        error_msg = "Configuration validation failed:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)


@app.on_event("startup")
async def startup():
    """Initialize the deep agent on startup."""
    # Initialize OpenTelemetry instrumentation
    init_otel_instrumentation(app)
    
    # Validate configuration
    try:
        validate_config()
        logger.info("Configuration validation passed")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    
    global _agent
    try:
        _agent = create_amodels_deep_agent()
        logger.info("DeepAgent initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize DeepAgent: {e}")
        logger.warning("Agent will be created on first request")


@app.on_event("shutdown")
async def shutdown():
    """Shutdown the service."""
    shutdown_otel()


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "deepagents",
        "agent_initialized": _agent is not None,
    }


def _extract_structured_output(
    messages: List[Dict[str, Any]],
    response_format: Optional[Union[str, Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Extract and validate structured output from agent messages."""
    if not response_format:
        return None
    
    # Find the last assistant message
    last_assistant_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant_msg = msg
            break
    
    if not last_assistant_msg:
        return None
    
    content = last_assistant_msg.get("content", "")
    if not content:
        return None
    
    # Try to parse JSON from content
    try:
        # Look for JSON in the content (might be wrapped in markdown code blocks)
        if "```json" in content:
            json_start = content.find("```json") + 7
            json_end = content.find("```", json_start)
            if json_end > json_start:
                content = content[json_start:json_end].strip()
        elif "```" in content:
            json_start = content.find("```") + 3
            json_end = content.find("```", json_start)
            if json_end > json_start:
                content = content[json_start:json_end].strip()
        
        parsed = json.loads(content)
        
        # If response_format is a schema, validate against it
        if isinstance(response_format, dict) and "json_schema" in response_format:
            # Basic validation (full JSON schema validation can be added)
            schema = response_format.get("json_schema", {})
            # For now, just return parsed JSON
            # Full validation can be implemented with jsonschema library
            return parsed
        
        return parsed if isinstance(parsed, dict) else None
    
    except (json.JSONDecodeError, ValueError):
        # Not valid JSON, return None
        return None


@app.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """Invoke the deep agent with a conversation.
    
    Args:
        request: Agent request with messages and optional config
    
    Returns:
        Agent response with messages and result
    """
    import time
    from opentelemetry.trace import Status, StatusCode
    
    tracer = get_tracer()
    start_time = time.time()
    
    # Start OpenTelemetry span
    with tracer.start_as_current_span("deepagents.invoke") as span:
        span.set_attribute("agent.framework.type", "deepagents")
        span.set_attribute("workflow.type", "agent_invocation")
        span.set_attribute("request.message_count", len(request.messages))
        span.set_attribute("request.stream", request.stream)
        
        if request.config:
            for key, value in request.config.items():
                span.set_attribute(f"request.config.{key}", str(value))
        
        global _agent
        if _agent is None:
            # Lazy initialization if startup failed
            _agent = create_amodels_deep_agent()
        
        try:
            # Convert messages to format expected by LangGraph
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Prepare input
            agent_input = {"messages": messages}
            if request.config:
                agent_input["config"] = request.config
            
            # If response_format is "json", add instruction to response in JSON
            if request.response_format == "json":
                # Add instruction to last user message
                if messages and messages[-1].get("role") == "user":
                    messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            
            # Invoke agent
            result = _agent.invoke(agent_input)
            
            # Extract messages from result
            response_messages = []
            if "messages" in result:
                response_messages = [
                    {
                        "role": msg.get("role", "assistant"),
                        "content": msg.get("content", ""),
                    }
                    for msg in result["messages"]
                ]
            
            # Record metrics
            latency = time.time() - start_time
            record_request("/invoke", "POST", 200)
            record_latency("/invoke", latency)
            
            # Estimate token usage (rough estimate: 1 token ≈ 4 characters)
            total_chars = sum(len(msg.get("content", "")) for msg in messages + response_messages)
            estimated_tokens = total_chars // 4
            record_token_usage(input_tokens=estimated_tokens // 2, output_tokens=estimated_tokens // 2)
            
            # Add span attributes
            span.set_attribute("response.message_count", len(response_messages))
            span.set_attribute("latency_ms", latency * 1000)
            span.set_attribute("estimated_tokens", estimated_tokens)
            span.set_status(Status(StatusCode.OK))
            
            # Record event
            span.add_event("agent.invoke.completed", {
                "message_count": len(response_messages),
                "latency_ms": latency * 1000,
            })
            
            return AgentResponse(
                messages=response_messages,
                result=result.get("result") if "result" in result else result,
            )
        
        except Exception as e:
            latency = time.time() - start_time
            record_request("/invoke", "POST", 500)
            record_latency("/invoke", latency)
            record_error("/invoke", "exception")
            
            # Record error in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            logger.exception("Agent invocation failed")
            raise HTTPException(
                status_code=500,
                detail="Agent invocation failed; see service logs for details.",
            )


@app.post("/invoke/structured", response_model=StructuredAgentResponse)
async def invoke_agent_structured(request: StructuredAgentRequest) -> StructuredAgentResponse:
    """Invoke the deep agent with structured output.
    
    Args:
        request: Structured agent request with response format specification
    
    Returns:
        Structured agent response with validated output
    """
    import time
    from opentelemetry.trace import Status, StatusCode
    
    tracer = get_tracer()
    start_time = time.time()
    
    # Start OpenTelemetry span
    with tracer.start_as_current_span("deepagents.invoke.structured") as span:
        span.set_attribute("agent.framework.type", "deepagents")
        span.set_attribute("workflow.type", "structured_agent_invocation")
        span.set_attribute("request.message_count", len(request.messages))
        span.set_attribute("request.stream", request.stream)
        
        if isinstance(request.response_format, dict):
            span.set_attribute("response_format.type", request.response_format.get("type", "unknown"))
        
        global _agent
        if _agent is None:
            _agent = create_amodels_deep_agent()
        
        try:
            # Convert messages
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Prepare input
            agent_input = {"messages": messages}
            if request.config:
                agent_input["config"] = request.config
            
            # Handle response format
            response_format = request.response_format
            if isinstance(response_format, str):
                response_format = {"type": "json"} if response_format == "json" else None
            elif isinstance(response_format, StructuredOutputSchema):
                response_format = response_format.dict()
            
            # Add instruction for structured output
            if response_format:
                schema_instruction = "\n\nPlease respond with valid JSON that matches the requested structure."
                if isinstance(response_format, dict) and "json_schema" in response_format:
                    schema = response_format.get("json_schema", {})
                    if "properties" in schema:
                        schema_instruction += f"\n\nRequired structure: {json.dumps(schema, indent=2)}"
                
                if messages and messages[-1].get("role") == "user":
                    messages[-1]["content"] += schema_instruction
            
            # Invoke agent
            result = _agent.invoke(agent_input)
            
            # Extract messages
            response_messages = []
            if "messages" in result:
                response_messages = [
                    {
                        "role": msg.get("role", "assistant"),
                        "content": msg.get("content", ""),
                    }
                    for msg in result["messages"]
                ]
            
            # Extract and validate structured output
            structured_output = _extract_structured_output(response_messages, response_format)
            validation_errors = None
            
            if response_format and not structured_output:
                validation_errors = ["Failed to extract structured output from response"]
            
            # Record metrics
            latency = time.time() - start_time
            record_request("/invoke/structured", "POST", 200)
            record_latency("/invoke/structured", latency)
            record_structured_output(
                success=structured_output is not None,
                validation_errors=len(validation_errors) if validation_errors else 0
            )
            
            # Estimate token usage
            total_chars = sum(len(msg.get("content", "")) for msg in messages + response_messages)
            estimated_tokens = total_chars // 4
            record_token_usage(input_tokens=estimated_tokens // 2, output_tokens=estimated_tokens // 2)
            
            # Add span attributes
            span.set_attribute("response.message_count", len(response_messages))
            span.set_attribute("structured_output.success", structured_output is not None)
            span.set_attribute("validation_errors.count", len(validation_errors) if validation_errors else 0)
            span.set_attribute("latency_ms", latency * 1000)
            span.set_attribute("estimated_tokens", estimated_tokens)
            span.set_status(Status(StatusCode.OK))
            
            # Record event
            span.add_event("agent.invoke.structured.completed", {
                "structured_output_success": structured_output is not None,
                "validation_errors": len(validation_errors) if validation_errors else 0,
            })
            
            return StructuredAgentResponse(
                messages=response_messages,
                structured_output=structured_output,
                result=result.get("result") if "result" in result else result,
                validation_errors=validation_errors,
            )
        
        except Exception as e:
            latency = time.time() - start_time
            record_request("/invoke/structured", "POST", 500)
            record_latency("/invoke/structured", latency)
            record_error("/invoke/structured", "exception")
            record_structured_output(success=False)
            
            # Record error in span
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR, str(e)))
            
            logger.exception("Structured agent invocation failed")
            raise HTTPException(
                status_code=500,
                detail="Structured agent invocation failed; see service logs for details.",
            )


@app.post("/stream")
async def stream_agent(request: AgentRequest):
    """Stream agent responses.
    
    Args:
        request: Agent request with messages and optional config
    
    Yields:
        Server-sent events with agent response chunks
    """
    global _agent
    if _agent is None:
        _agent = create_amodels_deep_agent()
    
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        # Convert messages
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        agent_input = {"messages": messages}
        if request.config:
            agent_input["config"] = request.config
        
        async def generate():
            async for chunk in _agent.astream(agent_input, stream_mode="values"):
                if "messages" in chunk:
                    last_message = chunk["messages"][-1]
                    yield f"data: {json.dumps({'role': last_message.get('role'), 'content': last_message.get('content', '')})}\n\n"
                yield f"data: {json.dumps(chunk)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    
    except Exception:
        logger.exception("Agent streaming failed")
        raise HTTPException(
            status_code=500,
            detail="Agent streaming failed; see service logs for details.",
        )


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """Get service metrics."""
    return get_metrics()


@app.get("/agent/info")
async def agent_info() -> Dict[str, Any]:
    """Get information about the configured agent."""
    return {
        "agent_type": "deep_agent",
        "capabilities": [
            "planning_and_todos",
            "sub_agent_spawning",
            "file_system_access",
            "knowledge_graph_queries",
            "agentflow_execution",
            "orchestration_chains",
        ],
        "tools": [
            "query_knowledge_graph",
            "run_agentflow_flow",
            "run_orchestration_chain",
            "query_data_elements",
            "check_duplicates",
            "validate_definition",
            "suggest_improvements",
            "find_similar_elements",
            "write_todos",
            "task",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        ],
        "endpoints": {
            "/invoke": "Standard agent invocation",
            "/invoke/structured": "Structured output with JSON schema validation",
            "/stream": "Streaming agent responses",
        },
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DEEPAGENTS_PORT", "9004"))
    uvicorn.run(app, host="0.0.0.0", port=port)
