"""FastAPI service for DeepAgents integration."""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_factory import create_amodels_deep_agent

# Global agent instance (created on startup)
_agent = None

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


class AgentResponse(BaseModel):
    messages: List[Dict[str, Any]]
    result: Optional[Any] = None


@app.on_event("startup")
async def startup():
    """Initialize the deep agent on startup."""
    global _agent
    try:
        _agent = create_amodels_deep_agent()
        print("DeepAgent initialized successfully")
    except Exception as e:
        print(f"Warning: Failed to initialize DeepAgent: {e}")
        print("Agent will be created on first request")


@app.get("/healthz")
async def healthz() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "deepagents",
        "agent_initialized": _agent is not None,
    }


@app.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """Invoke the deep agent with a conversation.
    
    Args:
        request: Agent request with messages and optional config
    
    Returns:
        Agent response with messages and result
    """
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
        
        return AgentResponse(
            messages=response_messages,
            result=result.get("result") if "result" in result else result,
        )
    
    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=f"Agent invocation failed: {error_detail}")


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
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent streaming failed: {str(e)}")


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
            "write_todos",
            "task",
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DEEPAGENTS_PORT", "9004"))
    uvicorn.run(app, host="0.0.0.0", port=port)

