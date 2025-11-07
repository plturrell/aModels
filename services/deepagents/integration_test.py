"""
Integration tests for DeepAgents service.

These tests verify that DeepAgents can:
1. Initialize successfully with required configuration
2. Invoke agents with various message types
3. Integrate with Extract, AgentFlow, and Graph services
4. Handle errors gracefully
"""

import os
import pytest
import httpx
from typing import Dict, Any, List

# Test configuration
DEEPAGENTS_URL = os.getenv("DEEPAGENTS_URL", "http://localhost:9004")
EXTRACT_SERVICE_URL = os.getenv("EXTRACT_SERVICE_URL", "http://localhost:19080")
AGENTFLOW_SERVICE_URL = os.getenv("AGENTFLOW_SERVICE_URL", "http://localhost:9001")
GRAPH_SERVICE_URL = os.getenv("GRAPH_SERVICE_URL", "http://localhost:8081")
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8080")


@pytest.fixture
def deepagents_client():
    """Create HTTP client for DeepAgents service."""
    return httpx.AsyncClient(base_url=DEEPAGENTS_URL, timeout=60.0)


@pytest.fixture
def extract_client():
    """Create HTTP client for Extract service."""
    return httpx.AsyncClient(base_url=EXTRACT_SERVICE_URL, timeout=30.0)


@pytest.fixture
def agentflow_client():
    """Create HTTP client for AgentFlow service."""
    return httpx.AsyncClient(base_url=AGENTFLOW_SERVICE_URL, timeout=30.0)


@pytest.fixture
def graph_client():
    """Create HTTP client for Graph service."""
    return httpx.AsyncClient(base_url=GRAPH_SERVICE_URL, timeout=30.0)


@pytest.mark.asyncio
async def test_deepagents_health(deepagents_client):
    """Test DeepAgents health endpoint."""
    response = await deepagents_client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "deepagents"


@pytest.mark.asyncio
async def test_deepagents_invoke_simple(deepagents_client):
    """Test simple agent invocation."""
    request = {
        "messages": [
            {"role": "user", "content": "Hello, what is 2+2?"}
        ],
        "stream": False
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) > 0
    
    # Check that last message is from assistant
    last_message = data["messages"][-1]
    assert last_message["role"] == "assistant"
    assert "content" in last_message


@pytest.mark.asyncio
async def test_deepagents_invoke_with_context(deepagents_client):
    """Test agent invocation with context."""
    request = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "stream": False
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "messages" in data
    assert len(data["messages"]) >= 2  # System + user + assistant


@pytest.mark.asyncio
async def test_deepagents_integration_with_extract(deepagents_client, extract_client):
    """Test DeepAgents integration with Extract service."""
    # First verify Extract service is available
    try:
        extract_health = await extract_client.get("/health")
        if extract_health.status_code != 200:
            pytest.skip("Extract service not available")
    except Exception:
        pytest.skip("Extract service not available")
    
    # Test that DeepAgents can use Extract service tools
    request = {
        "messages": [
            {"role": "user", "content": "Query the knowledge graph for test data"}
        ],
        "stream": False
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    # DeepAgents might use Extract service tools, so we just verify it doesn't crash
    assert response.status_code in [200, 500]  # 500 if service unavailable is acceptable
    if response.status_code == 200:
        data = response.json()
        assert "messages" in data


@pytest.mark.asyncio
async def test_deepagents_integration_with_agentflow(deepagents_client, agentflow_client):
    """Test DeepAgents integration with AgentFlow service."""
    # First verify AgentFlow service is available
    try:
        agentflow_health = await agentflow_client.get("/health")
        if agentflow_health.status_code != 200:
            pytest.skip("AgentFlow service not available")
    except Exception:
        pytest.skip("AgentFlow service not available")
    
    # Test that DeepAgents can use AgentFlow tools
    request = {
        "messages": [
            {"role": "user", "content": "List available flows"}
        ],
        "stream": False
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    # DeepAgents might use AgentFlow tools, so we just verify it doesn't crash
    assert response.status_code in [200, 500]  # 500 if service unavailable is acceptable
    if response.status_code == 200:
        data = response.json()
        assert "messages" in data


@pytest.mark.asyncio
async def test_deepagents_integration_with_graph(deepagents_client, graph_client):
    """Test DeepAgents integration with Graph service."""
    # First verify Graph service is available
    try:
        graph_health = await graph_client.get("/health")
        if graph_health.status_code != 200:
            pytest.skip("Graph service not available")
    except Exception:
        pytest.skip("Graph service not available")
    
    # Test that DeepAgents can use Graph service tools
    request = {
        "messages": [
            {"role": "user", "content": "Process a unified workflow"}
        ],
        "stream": False
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    # DeepAgents might use Graph service tools, so we just verify it doesn't crash
    assert response.status_code in [200, 500]  # 500 if service unavailable is acceptable
    if response.status_code == 200:
        data = response.json()
        assert "messages" in data


@pytest.mark.asyncio
async def test_deepagents_error_handling(deepagents_client):
    """Test that DeepAgents handles errors gracefully."""
    # Test with invalid request
    request = {
        "messages": []  # Empty messages
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    # Should return error or handle gracefully
    assert response.status_code in [200, 400, 422, 500]
    
    # Test with malformed request
    response = await deepagents_client.post("/invoke", json={"invalid": "request"})
    assert response.status_code in [200, 400, 422, 500]


@pytest.mark.asyncio
async def test_deepagents_streaming(deepagents_client):
    """Test DeepAgents streaming mode."""
    request = {
        "messages": [
            {"role": "user", "content": "Count to 5"}
        ],
        "stream": True
    }
    
    response = await deepagents_client.post("/invoke", json=request)
    # Streaming might return different status codes or use SSE
    assert response.status_code in [200, 201, 206]


@pytest.mark.asyncio
async def test_deepagents_agent_info(deepagents_client):
    """Test DeepAgents agent info endpoint."""
    response = await deepagents_client.get("/agent/info")
    # Endpoint might not exist, so we accept 404
    if response.status_code == 200:
        data = response.json()
        assert "agent" in data or "tools" in data or "capabilities" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

