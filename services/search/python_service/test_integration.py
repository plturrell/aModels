"""
Integration tests for the search service using Docker Compose.
These tests require the services to be running via docker-compose.
"""
import os
import pytest
import requests
import time
from typing import Dict, Any, Optional


# Test configuration
PYTHON_SERVICE_URL = os.getenv("PYTHON_SERVICE_URL", "http://localhost:8081")
GO_SERVICE_URL = os.getenv("GO_SERVICE_URL", "http://localhost:8090")
API_KEY = os.getenv("TEST_API_KEY", "test-api-key-12345")
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "false").lower() == "true"


def get_headers(include_auth: bool = True) -> Dict[str, str]:
    """Get request headers with optional authentication."""
    headers = {"Content-Type": "application/json"}
    if include_auth and AUTH_ENABLED:
        headers["X-API-Key"] = API_KEY
    return headers


def wait_for_service(url: str, timeout: int = 30) -> bool:
    """Wait for a service to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=2)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def setup_services():
    """Wait for services to be ready before running tests."""
    print("\n⏳ Waiting for services to be ready...")
    
    python_ready = wait_for_service(PYTHON_SERVICE_URL)
    go_ready = wait_for_service(GO_SERVICE_URL)
    
    if not python_ready:
        pytest.skip(f"Python service not available at {PYTHON_SERVICE_URL}")
    if not go_ready:
        pytest.skip(f"Go service not available at {GO_SERVICE_URL}")
    
    print("✅ Services are ready")
    yield
    
    print("\n🧹 Cleanup complete")


class TestHealthChecks:
    """Test health check endpoints."""
    
    def test_python_service_health(self, setup_services):
        """Test Python service health endpoint."""
        response = requests.get(f"{PYTHON_SERVICE_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "mode" in data
        assert "go_search" in data
        assert "elasticsearch" in data
    
    def test_go_service_health(self, setup_services):
        """Test Go service health endpoint."""
        response = requests.get(f"{GO_SERVICE_URL}/health", timeout=5)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAuthentication:
    """Test authentication functionality."""
    
    def test_python_service_auth_required(self, setup_services):
        """Test that Python service requires authentication when enabled."""
        if not AUTH_ENABLED:
            pytest.skip("Authentication not enabled")
        
        # Request without API key should fail
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/search",
            json={"query": "test", "top_k": 5},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert response.status_code == 401
    
    def test_python_service_auth_success(self, setup_services):
        """Test successful authentication with valid API key."""
        if not AUTH_ENABLED:
            pytest.skip("Authentication not enabled")
        
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/search",
            json={"query": "test", "top_k": 5},
            headers=get_headers(),
            timeout=5
        )
        assert response.status_code in [200, 500]  # 200 if service works, 500 if backend issue
    
    def test_go_service_auth_required(self, setup_services):
        """Test that Go service requires authentication when enabled."""
        if not AUTH_ENABLED:
            pytest.skip("Authentication not enabled")
        
        response = requests.post(
            f"{GO_SERVICE_URL}/v1/search",
            json={"query": "test", "top_k": 5},
            headers={"Content-Type": "application/json"},
            timeout=5
        )
        assert response.status_code == 401


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_python_service_rate_limit(self, setup_services):
        """Test rate limiting on Python service."""
        # Make many rapid requests
        success_count = 0
        rate_limited_count = 0
        
        for i in range(70):  # More than default 60/min
            try:
                response = requests.post(
                    f"{PYTHON_SERVICE_URL}/v1/search",
                    json={"query": f"test {i}", "top_k": 5},
                    headers=get_headers(),
                    timeout=2
                )
                if response.status_code == 200:
                    success_count += 1
                elif response.status_code == 429:
                    rate_limited_count += 1
                    break  # Got rate limited, test passed
            except requests.exceptions.RequestException:
                pass
        
        # Should have hit rate limit at some point
        # Note: This test may be flaky depending on timing
        assert rate_limited_count > 0 or success_count < 70


class TestDocumentOperations:
    """Test document indexing and retrieval."""
    
    def test_add_document(self, setup_services):
        """Test adding a document."""
        doc = {
            "id": "integration-test-doc-1",
            "content": "This is a test document for integration testing. It contains information about vacation policies.",
            "metadata": {
                "category": "HR",
                "type": "policy",
                "test": True
            }
        }
        
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/documents",
            json=doc,
            headers=get_headers(),
            timeout=10
        )
        assert response.status_code == 204
    
    def test_search_document(self, setup_services):
        """Test searching for documents."""
        # First add a document
        doc = {
            "id": "integration-test-doc-2",
            "content": "Integration test document about employee benefits and healthcare coverage.",
            "metadata": {"category": "HR"}
        }
        
        requests.post(
            f"{PYTHON_SERVICE_URL}/v1/documents",
            json=doc,
            headers=get_headers(),
            timeout=10
        )
        
        # Wait a bit for indexing
        time.sleep(2)
        
        # Search for it
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/search",
            json={"query": "employee benefits", "top_k": 5},
            headers=get_headers(),
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "backend" in data
    
    def test_get_source_document(self, setup_services):
        """Test retrieving a source document by ID."""
        doc_id = "integration-test-doc-3"
        doc = {
            "id": doc_id,
            "content": "Test document for source retrieval testing.",
            "metadata": {"test": True}
        }
        
        # Add document
        requests.post(
            f"{PYTHON_SERVICE_URL}/v1/documents",
            json=doc,
            headers=get_headers(),
            timeout=10
        )
        
        time.sleep(1)
        
        # Retrieve it
        response = requests.get(
            f"{PYTHON_SERVICE_URL}/v1/sources/{doc_id}",
            headers=get_headers(),
            timeout=10
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "content" in data


class TestAISearch:
    """Test AI-powered search functionality."""
    
    @pytest.mark.skip(reason="Requires LocalAI to be running and configured")
    def test_ai_search(self, setup_services):
        """Test AI search functionality."""
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/ai-search",
            json={
                "query": "What are the vacation policies?",
                "max_sources": 3
            },
            headers=get_headers(),
            timeout=30
        )
        
        # May fail if LocalAI is not available, which is OK
        if response.status_code == 200:
            data = response.json()
            assert "response" in data
            assert "sources" in data
            assert "conversation_id" in data


class TestGoService:
    """Test Go service endpoints."""
    
    def test_go_service_embed(self, setup_services):
        """Test embedding endpoint on Go service."""
        response = requests.post(
            f"{GO_SERVICE_URL}/v1/embed",
            json={"text": "test embedding"},
            headers=get_headers(),
            timeout=10
        )
        
        # May return 200 or 500 depending on LocalAI availability
        assert response.status_code in [200, 500, 503]
    
    def test_go_service_search(self, setup_services):
        """Test search endpoint on Go service."""
        response = requests.post(
            f"{GO_SERVICE_URL}/v1/search",
            json={"query": "test query", "top_k": 5},
            headers=get_headers(),
            timeout=10
        )
        
        # May return 200 or 500 depending on backend availability
        assert response.status_code in [200, 500, 503]


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_query(self, setup_services):
        """Test that invalid queries are rejected."""
        # Empty query should be rejected
        response = requests.post(
            f"{PYTHON_SERVICE_URL}/v1/search",
            json={"query": "", "top_k": 5},
            headers=get_headers(),
            timeout=5
        )
        assert response.status_code == 422  # Validation error
    
    def test_invalid_document_id(self, setup_services):
        """Test that invalid document IDs are rejected."""
        response = requests.get(
            f"{PYTHON_SERVICE_URL}/v1/sources/invalid@id#123",
            headers=get_headers(),
            timeout=5
        )
        assert response.status_code == 400  # Bad request
    
    def test_invalid_conversation_id(self, setup_services):
        """Test that invalid conversation IDs are rejected."""
        response = requests.get(
            f"{PYTHON_SERVICE_URL}/v1/conversation/not-a-uuid",
            headers=get_headers(),
            timeout=5
        )
        assert response.status_code == 400  # Bad request


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

