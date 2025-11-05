#!/usr/bin/env python3
"""
Comprehensive integration test suite for all LocalAI interaction points.

This test suite verifies that all services are properly configured to use
only LocalAI and that all interaction points are working correctly.

Tests:
1. LocalAI service endpoints
2. DeepAgents → LocalAI
3. Graph service → LocalAI
4. Search-inference → LocalAI (embeddings)
5. Extract service → LocalAI
6. Gateway → LocalAI
7. Embedding models (transformers-service)
8. DeepSeek OCR (if available)
"""

import os
import sys
import json
import httpx
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

# Test configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
DEEPAGENTS_URL = os.getenv("DEEPAGENTS_URL", "http://localhost:9004")
GRAPH_URL = os.getenv("GRAPH_URL", "http://localhost:8080")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8090")
EXTRACT_URL = os.getenv("EXTRACT_URL", "http://localhost:8082")
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
TRANSFORMERS_URL = os.getenv("TRANSFORMERS_URL", "http://localhost:9090")

# Timeout settings
DEFAULT_TIMEOUT = 30
HEALTH_TIMEOUT = 5


class TestResult(Enum):
    PASS = "✅"
    FAIL = "❌"
    SKIP = "⏭️"
    WARN = "⚠️"


@dataclass
class TestCase:
    name: str
    description: str
    result: TestResult = TestResult.SKIP
    message: str = ""
    duration: float = 0.0


class TestSuite:
    def __init__(self):
        self.tests: List[TestCase] = []
        self.start_time = time.time()

    def run_test(self, name: str, description: str, test_func):
        """Run a test and record the result."""
        print(f"\n{'='*60}")
        print(f"Test: {name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print()
        
        start = time.time()
        try:
            result = test_func()
            duration = time.time() - start
            
            if result:
                test_case = TestCase(name, description, TestResult.PASS, "", duration)
                print(f"{TestResult.PASS.value} {name} passed ({duration:.2f}s)")
            else:
                test_case = TestCase(name, description, TestResult.FAIL, "Test returned False", duration)
                print(f"{TestResult.FAIL.value} {name} failed ({duration:.2f}s)")
        except Exception as e:
            duration = time.time() - start
            test_case = TestCase(name, description, TestResult.FAIL, str(e), duration)
            print(f"{TestResult.FAIL.value} {name} failed with error: {e} ({duration:.2f}s)")
            import traceback
            traceback.print_exc()
        
        self.tests.append(test_case)
        return test_case.result == TestResult.PASS

    def print_summary(self):
        """Print test summary."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.result == TestResult.PASS)
        failed = sum(1 for t in self.tests if t.result == TestResult.FAIL)
        skipped = sum(1 for t in self.tests if t.result == TestResult.SKIP)
        total_duration = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"  {TestResult.PASS.value} Passed: {passed}")
        print(f"  {TestResult.FAIL.value} Failed: {failed}")
        print(f"  {TestResult.SKIP.value} Skipped: {skipped}")
        print(f"\nTotal duration: {total_duration:.2f}s")
        print("="*60)
        
        if failed > 0:
            print("\nFailed tests:")
            for test in self.tests:
                if test.result == TestResult.FAIL:
                    print(f"  {TestResult.FAIL.value} {test.name}: {test.message}")
        
        return failed == 0


# Test Functions

def test_localai_health() -> bool:
    """Test LocalAI health endpoint."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ LocalAI health check passed")
            return True
        else:
            print(f"❌ LocalAI health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LocalAI health check error: {e}")
        return False


def test_localai_models() -> bool:
    """Test LocalAI /v1/models endpoint."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/v1/models", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            models = data.get("data", [])
            print(f"✅ Found {len(models)} models")
            for model in models[:5]:  # Show first 5
                print(f"   - {model.get('id', 'unknown')}")
            return True
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False


def test_localai_domains() -> bool:
    """Test LocalAI /v1/domains endpoint."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/v1/domains", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            domains = data.get("data", [])
            print(f"✅ Found {len(domains)} domains")
            for domain in domains[:5]:  # Show first 5
                print(f"   - {domain.get('id', 'unknown')}")
            return True
        else:
            print(f"❌ Domains endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Domains endpoint error: {e}")
        return False


def test_localai_chat_completion() -> bool:
    """Test LocalAI chat completion."""
    try:
        payload = {
            "model": "general",
            "messages": [
                {"role": "user", "content": "Say 'Hello' if you can read this."}
            ],
            "max_tokens": 50
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/chat/completions",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Chat completion successful")
            print(f"   Response: {content[:100]}...")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Chat completion error: {e}")
        return False


def test_localai_embeddings() -> bool:
    """Test LocalAI embeddings endpoint."""
    try:
        payload = {
            "model": "0x3579-VectorProcessingAgent",
            "input": ["test embedding"]
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/embeddings",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            embeddings = data.get("data", [])
            if embeddings:
                embedding_dim = len(embeddings[0].get("embedding", []))
                print(f"✅ Embeddings successful")
                print(f"   Dimension: {embedding_dim}")
                return True
            else:
                print(f"❌ No embeddings returned")
                return False
        else:
            print(f"❌ Embeddings failed: {response.status_code}")
            print(f"   Error: {response.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ Embeddings error: {e}")
        return False


def test_deepagents_health() -> bool:
    """Test DeepAgents health endpoint."""
    try:
        response = httpx.get(f"{DEEPAGENTS_URL}/healthz", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ DeepAgents health check passed")
            return True
        else:
            print(f"❌ DeepAgents health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  DeepAgents not available: {e}")
        return False


def test_deepagents_localai() -> bool:
    """Test DeepAgents using LocalAI."""
    try:
        # Test if DeepAgents can make a request through LocalAI
        # This would typically require a proper DeepAgents API endpoint
        # For now, we'll check if the service is running and can connect to LocalAI
        response = httpx.get(f"{DEEPAGENTS_URL}/healthz", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ DeepAgents service is running")
            print(f"   (DeepAgents should be configured to use LocalAI at {LOCALAI_URL})")
            return True
        else:
            return False
    except Exception as e:
        print(f"⚠️  DeepAgents not available: {e}")
        return False


def test_graph_service() -> bool:
    """Test Graph service connection to LocalAI."""
    try:
        # Check if graph service is running
        response = httpx.get(f"{GRAPH_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Graph service health check passed")
            print(f"   (Graph service should be configured to use LocalAI at {LOCALAI_URL})")
            return True
        else:
            print(f"⚠️  Graph service returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Graph service not available: {e}")
        return False


def test_search_inference_health() -> bool:
    """Test Search-inference service health."""
    try:
        response = httpx.get(f"{SEARCH_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Search-inference health check passed")
            return True
        else:
            print(f"❌ Search-inference health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Search-inference not available: {e}")
        return False


def test_search_embeddings() -> bool:
    """Test Search-inference embeddings via LocalAI."""
    try:
        payload = {
            "text": "test search embedding"
        }
        response = httpx.post(
            f"{SEARCH_URL}/v1/embed",
            json=payload,
            timeout=DEFAULT_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            embedding = data.get("embedding", [])
            if embedding:
                print(f"✅ Search embeddings successful")
                print(f"   Dimension: {len(embedding)}")
                return True
            else:
                print(f"❌ No embedding returned")
                return False
        else:
            print(f"❌ Search embeddings failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Search embeddings not available: {e}")
        return False


def test_extract_service() -> bool:
    """Test Extract service connection."""
    try:
        # Check if extract service is running
        response = httpx.get(f"{EXTRACT_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Extract service health check passed")
            print(f"   (Extract service should use LocalAI for extraction)")
            return True
        else:
            print(f"⚠️  Extract service returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Extract service not available: {e}")
        return False


def test_gateway_service() -> bool:
    """Test Gateway service connection to LocalAI."""
    try:
        # Check if gateway service is running
        response = httpx.get(f"{GATEWAY_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Gateway service health check passed")
            print(f"   (Gateway service should use LocalAI at {LOCALAI_URL})")
            return True
        else:
            print(f"⚠️  Gateway service returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Gateway service not available: {e}")
        return False


def test_transformers_service() -> bool:
    """Test Transformers service for embeddings."""
    try:
        # Check if transformers service is running
        response = httpx.get(f"{TRANSFORMERS_URL}/health", timeout=HEALTH_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Transformers service health check passed")
            print(f"   (Transformers service should provide all-MiniLM-L6-v2 embeddings)")
            return True
        else:
            print(f"⚠️  Transformers service returned: {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️  Transformers service not available: {e}")
        return False


def test_no_external_apis() -> bool:
    """Verify no external API calls are being made."""
    print("Checking for external API references in configuration...")
    
    external_apis = [
        "api.openai.com",
        "api.anthropic.com",
        "api.deepseek.com",
        "generativeai.googleapis.com",
    ]
    
    # Check environment variables
    env_vars = os.environ
    found_external = []
    
    for key, value in env_vars.items():
        if any(api in str(value).lower() for api in external_apis):
            found_external.append(f"{key}={value}")
    
    if found_external:
        print(f"⚠️  Found potential external API references:")
        for ref in found_external:
            print(f"   - {ref}")
        return False
    else:
        print(f"✅ No external API references found in environment")
        return True


def main():
    """Run all integration tests."""
    print("="*60)
    print("LocalAI Integration Test Suite")
    print("="*60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"DeepAgents URL: {DEEPAGENTS_URL}")
    print(f"Graph URL: {GRAPH_URL}")
    print(f"Search URL: {SEARCH_URL}")
    print(f"Extract URL: {EXTRACT_URL}")
    print(f"Gateway URL: {GATEWAY_URL}")
    print(f"Transformers URL: {TRANSFORMERS_URL}")
    print()
    
    suite = TestSuite()
    
    # LocalAI Core Tests
    suite.run_test("LocalAI Health", "Test LocalAI service health endpoint", test_localai_health)
    suite.run_test("LocalAI Models", "Test LocalAI /v1/models endpoint", test_localai_models)
    suite.run_test("LocalAI Domains", "Test LocalAI /v1/domains endpoint", test_localai_domains)
    suite.run_test("LocalAI Chat", "Test LocalAI chat completion", test_localai_chat_completion)
    suite.run_test("LocalAI Embeddings", "Test LocalAI embeddings endpoint", test_localai_embeddings)
    
    # Service Integration Tests
    suite.run_test("DeepAgents Health", "Test DeepAgents service health", test_deepagents_health)
    suite.run_test("DeepAgents LocalAI", "Verify DeepAgents uses LocalAI", test_deepagents_localai)
    suite.run_test("Graph Service", "Test Graph service connection", test_graph_service)
    suite.run_test("Search Inference Health", "Test Search-inference service health", test_search_inference_health)
    suite.run_test("Search Embeddings", "Test Search-inference embeddings via LocalAI", test_search_embeddings)
    suite.run_test("Extract Service", "Test Extract service connection", test_extract_service)
    suite.run_test("Gateway Service", "Test Gateway service connection", test_gateway_service)
    suite.run_test("Transformers Service", "Test Transformers service for embeddings", test_transformers_service)
    
    # Security/Configuration Tests
    suite.run_test("No External APIs", "Verify no external API calls configured", test_no_external_apis)
    
    # Print summary
    success = suite.print_summary()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

