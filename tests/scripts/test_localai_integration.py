#!/usr/bin/env python3
"""Test script to verify LocalAI integration with DeepAgents."""

import os
import sys
import json
import httpx
from typing import Optional

# Configuration
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localhost:8081")
LOCALAI_MODEL = os.getenv("LOCALAI_MODEL", "general")

def test_localai_health() -> bool:
    """Test if LocalAI service is healthy."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ LocalAI health check passed: {response.json()}")
            return True
        else:
            print(f"❌ LocalAI health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ LocalAI health check error: {e}")
        return False

def test_localai_models() -> Optional[list]:
    """Test /v1/models endpoint."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [m.get("id", "") for m in data.get("data", [])]
            print(f"✅ Available models: {models}")
            return models
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return None

def test_localai_domains() -> Optional[list]:
    """Test /v1/domains endpoint."""
    try:
        response = httpx.get(f"{LOCALAI_URL}/v1/domains", timeout=5)
        if response.status_code == 200:
            data = response.json()
            domains = [d.get("id", "") for d in data.get("data", [])]
            print(f"✅ Available domains: {domains}")
            return domains
        else:
            print(f"❌ Domains endpoint failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Domains endpoint error: {e}")
        return None

def test_chat_completion(model: str) -> bool:
    """Test chat completion with specified model/domain."""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Say 'Hello' if you can read this."}
            ],
            "max_tokens": 50
        }
        response = httpx.post(
            f"{LOCALAI_URL}/v1/chat/completions",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"✅ Chat completion successful with model '{model}':")
            print(f"   Response: {content[:100]}...")
            return True
        else:
            error_text = response.text[:200]
            print(f"❌ Chat completion failed with model '{model}':")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {error_text}")
            return False
    except Exception as e:
        print(f"❌ Chat completion error with model '{model}': {e}")
        return False

def test_langchain_integration() -> bool:
    """Test if LangChain ChatOpenAI can connect to LocalAI."""
    try:
        from langchain_community.chat_models import ChatOpenAI
        
        model = ChatOpenAI(
            base_url=f"{LOCALAI_URL}/v1",
            api_key="not-needed",
            model=LOCALAI_MODEL,
            temperature=0.7,
            max_tokens=50
        )
        
        response = model.invoke("Say 'Hello' if you can read this.")
        print(f"✅ LangChain integration successful:")
        print(f"   Response: {response.content[:100]}...")
        return True
    except Exception as e:
        print(f"❌ LangChain integration error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("LocalAI Integration Test")
    print("=" * 60)
    print(f"LocalAI URL: {LOCALAI_URL}")
    print(f"Model/Domain: {LOCALAI_MODEL}")
    print()
    
    # Test 1: Health check
    print("Test 1: Health Check")
    print("-" * 60)
    if not test_localai_health():
        print("\n❌ Health check failed. LocalAI may not be running.")
        sys.exit(1)
    print()
    
    # Test 2: Models endpoint
    print("Test 2: Models Endpoint")
    print("-" * 60)
    models = test_localai_models()
    print()
    
    # Test 3: Domains endpoint
    print("Test 3: Domains Endpoint")
    print("-" * 60)
    domains = test_localai_domains()
    if domains:
        if LOCALAI_MODEL not in domains:
            print(f"⚠️  Warning: Model '{LOCALAI_MODEL}' not found in available domains")
            if domains:
                print(f"   Available domains: {domains}")
                print(f"   Using first available: {domains[0]}")
                test_model = domains[0]
            else:
                print("   No domains available")
                sys.exit(1)
        else:
            test_model = LOCALAI_MODEL
    else:
        test_model = LOCALAI_MODEL
    print()
    
    # Test 4: Chat completion
    print(f"Test 4: Chat Completion (model: {test_model})")
    print("-" * 60)
    if not test_chat_completion(test_model):
        print("\n❌ Chat completion failed.")
        sys.exit(1)
    print()
    
    # Test 5: LangChain integration
    print("Test 5: LangChain Integration")
    print("-" * 60)
    if not test_langchain_integration():
        print("\n❌ LangChain integration failed.")
        sys.exit(1)
    print()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    main()

