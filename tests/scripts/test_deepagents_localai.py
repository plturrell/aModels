#!/usr/bin/env python3
"""Test DeepAgents → LocalAI integration."""

import requests
import json
import sys

def test_deepagents_health():
    """Test DeepAgents health endpoint."""
    try:
        response = requests.get("http://localhost:9004/healthz", timeout=5)
        print(f"✅ DeepAgents Health: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ DeepAgents Health Failed: {e}")
        return False

def test_localai_direct():
    """Test LocalAI directly."""
    try:
        response = requests.post(
            "http://localhost:8081/v1/chat/completions",
            json={
                "model": "general",
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_tokens": 20
            },
            timeout=30
        )
        print(f"✅ LocalAI Direct: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"   Response: {content[:100]}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ LocalAI Direct Failed: {e}")
        return False

def test_deepagents_invoke():
    """Test DeepAgents invoke endpoint (should call LocalAI)."""
    try:
        response = requests.post(
            "http://localhost:9004/invoke",
            json={
                "messages": [{"role": "user", "content": "Say hello in one word"}],
                "stream": False
            },
            timeout=60
        )
        print(f"✅ DeepAgents Invoke: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Full Response: {json.dumps(result, indent=2)}")
            messages = result.get("messages", [])
            if messages:
                print(f"   Last Message: {messages[-1]}")
            return True
        else:
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ DeepAgents Invoke Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Testing DeepAgents → LocalAI Integration ===\n")
    
    results = []
    
    print("1. Testing DeepAgents Health...")
    results.append(("DeepAgents Health", test_deepagents_health()))
    print()
    
    print("2. Testing LocalAI Direct...")
    results.append(("LocalAI Direct", test_localai_direct()))
    print()
    
    print("3. Testing DeepAgents → LocalAI (invoke endpoint)...")
    results.append(("DeepAgents → LocalAI", test_deepagents_invoke()))
    print()
    
    print("\n=== Summary ===")
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    if all(r[1] for r in results):
        print("\n🎉 All tests passed! DeepAgents → LocalAI integration is working.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check logs for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

