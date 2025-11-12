#!/usr/bin/env python3
"""Simple inline test that can be run directly"""

import os
import sys

try:
    import httpx
except ImportError:
    print("Installing httpx...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "httpx"])
    import httpx

# Use Docker service names
LOCALAI_URL = os.getenv("LOCALAI_URL", "http://localai:8081")
DEEPAGENTS_URL = os.getenv("DEEPAGENTS_URL", "http://deepagents-service:9004")
SEARCH_URL = os.getenv("SEARCH_URL", "http://search-inference:8090")
TRANSFORMERS_URL = os.getenv("TRANSFORMERS_URL", "http://transformers-service:9090")

def test_service(url, name, endpoint="/health"):
    """Test a service endpoint"""
    try:
        full_url = f"{url}{endpoint}"
        r = httpx.get(full_url, timeout=5)
        if r.status_code == 200:
            print(f"✅ {name}: {r.status_code}")
            return True
        else:
            print(f"⚠️  {name}: {r.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

print("=" * 60)
print("Simple LocalAI Integration Tests")
print("=" * 60)
print()

results = []

# Test LocalAI
print("Testing LocalAI...")
results.append(("LocalAI Health", test_service(LOCALAI_URL, "LocalAI", "/health")))
results.append(("LocalAI Domains", test_service(LOCALAI_URL, "LocalAI Domains", "/v1/domains")))

# Test DeepAgents
print("\nTesting DeepAgents...")
results.append(("DeepAgents Health", test_service(DEEPAGENTS_URL, "DeepAgents", "/healthz")))

# Test Search
print("\nTesting Search-inference...")
results.append(("Search Health", test_service(SEARCH_URL, "Search-inference", "/health")))

# Test Transformers
print("\nTesting Transformers...")
results.append(("Transformers Health", test_service(TRANSFORMERS_URL, "Transformers", "/health")))

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
passed = sum(1 for _, r in results if r)
total = len(results)
for name, result in results:
    status = "✅" if result else "❌"
    print(f"{status} {name}")
print(f"\nPassed: {passed}/{total}")

sys.exit(0 if passed == total else 1)

