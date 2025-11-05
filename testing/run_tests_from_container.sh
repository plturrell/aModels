#!/bin/bash
# Run tests from within DeepAgents container (which has network access)

set -e

echo "=========================================="
echo "Running Tests from DeepAgents Container"
echo "=========================================="
echo ""

docker exec deepagents-service python3 << 'PYTHON_SCRIPT'
import requests
import json
import sys

localai = 'http://localai:8080'
print('=' * 60)
print('LocalAI Integration Test Suite')
print('=' * 60)
print(f'LocalAI URL: {localai}')
print()

passed = 0
total = 0

# Test 1: LocalAI Health
total += 1
try:
    r = requests.get(f'{localai}/health', timeout=5)
    if r.status_code == 200:
        print(f'✅ LocalAI Health: {r.status_code}')
        passed += 1
    else:
        print(f'❌ LocalAI Health: {r.status_code}')
except Exception as e:
    print(f'❌ LocalAI Health: {e}')

# Test 2: LocalAI Domains
total += 1
try:
    r = requests.get(f'{localai}/v1/domains', timeout=5)
    if r.status_code == 200:
        data = r.json()
        domains = data.get('data', [])
        print(f'✅ LocalAI Domains: Found {len(domains)} domains')
        if len(domains) > 0:
            for d in domains[:5]:
                print(f'   - {d.get("id", "unknown")}')
        passed += 1
    else:
        print(f'❌ LocalAI Domains: {r.status_code}')
except Exception as e:
    print(f'❌ LocalAI Domains: {e}')

# Test 3: LocalAI Chat
total += 1
try:
    payload = {'model': 'general', 'messages': [{'role': 'user', 'content': 'Say hello'}], 'max_tokens': 20}
    r = requests.post(f'{localai}/v1/chat/completions', json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
        print(f'✅ LocalAI Chat: Success')
        print(f'   Response: {content[:60]}...')
        passed += 1
    else:
        print(f'❌ LocalAI Chat: {r.status_code}')
except Exception as e:
    print(f'❌ LocalAI Chat: {e}')

# Test 4: LocalAI Embeddings
total += 1
try:
    payload = {'model': '0x3579-VectorProcessingAgent', 'input': ['test embedding']}
    r = requests.post(f'{localai}/v1/embeddings', json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        embedding = data.get('data', [{}])[0].get('embedding', [])
        print(f'✅ LocalAI Embeddings: Success (dim: {len(embedding)})')
        passed += 1
    else:
        print(f'❌ LocalAI Embeddings: {r.status_code}')
except Exception as e:
    print(f'❌ LocalAI Embeddings: {e}')

# Test 5: DeepAgents Health
total += 1
try:
    r = requests.get('http://localhost:9004/healthz', timeout=5)
    if r.status_code == 200:
        print(f'✅ DeepAgents Health: {r.status_code}')
        passed += 1
    else:
        print(f'❌ DeepAgents Health: {r.status_code}')
except Exception as e:
    print(f'❌ DeepAgents Health: {e}')

# Test 6: Search-inference
total += 1
try:
    r = requests.get('http://search-inference:8090/health', timeout=5)
    if r.status_code == 200:
        print(f'✅ Search-inference Health: {r.status_code}')
        passed += 1
    else:
        print(f'❌ Search-inference Health: {r.status_code}')
except Exception as e:
    print(f'❌ Search-inference Health: {e}')

# Test 7: Transformers
total += 1
try:
    r = requests.get('http://transformers-service:9090/health', timeout=5)
    if r.status_code == 200:
        print(f'✅ Transformers Health: {r.status_code}')
        passed += 1
    else:
        print(f'❌ Transformers Health: {r.status_code}')
except Exception as e:
    print(f'❌ Transformers Health: {e}')

# Test 8: Search Embeddings
total += 1
try:
    payload = {'text': 'test search embedding'}
    r = requests.post('http://search-inference:8090/v1/embed', json=payload, timeout=30)
    if r.status_code == 200:
        data = r.json()
        embedding = data.get('embedding', [])
        print(f'✅ Search Embeddings: Success (dim: {len(embedding)})')
        passed += 1
    else:
        print(f'❌ Search Embeddings: {r.status_code}')
except Exception as e:
    print(f'❌ Search Embeddings: {e}')

print()
print('=' * 60)
print(f'Summary: {passed}/{total} tests passed')
print('=' * 60)

sys.exit(0 if passed == total else 1)
PYTHON_SCRIPT

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed"
fi
exit $EXIT_CODE

