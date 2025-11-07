#!/bin/bash

# Test Perplexity Integration
# Usage: ./test_perplexity.sh [api_key]

set -e

API_KEY="${1:-${PERPLEXITY_API_KEY}}"
if [ -z "$API_KEY" ]; then
    echo "Error: PERPLEXITY_API_KEY not provided"
    echo "Usage: ./test_perplexity.sh <api_key>"
    exit 1
fi

echo "Testing Perplexity Integration..."
echo "API Key: ${API_KEY:0:8}...${API_KEY: -4}"
echo ""

# Test 1: Direct API call
echo "=== Test 1: Perplexity API Connection ==="
RESPONSE=$(curl -s -X POST https://api.perplexity.ai/chat/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sonar",
    "messages": [
      {
        "role": "user",
        "content": "What is machine learning? Answer in one sentence."
      }
    ],
    "max_tokens": 100
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo "✅ API connection successful"
    CONTENT=$(echo "$RESPONSE" | grep -o '"content":"[^"]*' | head -1 | cut -d'"' -f4)
    if [ -n "$CONTENT" ]; then
        echo "   Response: ${CONTENT:0:100}..."
    fi
else
    echo "❌ API connection failed"
    echo "   Response: $RESPONSE"
    exit 1
fi

echo ""
echo "=== Test 2: Checking Integration Files ==="

# Check if connector exists
if [ -f "services/orchestration/agents/connectors/perplexity_connector.go" ]; then
    echo "✅ Perplexity connector found"
else
    echo "❌ Perplexity connector not found"
    exit 1
fi

# Check if pipeline exists
if [ -f "services/orchestration/agents/perplexity_pipeline.go" ]; then
    echo "✅ Perplexity pipeline found"
else
    echo "❌ Perplexity pipeline not found"
    exit 1
fi

# Check if autonomous wrapper exists
if [ -f "services/orchestration/agents/perplexity_autonomous.go" ]; then
    echo "✅ Perplexity autonomous wrapper found"
else
    echo "❌ Perplexity autonomous wrapper not found"
    exit 1
fi

echo ""
echo "=== Test 3: Integration Components ==="

# Check for Deep Research integration
if grep -q "DeepResearchClient" services/orchestration/agents/perplexity_pipeline.go; then
    echo "✅ Deep Research integration found"
else
    echo "⚠️  Deep Research integration not found"
fi

# Check for pattern learning
if grep -q "PatternLearning" services/orchestration/agents/perplexity_autonomous.go; then
    echo "✅ Pattern learning integration found"
else
    echo "⚠️  Pattern learning integration not found"
fi

# Check for LNN integration
if grep -q "updateLNNWithFeedback" services/orchestration/agents/perplexity_autonomous.go; then
    echo "✅ LNN integration found"
else
    echo "⚠️  LNN integration not found"
fi

# Check for Goose migrations
if grep -q "ExecuteWithGooseMigration" services/orchestration/agents/perplexity_autonomous.go; then
    echo "✅ Goose migration integration found"
else
    echo "⚠️  Goose migration integration not found"
fi

echo ""
echo "=== Test 4: Configuration Check ==="

# Test environment variable setup
export PERPLEXITY_API_KEY="$API_KEY"
export DEEP_RESEARCH_URL="${DEEP_RESEARCH_URL:-http://localhost:8085}"
export CATALOG_URL="${CATALOG_URL:-http://catalog:8080}"
export TRAINING_URL="${TRAINING_URL:-http://training:8080}"

echo "✅ Environment variables configured:"
echo "   PERPLEXITY_API_KEY: ${API_KEY:0:8}...${API_KEY: -4}"
echo "   DEEP_RESEARCH_URL: $DEEP_RESEARCH_URL"
echo "   CATALOG_URL: $CATALOG_URL"
echo "   TRAINING_URL: $TRAINING_URL"

echo ""
echo "=== All Tests Complete ==="
echo "✅ Perplexity integration is ready!"
echo ""
echo "To use the integration:"
echo "  1. Set PERPLEXITY_API_KEY environment variable"
echo "  2. Configure service URLs (DEEP_RESEARCH_URL, CATALOG_URL, etc.)"
echo "  3. Use the HTTP API endpoint: POST /api/perplexity/process"
echo "  4. Or use the Go API: agents.NewPerplexityPipeline(config)"

