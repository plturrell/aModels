#!/bin/bash
set -e

# Deployment script for wiring models into aModels system

echo "🚀 Deploying Models into aModels System"
echo ""

# Check LocalAI is running
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1: Verify LocalAI Service"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if curl -s http://localhost:8080/v1/models > /dev/null 2>&1; then
    echo "✅ LocalAI is running"
    curl -s http://localhost:8080/v1/models | head -c 200
    echo ""
else
    echo "❌ LocalAI is not running"
    echo "   Starting LocalAI..."
    cd /home/aModels/services/localai
    ./start-production.sh
    sleep 5
fi

# Check Gateway integration
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2: Verify Gateway Integration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    echo "✅ Gateway is running"
    LOCALAI_STATUS=$(curl -s http://localhost:8000/healthz | grep -o '"localai":"[^"]*"' | cut -d'"' -f4)
    echo "   LocalAI status: $LOCALAI_STATUS"
else
    echo "⚠️  Gateway is not running"
    echo "   Start with: cd services/gateway && ./start.sh"
fi

# Test model inference
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3: Test Model Inference"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Testing VaultGemma model..."
RESPONSE=$(curl -s http://localhost:8080/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"vaultgemma","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}')

if echo "$RESPONSE" | grep -q "choices\|content"; then
    echo "✅ Model inference working"
    echo "$RESPONSE" | head -c 200
    echo ""
else
    echo "⚠️  Model inference may be in stub mode"
    echo "   Response: $RESPONSE"
fi

# Check environment variables
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 4: Environment Configuration"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "LOCALAI_URL: ${LOCALAI_URL:-http://localhost:8080}"
echo "GATEWAY_PORT: ${GATEWAY_PORT:-8000}"

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment Check Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Verify all services can reach LocalAI"
echo "  2. Update service configs with LOCALAI_URL"
echo "  3. Test end-to-end workflows"
echo "  4. Monitor model performance"


