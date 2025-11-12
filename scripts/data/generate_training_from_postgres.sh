#!/usr/bin/env bash
# Generate training data from Postgres SGMI data
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

echo "=== Generating Training Data from Postgres ==="
echo ""

# Check if Postgres has data
NODE_COUNT=$(docker exec postgres psql -U postgres -d amodels -t -c "SELECT COUNT(*) FROM glean_nodes;" 2>/dev/null | tr -d '[:space:]')
if [[ "$NODE_COUNT" == "0" ]]; then
    echo "❌ No data in Postgres. Please load SGMI data first."
    exit 1
fi

echo "✅ Found $NODE_COUNT nodes in Postgres"
echo ""

# Extract service URL
EXTRACT_URL="${EXTRACT_URL:-http://localhost:8082}"
if [[ "$EXTRACT_URL" == "http://localhost:8082" ]]; then
    # Try external IP if localhost doesn't work
    EXTRACT_URL="http://54.196.0.75:8082"
fi

# Generate training data
echo "Generating training data via extract service..."
echo "URL: $EXTRACT_URL/generate/training"
echo ""

# Generate table extracts
echo "1. Generating table extracts..."
TABLE_RESPONSE=$(curl -s -X POST "$EXTRACT_URL/generate/training" \
    -H "Content-Type: application/json" \
    -d '{
        "mode": "table",
        "table_options": {
            "project_id": "sgmi-full"
        }
    }')

if echo "$TABLE_RESPONSE" | grep -q '"success":true'; then
    echo "✅ Table extracts generated"
    echo "$TABLE_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"   Manifest: {d.get('manifest', 'N/A')}\"); print(f\"   Files: {len(d.get('files', []))}\")"
else
    echo "⚠️  Table extract generation failed or returned unexpected response"
    echo "$TABLE_RESPONSE" | head -5
fi
echo ""

# Generate document extracts
echo "2. Generating document extracts..."
DOC_RESPONSE=$(curl -s -X POST "$EXTRACT_URL/generate/training" \
    -H "Content-Type: application/json" \
    -d '{
        "mode": "document",
        "document_options": {
            "project_id": "sgmi-full"
        }
    }')

if echo "$DOC_RESPONSE" | grep -q '"success":true'; then
    echo "✅ Document extracts generated"
    echo "$DOC_RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"   Manifest: {d.get('manifest', 'N/A')}\"); print(f\"   Files: {len(d.get('files', []))}\")"
else
    echo "⚠️  Document extract generation failed or returned unexpected response"
    echo "$DOC_RESPONSE" | head -5
fi
echo ""

# Check output directory
TRAINING_DIR="${REPO_ROOT}/data/training/extracts"
if [[ -d "$TRAINING_DIR" ]]; then
    echo "3. Training data directory:"
    echo "   $TRAINING_DIR"
    echo ""
    echo "   Contents:"
    ls -lh "$TRAINING_DIR" | head -10
    echo ""
    
    # Count files
    FILE_COUNT=$(find "$TRAINING_DIR" -type f | wc -l)
    echo "   Total files: $FILE_COUNT"
else
    echo "⚠️  Training directory not found: $TRAINING_DIR"
fi

echo ""
echo "=== Complete ==="
echo ""
echo "Next steps:"
echo "1. Review training data in: $TRAINING_DIR"
echo "2. Use training data for Relational Transformer training"
echo "3. See docs/POSTGRES_QUERIES.md for querying Postgres data"

