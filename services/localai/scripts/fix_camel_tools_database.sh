#!/bin/bash
# Fix camel-tools morphology database download
# This script manually downloads the database if automatic download fails

set -e

DATA_DIR="$HOME/.camel_tools/data/morphology_db"
TARGET_DIR="$DATA_DIR/calima-msa-r13"

echo "🔧 Fixing camel-tools Morphology Database"
echo "=" * 80

# Check if already exists
if [ -f "$TARGET_DIR/morphology.db" ]; then
    size=$(du -sh "$TARGET_DIR/morphology.db" | cut -f1)
    echo "✅ Database already exists ($size)"
    exit 0
fi

echo "📥 Database not found, downloading..."
mkdir -p "$DATA_DIR"

# Method 1: Clone from GitHub
TEMP_DIR="/tmp/camel_tools_data"
if [ ! -d "$TEMP_DIR" ]; then
    echo "   Cloning camel-tools-data repository..."
    git clone --depth 1 https://github.com/CAMeL-Lab/camel_tools_data.git "$TEMP_DIR" 2>&1 | tail -3
fi

# Copy database
if [ -d "$TEMP_DIR/morphology_db/calima-msa-r13" ]; then
    echo "   Copying database..."
    cp -r "$TEMP_DIR/morphology_db/calima-msa-r13" "$DATA_DIR/" 2>&1
    
    if [ -f "$TARGET_DIR/morphology.db" ]; then
        size=$(du -sh "$TARGET_DIR/morphology.db" | cut -f1)
        echo "✅ Database installed successfully! ($size)"
    else
        echo "⚠️  Copy completed but file not found"
    fi
else
    echo "⚠️  Source directory not found in repository"
    echo "   Repository structure may have changed"
fi

# Verify
if [ -f "$TARGET_DIR/morphology.db" ]; then
    echo ""
    echo "🧪 Testing database..."
    python3 << PYEOF
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer

try:
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    result = analyzer.analyze("الكتاب")
    print(f"✅ Database working! Analyzed 'الكتاب': {len(result)} analyses")
except Exception as e:
    print(f"⚠️  Database test failed: {e}")
PYEOF
else
    echo ""
    echo "❌ Database installation failed"
    echo "   Manual steps:"
    echo "   1. Visit: https://github.com/CAMeL-Lab/camel_tools_data"
    echo "   2. Download morphology_db/calima-msa-r13/"
    echo "   3. Copy to: $TARGET_DIR"
fi

