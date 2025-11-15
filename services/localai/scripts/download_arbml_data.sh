#!/bin/bash
# Manual ARBML morphology database download script
# This downloads the database from camel-tools-data repository

set -e

DATA_DIR="$HOME/.camel_tools/data/morphology_db"
mkdir -p "$DATA_DIR"

echo "📥 Downloading ARBML Morphology Database..."
echo "Target: $DATA_DIR"

# Method 1: Try using camel-tools Python API
python3 << PYEOF
from camel_tools.morphology.database import MorphologyDB
try:
    db = MorphologyDB.builtin_db()
    print("✅ Database loaded successfully!")
except Exception as e:
    print(f"⚠️  Auto-download failed: {e}")
    print("   Will need manual download")
PYEOF

# Method 2: Clone repository and copy data
if [ ! -f "$DATA_DIR/calima-msa-r13/morphology.db" ]; then
    echo ""
    echo "Attempting manual download from GitHub..."
    TEMP_DIR="/tmp/camel_tools_data"
    
    if [ ! -d "$TEMP_DIR" ]; then
        git clone --depth 1 https://github.com/CAMeL-Lab/camel_tools_data.git "$TEMP_DIR" 2>/dev/null || echo "Git clone failed, manual download needed"
    fi
    
    if [ -d "$TEMP_DIR/morphology_db" ]; then
        echo "Copying morphology database..."
        cp -r "$TEMP_DIR/morphology_db"/* "$DATA_DIR/" 2>/dev/null && echo "✅ Database copied" || echo "⚠️  Copy failed"
    fi
fi

# Verify
if [ -f "$DATA_DIR/calima-msa-r13/morphology.db" ]; then
    echo "✅ ARBML database is ready!"
else
    echo "⚠️  Database not found. Manual download required:"
    echo "   Visit: https://github.com/CAMeL-Lab/camel_tools_data"
    echo "   Or: The database will download automatically on first use"
fi
