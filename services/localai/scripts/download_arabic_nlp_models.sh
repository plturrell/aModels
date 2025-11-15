#!/bin/bash
set -e

# Script to download Arabic NLP models and tools for TOON generation
# These models are used for Arabic-to-English translation with TOON enhancement

MODELS_BASE="${MODELS_BASE:-/home/aModels/models/arabic_models}"
ARABIC_MODELS_DIR="${MODELS_BASE}"

echo "📥 Downloading Arabic NLP models and tools for TOON..."
echo "   Models directory: $ARABIC_MODELS_DIR"
echo ""

# Create directory if it doesn't exist
mkdir -p "$ARABIC_MODELS_DIR"

# Function to download from HuggingFace
download_hf_model() {
    local model_name=$1
    local output_dir=$2
    local model_id=$3
    
    if [ -d "$output_dir" ]; then
        echo "✅ $model_name already exists at $output_dir"
        return 0
    fi
    
    echo "📥 Downloading $model_name..."
    echo "   Model ID: $model_id"
    echo "   Output: $output_dir"
    
    python3 << EOF
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="$model_id",
        local_dir="$output_dir",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ Successfully downloaded {model_name}")
except Exception as e:
    print(f"❌ Failed to download {model_name}: {e}")
    exit(1)
EOF
}

# 1. Kuwain 1.5B - Arabic language model
# Note: Kuwain might refer to a specific Arabic model or we can use alternatives
# For now, we'll use a well-known Arabic language model as placeholder
# Options: arabert-base, camelbert-base, or other Arabic BERT models
KUWAIN_DIR="$ARABIC_MODELS_DIR/kuwain-1.5B"
KUWAIN_MODEL_ID="${KUWAIN_MODEL_ID:-aubmindlab/bert-base-arabertv02}"  # Placeholder - update when Kuwain model ID is found

if [ -d "$KUWAIN_DIR" ] && [ -f "$KUWAIN_DIR/config.json" ]; then
    echo "✅ Kuwain model already exists at $KUWAIN_DIR"
else
    echo "📥 Downloading Arabic language model (Kuwain placeholder)..."
    echo "   Using: $KUWAIN_MODEL_ID"
    echo "   Output: $KUWAIN_DIR"
    
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$KUWAIN_MODEL_ID" \
            --local-dir "$KUWAIN_DIR" \
            --local-dir-use-symlinks False \
            --exclude "*.msgpack" \
            --exclude "*.h5" 2>&1 | tail -10
        if [ -f "$KUWAIN_DIR/config.json" ]; then
            echo "✅ Successfully downloaded Arabic language model"
        else
            echo "⚠️  Download may have failed, but continuing..."
        fi
    else
        echo "⚠️  huggingface-cli not found, trying Python download..."
        python3 << EOF
from huggingface_hub import snapshot_download
import os

try:
    snapshot_download(
        repo_id="$KUWAIN_MODEL_ID",
        local_dir="$KUWAIN_DIR",
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("✅ Successfully downloaded Arabic language model")
except Exception as e:
    print(f"⚠️  Download failed: {e}")
    print("   You can download manually or update KUWAIN_MODEL_ID")
EOF
    fi
fi

# 2. QADI - Dialect classification model
# QADI is available via camel-tools DialectIdPredictor (no separate download needed)
# However, we can also download a standalone model if needed
QADI_DIR="$ARABIC_MODELS_DIR/qadi"

echo "📦 QADI Dialect Classification:"
if python3 -c "from camel_tools.dialectid import DialectIdPredictor" 2>/dev/null; then
    echo "✅ QADI available via camel-tools (DialectIdPredictor)"
    echo "   No separate model download needed - uses pretrained() method"
    
    # Try to initialize to download pretrained model
    echo "   Initializing QADI predictor (will download pretrained model if needed)..."
    python3 << EOF
try:
    from camel_tools.dialectid import DialectIdPredictor
    predictor = DialectIdPredictor.pretrained()
    print("✅ QADI predictor initialized successfully")
except Exception as e:
    print(f"⚠️  QADI initialization: {e}")
    print("   This is normal if models are downloaded on first use")
EOF
else
    echo "⚠️  QADI not available - camel-tools not properly installed"
    echo "   Install with: pip install camel-tools"
fi

# 3. ARBML Tools (camel-tools)
# These are Python packages, not model files
echo ""
echo "📦 Installing ARBML tools (camel-tools)..."
echo ""

# Check if camel-tools is installed
if python3 -c "import camel_tools" 2>/dev/null; then
    echo "✅ camel-tools is already installed"
    
    # Test ARBML components
    echo "   Testing ARBML components..."
    python3 << EOF
try:
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    from camel_tools.tokenizers.word import simple_word_tokenize
    
    db = MorphologyDB.builtin_db()
    analyzer = Analyzer(db)
    print("✅ ARBML MorphologyDB and Analyzer working")
    
    # Test tokenization
    tokens = simple_word_tokenize("مرحبا")
    print(f"✅ ARBML tokenization working: {tokens}")
except Exception as e:
    print(f"⚠️  ARBML test failed: {e}")
EOF
else
    echo "📥 Installing camel-tools..."
    pip3 install camel-tools --quiet 2>&1 | tail -3
    if python3 -c "import camel_tools" 2>/dev/null; then
        echo "✅ camel-tools installed successfully"
    else
        echo "❌ Failed to install camel-tools"
        echo "   Try: pip install camel-tools"
        echo "   Or: pip install camel-tools[all]  # for full features"
    fi
fi

# 4. Tarjama-25 dataset (for training/validation)
# This is a parallel corpus, typically available from research repositories
TARJAMA_DIR="$ARABIC_MODELS_DIR/tarjama-25"
if [ ! -d "$TARJAMA_DIR" ]; then
    echo ""
    echo "⚠️  Tarjama-25 dataset location not yet determined"
    echo "   Please download manually or update this script with the correct source"
    echo "   Expected location: $TARJAMA_DIR"
    echo "   Tarjama-25 is typically available from research repositories"
else
    echo "✅ Tarjama-25 found at $TARJAMA_DIR"
fi

# 5. Mutarjim model (for quality benchmarking)
# Note: Mutarjim might be available from HuggingFace or other sources
MUTARJIM_DIR="$ARABIC_MODELS_DIR/mutarjim"
if [ ! -d "$MUTARJIM_DIR" ]; then
    echo ""
    echo "⚠️  Mutarjim model location not yet determined"
    echo "   Please download manually or update this script with the correct model ID"
    echo "   Expected location: $MUTARJIM_DIR"
else
    echo "✅ Mutarjim found at $MUTARJIM_DIR"
fi

echo ""
echo "📋 Summary:"
echo "   - Models directory: $ARABIC_MODELS_DIR"
echo "   - Kuwain 1.5B: $([ -d "$KUWAIN_DIR" ] && echo "✅ Found" || echo "❌ Not found")"
echo "   - QADI: $([ -d "$QADI_DIR" ] && echo "✅ Found" || echo "❌ Not found")"
echo "   - camel-tools: $(python3 -c 'import camel_tools; print("✅ Installed")' 2>/dev/null || echo "❌ Not installed")"
echo "   - Tarjama-25: $([ -d "$TARJAMA_DIR" ] && echo "✅ Found" || echo "❌ Not found")"
echo "   - Mutarjim: $([ -d "$MUTARJIM_DIR" ] && echo "✅ Found" || echo "❌ Not found")"
echo ""
echo "💡 Note: Some models need to be downloaded manually or from specific sources."
echo "   Update this script with correct model IDs/sources when available."
echo ""

