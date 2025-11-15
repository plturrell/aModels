#!/bin/bash
set -e

# Script to download NLLB-200 and M2M100 translation models from HuggingFace
# These models are used for English ↔ Arabic translation

MODELS_BASE="${MODELS_BASE:-/home/aModels/models/arabic_models}"

echo "📥 Downloading NLLB-200 and M2M100 translation models..."
echo "   Models directory: $MODELS_BASE"
echo ""

# Check if models directory exists
if [ ! -d "$MODELS_BASE" ]; then
    echo "❌ Models directory not found: $MODELS_BASE"
    echo "   Creating directory..."
    mkdir -p "$MODELS_BASE"
fi

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-hub..."
    pip install --quiet huggingface-hub
fi

# Function to download a model
download_model() {
    local model_name=$1
    local local_dir=$2
    local model_display_name=$3
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📥 Downloading $model_display_name"
    echo "   HuggingFace: $model_name"
    echo "   Local path: $local_dir"
    echo ""
    
    # Check if model already exists
    if [ -d "$local_dir" ] && [ -f "$local_dir/config.json" ]; then
        echo "✅ Model already exists at $local_dir"
        echo "   To re-download, delete it first: rm -rf $local_dir"
        return 0
    fi
    
    # Create directory
    mkdir -p "$local_dir"
    cd "$local_dir"
    
    echo "🔍 Downloading from HuggingFace..."
    if huggingface-cli download "$model_name" \
        --local-dir . \
        --local-dir-use-symlinks False \
        --exclude "*.msgpack" \
        --exclude "*.h5" 2>&1 | tee /tmp/hf_download.log; then
        echo "✅ Successfully downloaded $model_display_name"
    else
        echo "❌ Failed to download $model_display_name"
        echo "   Check /tmp/hf_download.log for details"
        return 1
    fi
    
    cd - > /dev/null
}

# Download NLLB-200 models
echo "🎯 Primary MT Models (NLLB-200)"
echo ""

# NLLB-200 Distilled 600M (lighter, good quality)
download_model \
    "facebook/nllb-200-distilled-600M" \
    "$MODELS_BASE/nllb-200-distilled-600M" \
    "NLLB-200 Distilled 600M"

# NLLB-200 1.3B (heavier, better quality - recommended for T4)
download_model \
    "facebook/nllb-200-1.3B" \
    "$MODELS_BASE/nllb-200-1.3B" \
    "NLLB-200 1.3B"

# Optional: M2M100 models (lighter backup)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎯 Optional Backup MT Models (M2M100)"
echo "   These are lighter alternatives for high-throughput scenarios"
echo ""

read -p "Download M2M100 models? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # M2M100 418M
    download_model \
        "facebook/m2m100_418M" \
        "$MODELS_BASE/m2m100-418M" \
        "M2M100 418M"
    
    # M2M100 1.2B
    download_model \
        "facebook/m2m100_1.2B" \
        "$MODELS_BASE/m2m100-1.2B" \
        "M2M100 1.2B"
else
    echo "⏭️  Skipping M2M100 models"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Download complete!"
echo ""
echo "📋 Summary:"
echo "   Models downloaded to: $MODELS_BASE"
echo ""
echo "🎯 Next steps:"
echo "   1. Restart the transformers-service container"
echo "   2. Verify models are accessible:"
echo "      docker exec transformers-service ls -lh /models/nllb-200-1.3B"
echo "   3. Test translation endpoint:"
echo "      curl -X POST http://localhost:9090/v1/translate \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\": \"nllb-200-1.3B\", \"text\": \"Hello world\"}'"
echo ""

