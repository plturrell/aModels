#!/bin/bash
set -e

# Script to download VaultGemma model weights from HuggingFace
# This script downloads the safetensors file(s) needed for inference

MODEL_DIR="/home/aModels/models/vaultgemm/vaultgemma-transformers-1b-v1"
MODEL_NAME="google/vaultgemma-1b"  # Default HuggingFace model name

echo "📥 Downloading VaultGemma model weights..."
echo "   Model directory: $MODEL_DIR"
echo ""

# Check if model directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if weights already exist
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    echo "✅ model.safetensors already exists"
    echo "   To re-download, delete it first: rm $MODEL_DIR/model.safetensors"
    exit 0
fi

# Check for huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo "📦 Installing huggingface-hub..."
    pip install --quiet huggingface-hub
fi

echo "🔍 Searching for model on HuggingFace..."
echo ""

# Try to download using huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    echo "📥 Downloading using huggingface-cli..."
    cd "$MODEL_DIR"
    
    # Try common VaultGemma model names
    for model_name in "google/vaultgemma-1b" "google/gemma-1b" "google/vaultgemma-transformers-1b-v1"; do
        echo "   Trying: $model_name"
        if huggingface-cli download "$model_name" \
            --local-dir . \
            --local-dir-use-symlinks False \
            --include "model.safetensors*" \
            --exclude "*.bin" \
            --exclude "*.pt" \
            --exclude "*.pth" 2>&1 | grep -q "model.safetensors"; then
            echo "✅ Successfully downloaded from: $model_name"
            break
        fi
    done
else
    echo "⚠️  huggingface-cli not available"
    echo "   Install with: pip install huggingface-hub"
    echo ""
    echo "   Or manually download from:"
    echo "   https://huggingface.co/google/vaultgemma-1b"
    echo "   and place model.safetensors in: $MODEL_DIR"
    exit 1
fi

# Verify download
if [ -f "$MODEL_DIR/model.safetensors" ] || [ -f "$MODEL_DIR/model.safetensors.index.json" ]; then
    echo ""
    echo "✅ Model weights downloaded successfully!"
    echo ""
    echo "📊 Files:"
    ls -lh "$MODEL_DIR"/model.safetensors* 2>/dev/null || echo "   (checking for sharded files...)"
    echo ""
    echo "🚀 You can now restart the LocalAI server:"
    echo "   cd /home/aModels/services/localai && ./start-production.sh"
else
    echo ""
    echo "❌ Download failed or incomplete"
    echo ""
    echo "💡 Manual download options:"
    echo "   1. Visit https://huggingface.co/models?search=vaultgemma"
    echo "   2. Find the correct model repository"
    echo "   3. Download model.safetensors or model.safetensors.index.json"
    echo "   4. Place in: $MODEL_DIR"
    exit 1
fi

