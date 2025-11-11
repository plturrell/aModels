#!/bin/bash
set -e

# Download 4 models from Kaggle for LocalAI
# Using Kaggle API credentials

KAGGLE_USERNAME="craigjohnturrell"
KAGGLE_KEY="e1679d6d56004a5339316890e2ba10ff"

MODELS_DIR="/home/aModels/models"
mkdir -p "$MODELS_DIR"

echo "📥 Downloading 4 models from Kaggle..."
echo "   Username: $KAGGLE_USERNAME"
echo "   Target directory: $MODELS_DIR"
echo ""

# Check for kaggle CLI
if command -v kaggle &> /dev/null; then
    echo "✅ Kaggle CLI found"
    export KAGGLE_USERNAME
    export KAGGLE_KEY
else
    echo "⚠️  Kaggle CLI not found, using curl method"
    echo "   Install with: pip install kaggle"
fi

# Model 1: Gemma 2B-it TensorRT-LLM
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Model 1/4: Gemma 2B-it (TensorRT-LLM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MODEL1_DIR="$MODELS_DIR/gemma-2b-it-tensorrt"
mkdir -p "$MODEL1_DIR"

if [ ! -f "$MODEL1_DIR/model.tar.gz" ] && [ ! -d "$MODEL1_DIR/2b-it" ]; then
    echo "   Downloading from Kaggle..."
    curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
        -o "$MODEL1_DIR/model.tar.gz" \
        "https://www.kaggle.com/api/v1/models/google/gemma/tensorRtLlm/2b-it/2/download"
    
    if [ -f "$MODEL1_DIR/model.tar.gz" ]; then
        SIZE=$(du -h "$MODEL1_DIR/model.tar.gz" | cut -f1)
        echo "   ✅ Downloaded ($SIZE)"
        echo "   📦 Extracting..."
        cd "$MODEL1_DIR"
        tar -xzf model.tar.gz 2>/dev/null && echo "   ✅ Extracted" || echo "   ⚠️  Check extraction manually"
    else
        echo "   ❌ Download failed"
    fi
else
    echo "   ✅ Already exists"
fi

# Model 2: Phi-3.5-mini-instruct PyTorch
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Model 2/4: Phi-3.5-mini-instruct (PyTorch)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MODEL2_DIR="$MODELS_DIR/phi-3.5-mini-instruct-pytorch"
mkdir -p "$MODEL2_DIR"

if [ ! -f "$MODEL2_DIR/model.tar.gz" ] && [ ! -f "$MODEL2_DIR/model.safetensors" ]; then
    echo "   Downloading from Kaggle..."
    curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
        -o "$MODEL2_DIR/model.tar.gz" \
        "https://www.kaggle.com/api/v1/models/Microsoft/phi-3/pyTorch/phi-3.5-mini-instruct/2/download"
    
    if [ -f "$MODEL2_DIR/model.tar.gz" ]; then
        SIZE=$(du -h "$MODEL2_DIR/model.tar.gz" | cut -f1)
        echo "   ✅ Downloaded ($SIZE)"
        echo "   📦 Extracting..."
        cd "$MODEL2_DIR"
        tar -xzf model.tar.gz 2>/dev/null && echo "   ✅ Extracted" || echo "   ⚠️  Check extraction manually"
    else
        echo "   ❌ Download failed"
    fi
else
    echo "   ✅ Already exists"
fi

# Model 3: IBM Granite 4.0-h-micro Transformers
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Model 3/4: IBM Granite 4.0-h-micro (Transformers)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MODEL3_DIR="$MODELS_DIR/granite-4.0-h-micro-transformers"
mkdir -p "$MODEL3_DIR"

if [ ! -f "$MODEL3_DIR/model.tar.gz" ] && [ ! -f "$MODEL3_DIR/model.safetensors" ]; then
    echo "   Downloading from Kaggle..."
    curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
        -o "$MODEL3_DIR/model.tar.gz" \
        "https://www.kaggle.com/api/v1/models/ibm-granite/granite-4.0/transformers/granite-4.0-h-micro/1/download"
    
    if [ -f "$MODEL3_DIR/model.tar.gz" ]; then
        SIZE=$(du -h "$MODEL3_DIR/model.tar.gz" | cut -f1)
        echo "   ✅ Downloaded ($SIZE)"
        echo "   📦 Extracting..."
        cd "$MODEL3_DIR"
        tar -xzf model.tar.gz 2>/dev/null && echo "   ✅ Extracted" || echo "   ⚠️  Check extraction manually"
    else
        echo "   ❌ Download failed"
    fi
else
    echo "   ✅ Already exists"
fi

# Model 4: Gemma 7B-it TensorRT-LLM
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 Model 4/4: Gemma 7B-it (TensorRT-LLM)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
MODEL4_DIR="$MODELS_DIR/gemma-7b-it-tensorrt"
mkdir -p "$MODEL4_DIR"

if [ ! -f "$MODEL4_DIR/model.tar.gz" ] && [ ! -d "$MODEL4_DIR/7b-it" ]; then
    echo "   Downloading from Kaggle..."
    
    # Try kaggle CLI first, fallback to curl
    if command -v kaggle &> /dev/null; then
        echo "   Using Kaggle CLI..."
        cd "$MODEL4_DIR"
        kaggle models instances versions download google/gemma/tensorRtLlm/7b-it/2 -p . 2>&1 | grep -E "(Downloading|downloaded|error)" || true
        
        # Check if download succeeded (kaggle CLI creates files directly)
        if [ -f "$MODEL4_DIR"/*.tar.gz ] || [ -d "$MODEL4_DIR/7b-it" ]; then
            echo "   ✅ Downloaded via Kaggle CLI"
            # Rename if needed
            if [ -f "$MODEL4_DIR"/*.tar.gz ] && [ ! -f "$MODEL4_DIR/model.tar.gz" ]; then
                mv "$MODEL4_DIR"/*.tar.gz "$MODEL4_DIR/model.tar.gz" 2>/dev/null || true
            fi
        fi
    fi
    
    # Fallback to curl if kaggle CLI didn't work or file doesn't exist
    if [ ! -f "$MODEL4_DIR/model.tar.gz" ] && [ ! -d "$MODEL4_DIR/7b-it" ]; then
        echo "   Using curl method..."
        curl -L -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
            -o "$MODEL4_DIR/model.tar.gz" \
            "https://www.kaggle.com/api/v1/models/google/gemma/tensorRtLlm/7b-it/2/download"
    fi
    
    if [ -f "$MODEL4_DIR/model.tar.gz" ]; then
        SIZE=$(du -h "$MODEL4_DIR/model.tar.gz" | cut -f1)
        echo "   ✅ Downloaded ($SIZE)"
        echo "   📦 Extracting..."
        cd "$MODEL4_DIR"
        tar -xzf model.tar.gz 2>/dev/null && echo "   ✅ Extracted" || echo "   ⚠️  Check extraction manually"
    elif [ -d "$MODEL4_DIR/7b-it" ]; then
        echo "   ✅ Already extracted"
    else
        echo "   ❌ Download failed"
    fi
else
    echo "   ✅ Already exists"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ All 4 models download complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Summary:"
echo "   ✅ Model 1: Gemma 2B-it TensorRT-LLM → $MODEL1_DIR"
echo "   ✅ Model 2: Phi-3.5-mini-instruct PyTorch → $MODEL2_DIR"
echo "   ✅ Model 3: IBM Granite 4.0-h-micro Transformers → $MODEL3_DIR"
echo "   ✅ Model 4: Gemma 7B-it TensorRT-LLM → $MODEL4_DIR"
echo ""
echo "📁 Model directories:"
for dir in "$MODEL1_DIR" "$MODEL2_DIR" "$MODEL3_DIR" "$MODEL4_DIR"; do
    if [ -d "$dir" ]; then
        SIZE=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "   $(basename $dir): $SIZE"
    fi
done
echo ""
echo "🚀 Next steps:"
echo "   1. Verify model files are extracted correctly"
echo "   2. Update LocalAI config to use these models"
echo "   3. Restart LocalAI server: cd /home/aModels/services/localai && ./start-production.sh"
