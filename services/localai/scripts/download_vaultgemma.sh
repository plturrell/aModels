#!/bin/bash
set -e

# Download VaultGemma 1B model weights from HuggingFace
# This script uses huggingface-cli to download the safetensors file

MODEL_DIR="/home/aModels/models/vaultgemm/vaultgemma-transformers-1b-v1"
HF_MODEL="google/vaultgemma-1b"  # HuggingFace model identifier

echo "📥 Downloading VaultGemma 1B model weights..."
echo "   Target: $HF_MODEL"
echo "   Destination: $MODEL_DIR"
echo ""

# Check if directory exists
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if already downloaded
if [ -f "$MODEL_DIR/model.safetensors" ]; then
    SIZE=$(du -h "$MODEL_DIR/model.safetensors" | cut -f1)
    echo "✅ model.safetensors already exists ($SIZE)"
    echo "   To re-download, delete it first: rm $MODEL_DIR/model.safetensors"
    exit 0
fi

# Install huggingface-hub if needed
if ! python3 -c "import huggingface_hub" 2>/dev/null; then
    echo "📦 Installing huggingface-hub..."
    pip3 install --quiet huggingface-hub
fi

cd "$MODEL_DIR"

echo "🔍 Attempting to download from HuggingFace..."
echo ""

# Try downloading with huggingface_hub Python library
python3 << EOF
import os
from huggingface_hub import hf_hub_download, snapshot_download
import sys

model_id = "$HF_MODEL"
target_dir = "$MODEL_DIR"

print(f"📥 Downloading {model_id}...")
print("")

try:
    # First, try to download just the safetensors file
    try:
        safetensors_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded: model.safetensors")
        sys.exit(0)
    except Exception as e:
        print(f"⚠️  Single file download failed: {e}")
        print("   Trying sharded version...")
        
        # Try downloading the index file first
        try:
            index_path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            print(f"✅ Downloaded: model.safetensors.index.json")
            
            # Read index to find shard files
            import json
            with open(index_path, 'r') as f:
                index = json.load(f)
            
            weight_map = index.get('weight_map', {})
            shard_files = set(weight_map.values())
            
            print(f"📦 Found {len(shard_files)} shard file(s)")
            
            # Download each shard
            for shard_file in shard_files:
                print(f"   Downloading: {shard_file}...")
                hf_hub_download(
                    repo_id=model_id,
                    filename=shard_file,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
                print(f"   ✅ {shard_file}")
            
            print("")
            print("✅ All shards downloaded successfully!")
            sys.exit(0)
            
        except Exception as e2:
            print(f"❌ Sharded download also failed: {e2}")
            print("")
            print("💡 Alternative: Try downloading manually from:")
            print(f"   https://huggingface.co/{model_id}")
            print("")
            print("   Or try a different model identifier:")
            print("   - google/gemma-1b")
            print("   - google/vaultgemma-1b-it")
            sys.exit(1)
            
except Exception as e:
    print(f"❌ Download failed: {e}")
    print("")
    print("💡 Troubleshooting:")
    print("   1. Check internet connection")
    print("   2. Verify model ID: $HF_MODEL")
    print("   3. Try: pip3 install --upgrade huggingface-hub")
    print("   4. Manual download: https://huggingface.co/$HF_MODEL")
    sys.exit(1)
EOF

DOWNLOAD_RESULT=$?

if [ $DOWNLOAD_RESULT -eq 0 ]; then
    echo ""
    echo "✅ Model weights downloaded successfully!"
    echo ""
    echo "📊 Files in $MODEL_DIR:"
    ls -lh "$MODEL_DIR"/model.safetensors* 2>/dev/null | head -10
    echo ""
    echo "🚀 Next step: Restart the LocalAI server"
    echo "   cd /home/aModels/services/localai && ./start-production.sh"
else
    echo ""
    echo "❌ Download failed. See errors above."
    exit 1
fi

