#!/bin/bash
set -e

cd /workspace || cd /home/aModels || true

# Install Python dependencies for tests
if [ -f testing/requirements.txt ]; then
  python3 -m pip install -q -r testing/requirements.txt || pip3 install -q -r testing/requirements.txt
fi

# Export PYTHONPATH for our services
export PYTHONPATH="/workspace/services/training:/workspace/services/localai:/workspace/services/extract:${PYTHONPATH}"

# Persist for login shells
if ! grep -q "PYTHONPATH=/workspace/services" ~/.bashrc 2>/dev/null; then
  echo "export PYTHONPATH=/workspace/services/training:/workspace/services/localai:/workspace/services/extract:\$PYTHONPATH" >> ~/.bashrc
fi

echo "✅ training-shell bootstrap complete"
