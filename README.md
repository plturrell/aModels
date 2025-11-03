# aModels - Training Stack

This repository contains the AgenticAI Layer 4 training project, exported from the mono-repo so it can be managed independently on GPU infrastructure.

Key entry points:
- `scripts/train_relational_transformer.py`
- `scripts/eval_relational_transformer.py`
- `scripts/profile_rt_inference.py`

Refer to `docs/` inside this tree for detailed setup and operations guidance.

## Stage 2: Models and LocalAI

The `models/` directory contains lightweight metadata/checkpoints (large weight files ignored by `.gitignore`).
The `localai/` directory mirrors the Go-based inference server for hosting quantized models.

### Publishing Weights via GitHub Releases

GitHub enforces a 2 GB limit per release asset, so the actual model binaries need to live outside the repo. Use the helper scripts to split and download them:

1. From a machine that has the original weights (e.g. `agenticAiETH_layer4_Models/`), run:
   ```bash
   python tools/package_model_assets.py --source-root /path/to/agenticAiETH_layer4_Models --output-dir artifacts/model-release
   ```
   This produces `manifest.json` plus `*.part###` chunks under `artifacts/model-release/`.
2. Create a GitHub release (for example `weights-v1`) and upload **all** generated `.part###` files and the `manifest.json`.
3. On any machine that needs the weights, restore them with:
   ```bash
   python tools/download_model_assets.py --tag weights-v1 --repo plturrell/aModels --output-dir .
   ```
   The script reconstructs the `.gguf` files and unpacks transformer checkpoints back into `models/`.

> **Note:** Model weights and tokenizers are stored via GitHub Releases using the chunked artifacts above. Run `git lfs install` only if you plan to host sub-2 GB files via LFS; larger files must be shipped through releases.
