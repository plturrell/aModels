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
