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

## Training Data

The `training/` directory contains training datasets and process artifacts for model training:

- **`training/sgmi/`**: SGMI (SAP Global Manufacturing Intelligence) training data
  - Hive DDL files for database schema extraction
  - Pipeline metamodel definitions (Control-M, Hive, JSON)
  - Execution scripts for process understanding
  - Annotated JSON dataset (`JSON_with_changes.json`) for process learning
  - See `training/sgmi/README.md` for detailed documentation

This data is used to train process understanding models that can extract schemas, understand workflows, and learn from Control-M job definitions.

## Stage 3: Search and Graph Services

The `stage3/search/` folder contains the Layer 4 search microservices, and `stage3/graph/` holds the GPU-accelerated graph components. Treat them like read-only exports—each directory keeps its original README and build scripts. Deploy them to your GPU infrastructure as separate services or integrate them into CI as needed.

## GPU Quickstart (Brev)

The repo ships with starter Docker assets so you can test the graph runtime and LocalAI on a Brev GPU workspace:

```bash
# 1. Clone the repo and (optionally) restore heavy model assets (Stage 2)
git clone https://github.com/plturrell/aModels.git
cd aModels
# Optional if you need local models immediately:
python3 tools/download_model_assets.py --tag weights-v1 --repo plturrell/aModels --output-dir .

# 2. Build the containers (graph + LocalAI + search services + Elasticsearch + training shell)
docker compose -f docker/brev/docker-compose.yml build

# 3. Launch them – Brev automatically exposes the GPU to the containers
docker compose -f docker/brev/docker-compose.yml up

# The graph API listens on :8080, LocalAI VaultGemma on :8081,
# Elasticsearch on :9200, search inference on :8090,
# the Python helper service on :8091, and the trainer container
# idles so you can exec into it.

# 4. To run training jobs inside the prepared CUDA/Go/Python environment:
#    (in a new terminal)
# docker compose -f docker/brev/docker-compose.yml exec trainer bash
# python scripts/train_relational_transformer.py --config configs/rt.yaml --mode pretrain ...

### Optional: fetch Gemma weights from Kaggle

Instead of (or in addition to) the GitHub release artifacts you can pull Gemma GGUF weights straight
from Kaggle. Save a new API token in your Kaggle account, export the credentials, and run:

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_kaggle_key"
./tools/fetch_kaggle_gemma.sh                # downloads to ./models by default
```

The script will place `gemma-2b-gguf.tar.gz` in the models directory so the LocalAI container can
mount it.
```

Notes:
- Elasticsearch may require a smaller heap on small GPUs/hosts. You can export `ES_JAVA_OPTS="-Xms512m -Xmx512m"` before `docker compose up` or adjust it in `docker/brev/docker-compose.yml`.
- The search service uses SQLite by default and Redis cache is optional; if no Redis is configured it falls back to in-memory cache.
- The AgentSDK catalog/watch features are disabled in this standalone repo.

## GPU Compose (gateway/local GPU services)

To run the lightweight gateway stack with a GPU-accelerated LocalAI container:

```bash
cd docker
# Base services + GPU override (enables the LocalAI CUDA service)
docker compose -f compose.yml -f compose.gpu.yml --profile gpu up --build
```
