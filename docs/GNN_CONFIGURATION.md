# GNN Configuration Guide

## Overview

The GNN (Graph Neural Network) modules are now integrated into the training pipeline (Priority 1: Integration complete). This guide explains how to configure and use them.

---

## Environment Variables

### Enable/Disable GNN

```bash
# Enable GNN modules (default: false)
ENABLE_GNN=true

# Enable specific GNN features (default: true when ENABLE_GNN=true)
ENABLE_GNN_EMBEDDINGS=true
ENABLE_GNN_CLASSIFICATION=true
ENABLE_GNN_LINK_PREDICTION=true

# Optional GNN features
ENABLE_GNN_ANOMALY_DETECTION=false
ENABLE_GNN_SCHEMA_MATCHING=false
```

### GNN Configuration

```bash
# Device configuration
GNN_DEVICE=auto          # Options: "auto", "cuda", "cpu" (default: "auto")

# Model hyperparameters
GNN_EMBEDDING_DIM=128    # Embedding dimension (default: 128)
GNN_HIDDEN_DIM=64        # Hidden layer dimension (default: 64)
GNN_NUM_LAYERS=3         # Number of GNN layers (default: 3)

# Link prediction
GNN_LINK_PREDICTION_TOP_K=10  # Number of top link predictions (default: 10)
```

---

## Integration Points

### 1. Pipeline Initialization

GNN modules are initialized in `TrainingPipeline.__init__()`:

```python
pipeline = TrainingPipeline(
    enable_gnn=True,                    # Enable GNN
    enable_gnn_embeddings=True,         # Enable embeddings
    enable_gnn_classification=True,     # Enable classification
    enable_gnn_link_prediction=True,    # Enable link prediction
    enable_gnn_device="cuda"            # Use GPU
)
```

### 2. Step 3c: GNN Embeddings Generation

**Location**: `pipeline.py` - Step 3c

**What happens**:
- Generates graph-level embeddings
- Generates node-level embeddings
- Optionally classifies nodes
- Optionally predicts missing links

**Output**:
- `gnn_embeddings`: Graph and node embeddings
- `gnn_classifications`: Node classifications (if enabled and model trained)
- `gnn_link_predictions`: Predicted links (if enabled and model trained)

### 3. Step 4: Feature Generation

**Location**: `pipeline.py` - `_generate_training_features()`

**What happens**:
- **Primary**: Uses GNN embeddings as features (if available)
- **Fallback**: Uses manual feature engineering (if GNN not available)
- Adds GNN classifications and link predictions as additional features

**Feature Types**:
- `gnn_graph_embedding`: Graph-level embedding vector
- `gnn_node_embeddings`: Node-level embeddings dictionary
- `gnn_node_classifications`: Node classification results
- `gnn_link_predictions`: Predicted missing links

---

## Usage Examples

### Basic Usage (Embeddings Only)

```python
import os
os.environ["ENABLE_GNN"] = "true"
os.environ["ENABLE_GNN_EMBEDDINGS"] = "true"
os.environ["ENABLE_GNN_CLASSIFICATION"] = "false"
os.environ["ENABLE_GNN_LINK_PREDICTION"] = "false"

from training.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline(
    project_id="project-001",
    system_id="system-001",
    json_tables=["tables.json"]
)

# Check GNN results
if results["steps"]["gnn_embeddings"]["status"] == "success":
    print(f"Generated embeddings: {results['steps']['gnn_embeddings']['embedding_dim']} dimensions")
```

### Full GNN Usage

```python
import os
os.environ["ENABLE_GNN"] = "true"
os.environ["ENABLE_GNN_EMBEDDINGS"] = "true"
os.environ["ENABLE_GNN_CLASSIFICATION"] = "true"
os.environ["ENABLE_GNN_LINK_PREDICTION"] = "true"
os.environ["GNN_DEVICE"] = "cuda"
os.environ["GNN_EMBEDDING_DIM"] = "256"
os.environ["GNN_HIDDEN_DIM"] = "128"

from training.pipeline import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline(
    project_id="project-001",
    system_id="system-001",
    json_tables=["tables.json"]
)

# Check all GNN results
print(f"Embeddings: {results['steps']['gnn_embeddings']}")
print(f"Classifications: {results['steps']['gnn_classification']}")
print(f"Link Predictions: {results['steps']['gnn_link_prediction']}")
```

### Direct Module Usage

```python
from training.gnn_embeddings import GNNEmbedder
from training.gnn_node_classifier import GNNNodeClassifier

# Initialize embedder
embedder = GNNEmbedder(
    embedding_dim=128,
    hidden_dim=64,
    num_layers=3,
    device="cuda"
)

# Generate embeddings
nodes = [...]  # Your graph nodes
edges = [...]  # Your graph edges

embeddings = embedder.generate_embeddings(nodes, edges, graph_level=True)
print(f"Graph embedding: {embeddings['graph_embedding']}")
```

---

## Training Models

### Node Classifier Training

```python
from training.gnn_node_classifier import GNNNodeClassifier

classifier = GNNNodeClassifier()

# Train on labeled data
training_result = classifier.train(
    nodes=training_nodes,
    edges=training_edges,
    labels=node_labels,  # Dict mapping node_id -> class
    epochs=100,
    lr=0.01
)

# Save model
classifier.save_model("models/node_classifier.pt")

# Later: Load and use
classifier.load_model("models/node_classifier.pt")
classifications = classifier.classify_nodes(nodes, edges)
```

### Link Predictor Training

```python
from training.gnn_link_predictor import GNNLinkPredictor

predictor = GNNLinkPredictor()

# Train on existing edges
training_result = predictor.train(
    nodes=training_nodes,
    edges=training_edges,
    epochs=100,
    lr=0.01,
    neg_samples=1
)

# Save model
predictor.save_model("models/link_predictor.pt")

# Later: Predict links
predictor.load_model("models/link_predictor.pt")
predictions = predictor.predict_links(nodes, edges, top_k=10)
```

---

## Pipeline Flow with GNN

```
Step 1: Extract Knowledge Graph
  ↓
Step 2: Query Glean (optional)
  ↓
Step 3: Learn Patterns (traditional)
  ↓
Step 3c: Generate GNN Embeddings ← NEW
  ├─ Graph-level embeddings
  ├─ Node-level embeddings
  ├─ Node classification (if enabled & trained)
  └─ Link prediction (if enabled & trained)
  ↓
Step 3d: Get Semantic Embeddings (optional)
  ↓
Step 4: Generate Training Features
  ├─ Primary: GNN embeddings (if available)
  ├─ Fallback: Manual features (if GNN not available)
  ├─ GNN classifications (if available)
  └─ GNN link predictions (if available)
  ↓
Step 5: Prepare Dataset
```

---

## Results Tracking

The pipeline now tracks GNN results in `results["steps"]`:

```python
results = {
    "steps": {
        "gnn_embeddings": {
            "status": "success",
            "embedding_dim": 128,
            "num_nodes": 100,
            "num_edges": 200
        },
        "gnn_classification": {
            "status": "success",
            "num_classified": 100,
            "num_classes": 5
        },
        "gnn_link_prediction": {
            "status": "success",
            "num_predictions": 10,
            "num_candidates": 1000
        }
    }
}
```

---

## Feature Output Format

### GNN Graph Embedding

```json
{
    "type": "gnn_graph_embedding",
    "data": [0.1, 0.2, 0.3, ...],  // 128-dim vector
    "embedding_dim": 128,
    "source": "gnn_embedder"
}
```

### GNN Node Embeddings

```json
{
    "type": "gnn_node_embeddings",
    "data": {
        "node_id_1": [0.1, 0.2, ...],
        "node_id_2": [0.3, 0.4, ...]
    },
    "embedding_dim": 128,
    "num_nodes": 100,
    "source": "gnn_embedder"
}
```

### GNN Classifications

```json
{
    "type": "gnn_node_classifications",
    "data": [
        {
            "node_id": "node_1",
            "predicted_class": "table",
            "confidence": 0.95
        }
    ],
    "class_mapping": {"0": "table", "1": "column"},
    "source": "gnn_classifier"
}
```

### GNN Link Predictions

```json
{
    "type": "gnn_link_predictions",
    "data": [
        {
            "source_id": "node_1",
            "target_id": "node_2",
            "probability": 0.87
        }
    ],
    "num_candidates": 1000,
    "source": "gnn_link_predictor"
}
```

---

## Troubleshooting

### GNN Not Running

**Check**:
1. `ENABLE_GNN=true` is set
2. PyTorch Geometric is installed: `pip install torch-geometric`
3. Check logs for initialization messages

### Models Not Trained

**Issue**: Classification/link prediction returns "model not trained"

**Solution**: Train models first:
```python
classifier.train(nodes, edges, labels, epochs=100)
classifier.save_model("model.pt")
```

### GPU Not Used

**Check**:
1. CUDA available: `torch.cuda.is_available()`
2. `GNN_DEVICE=cuda` is set
3. Check logs for device initialization

### Memory Issues

**Solutions**:
- Reduce `GNN_EMBEDDING_DIM` (default: 128)
- Reduce `GNN_HIDDEN_DIM` (default: 64)
- Use CPU: `GNN_DEVICE=cpu`
- Process smaller graphs

---

## Next Steps

1. **Train Models**: Train node classifier and link predictor on your data
2. **Evaluate**: Compare GNN features vs manual features
3. **Tune Hyperparameters**: Adjust embedding_dim, hidden_dim, num_layers
4. **Integrate with Existing**: Combine with `GNNRelationshipPatternLearner`

---

## Related Documentation

- [GNN Review and Rating](./GNN_REVIEW_AND_RATING.md)
- [Training Pipeline GNN Opportunities](./TRAINING_PIPELINE_GNN_OPPORTUNITIES.md)
- [Training Pipeline](../services/training/pipeline.py)

