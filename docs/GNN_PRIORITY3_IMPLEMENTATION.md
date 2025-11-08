# GNN Priority 3 Implementation: Advanced Intelligence Features

## Overview

Priority 3 implementation adds advanced intelligence features to the GNN system:
- Multi-modal learning (combining GNN with semantic, temporal, and domain features)
- Hyperparameter tuning (automated optimization)
- Cross-validation (robust evaluation)
- Model ensembling (combining multiple models)
- Transfer learning (pre-training and fine-tuning)
- Active learning (uncertainty-based sampling and continuous learning)

## Components Implemented

### 1. Multi-Modal Learning (`gnn_multimodal.py`)

**Purpose**: Combines GNN embeddings with semantic embeddings (SAP RPT), temporal patterns, and domain configurations.

**Features**:
- **MultiModalFusion**: Neural fusion layer with attention, concatenation, or weighted combination
- **MultiModalGNN**: End-to-end multi-modal GNN that combines:
  - GNN embeddings (graph structure)
  - Semantic embeddings (SAP RPT)
  - Temporal pattern features
  - Domain configuration features

**Fusion Methods**:
- **Attention**: Multi-head attention over modalities
- **Concatenation**: Simple concatenation + projection
- **Weighted**: Learnable weighted combination

**Usage**:
```python
from training.gnn_multimodal import MultiModalGNN

# Initialize with GNN embedder
multimodal_gnn = MultiModalGNN(
    gnn_embedder=gnn_embedder,
    fusion_method="attention",
    output_dim=256
)

# Generate multi-modal embeddings
result = multimodal_gnn.generate_multimodal_embeddings(
    nodes, edges,
    semantic_embeddings=semantic_embeddings,
    temporal_patterns=temporal_patterns,
    domain_configs=domain_configs
)
```

### 2. Hyperparameter Tuning (`gnn_hyperparameter_tuning.py`)

**Purpose**: Automated hyperparameter optimization for GNN models.

**Features**:
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling from parameter space
- **Bayesian Optimization**: Optuna-based optimization (requires optuna)

**Supported Models**:
- Node classifier
- Link predictor

**Usage**:
```python
from training.gnn_hyperparameter_tuning import GNNHyperparameterTuner

tuner = GNNHyperparameterTuner(
    method="bayesian",  # or "grid_search", "random_search"
    n_trials=20
)

# Tune node classifier
result = tuner.tune_node_classifier(
    nodes, edges, labels,
    epochs_per_trial=50
)

print(f"Best params: {result['best_params']}")
print(f"Best score: {result['best_score']}")
```

### 3. Cross-Validation (`gnn_cross_validation.py`)

**Purpose**: K-fold cross-validation for robust evaluation.

**Features**:
- K-fold splitting for nodes (classification)
- K-fold splitting for edges (link prediction)
- Aggregated metrics (mean, std) across folds
- Overall evaluation on all predictions

**Usage**:
```python
from training.gnn_cross_validation import GNNCrossValidator

validator = GNNCrossValidator(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Cross-validate node classifier
cv_results = validator.cross_validate_node_classifier(
    nodes, edges, labels,
    epochs=100
)

print(f"Mean accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
```

### 4. Model Ensembling (`gnn_ensembling.py`)

**Purpose**: Combine multiple GNN models for improved performance.

**Features**:
- **GNNEnsemble**: Ensemble of trained models
- **Voting**: Majority voting for classification
- **Weighted Voting**: Weighted combination
- **Stacking**: Meta-learner (placeholder)
- **GNNEnsembleBuilder**: Builder for creating diverse ensembles

**Usage**:
```python
from training.gnn_ensembling import GNNEnsembleBuilder

builder = GNNEnsembleBuilder(device="cpu")

# Build diverse ensemble
ensemble = builder.build_diverse_ensemble(
    nodes, edges, labels,
    num_models=3
)

# Make predictions
predictions = ensemble.predict_classification(nodes, edges)
```

### 5. Transfer Learning (`gnn_transfer_learning.py`)

**Purpose**: Pre-train on large graphs, fine-tune for specific domains.

**Features**:
- **Pre-training**: Train on large graphs
- **Fine-tuning**: Adapt to specific domains
- **Model Registry**: Store and manage pre-trained models
- **Model Sharing**: Copy models between locations

**Usage**:
```python
from training.gnn_transfer_learning import GNNTransferLearner

learner = GNNTransferLearner(
    model_registry_dir="./gnn_models"
)

# Pre-train on large graph
pretrain_result = learner.pretrain_on_large_graph(
    large_graph_nodes, large_graph_edges,
    model_type="classifier",
    epochs=200
)

# Fine-tune for domain
finetune_result = learner.fine_tune_for_domain(
    pretrain_result["model_path"],
    domain_nodes, domain_edges,
    domain_labels,
    fine_tune_epochs=50,
    freeze_backbone=False
)

# List available models
models = learner.list_available_models(model_type="classifier")
```

### 6. Active Learning (`gnn_active_learning.py`)

**Purpose**: Uncertainty-based sampling and continuous learning.

**Features**:
- **Uncertainty Sampling**: Select samples with highest uncertainty
- **Diversity Sampling**: Select diverse samples
- **Hybrid Sampling**: Combine uncertainty and diversity
- **User Feedback**: Integrate user-provided labels
- **Continuous Learning**: Iterative learning loop

**Usage**:
```python
from training.gnn_active_learning import GNNActiveLearner

learner = GNNActiveLearner(
    model=classifier,
    sampling_strategy="uncertainty"
)

# Select samples for labeling
selected = learner.select_samples_for_labeling(
    nodes, edges,
    num_samples=10
)

# Add user feedback
for sample in selected:
    learner.add_user_feedback(
        sample["node_id"],
        user_provided_label
    )

# Update model
update_result = learner.update_model_with_feedback(
    nodes, edges,
    epochs=10
)

# Continuous learning loop
history = learner.continuous_learning_loop(
    nodes, edges,
    num_iterations=5,
    samples_per_iteration=10
)
```

## Dependencies

### New Dependencies
- `optuna>=3.0.0`: For Bayesian hyperparameter optimization (optional)

### Existing Dependencies
- `torch>=2.0.0`: PyTorch
- `torch-geometric>=2.3.0`: Graph neural networks
- `scikit-learn>=1.3.0`: For cross-validation and metrics
- `numpy>=1.24.0`: Numerical operations

## Integration with Pipeline

Priority 3 features can be integrated into the training pipeline:

```python
from training.pipeline import TrainingPipeline

pipeline = TrainingPipeline(
    enable_gnn=True,
    enable_gnn_training=True
)

# Use multi-modal learning
from training.gnn_multimodal import MultiModalGNN
multimodal_gnn = MultiModalGNN(
    gnn_embedder=pipeline.gnn_embedder,
    fusion_method="attention"
)

# Use hyperparameter tuning
from training.gnn_hyperparameter_tuning import GNNHyperparameterTuner
tuner = GNNHyperparameterTuner(method="bayesian", n_trials=20)
best_params = tuner.tune_node_classifier(nodes, edges, labels)

# Use cross-validation
from training.gnn_cross_validation import GNNCrossValidator
validator = GNNCrossValidator(n_splits=5)
cv_results = validator.cross_validate_node_classifier(nodes, edges, labels)
```

## Configuration

### Environment Variables

```bash
# Multi-modal learning
ENABLE_MULTIMODAL_GNN=true
MULTIMODAL_FUSION_METHOD=attention  # attention, concat, weighted

# Hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING=true
HYPERPARAMETER_METHOD=bayesian  # grid_search, random_search, bayesian
HYPERPARAMETER_N_TRIALS=20

# Cross-validation
ENABLE_CROSS_VALIDATION=true
CV_N_SPLITS=5

# Transfer learning
GNN_MODEL_REGISTRY=./gnn_model_registry

# Active learning
ENABLE_ACTIVE_LEARNING=true
ACTIVE_LEARNING_STRATEGY=uncertainty  # uncertainty, diversity, hybrid
```

## Benefits

### Multi-Modal Learning
- **Enhanced Intelligence**: Combines graph structure with semantic meaning
- **Better Representations**: Richer feature space
- **Domain Awareness**: Incorporates domain-specific knowledge

### Hyperparameter Tuning
- **Automated Optimization**: No manual tuning needed
- **Better Performance**: Finds optimal hyperparameters
- **Time Savings**: Reduces trial-and-error

### Cross-Validation
- **Robust Evaluation**: More reliable performance estimates
- **Reduced Overfitting**: Better generalization assessment
- **Confidence Intervals**: Mean ± std metrics

### Model Ensembling
- **Improved Accuracy**: Combines strengths of multiple models
- **Reduced Variance**: More stable predictions
- **Diversity**: Different architectures capture different patterns

### Transfer Learning
- **Faster Training**: Pre-trained models reduce training time
- **Better Performance**: Leverage knowledge from large graphs
- **Domain Adaptation**: Fine-tune for specific use cases

### Active Learning
- **Efficient Labeling**: Focus on most informative samples
- **Reduced Labeling Cost**: Fewer labels needed
- **Continuous Improvement**: Iterative learning from feedback

## Status

✅ **Completed**:
- Multi-modal learning module
- Hyperparameter tuning module
- Cross-validation module
- Model ensembling module
- Transfer learning module
- Active learning module
- Integration with training pipeline
- Documentation

## Next Steps

1. **Pipeline Integration**: Full integration of Priority 3 features into `TrainingPipeline.run_full_pipeline()`
2. **Performance Optimization**: Optimize multi-modal fusion for large graphs
3. **Advanced Ensembling**: Implement stacking with meta-learner
4. **Active Learning UI**: User interface for feedback collection
5. **Model Registry API**: REST API for model sharing
6. **Benchmarking**: Performance benchmarks on real datasets

## Notes

- Multi-modal learning requires SAP RPT embeddings (optional)
- Bayesian optimization requires Optuna (optional, falls back to random search)
- Active learning requires user feedback (simulated in continuous learning loop)
- Transfer learning requires pre-trained models (can be created with pre-training)

