# GNN Priority 2 Implementation: Training & Evaluation

## Overview

Priority 2 implementation adds comprehensive training capabilities, evaluation metrics, and validation for GNN models integrated with the training pipeline.

## Components Implemented

### 1. GNN Evaluation Module (`gnn_evaluation.py`)

**Purpose**: Provides evaluation metrics for all GNN models.

**Features**:
- **Classification Metrics**: Accuracy, precision, recall, F1-score, confusion matrix, ROC AUC
- **Link Prediction Metrics**: Accuracy, precision, recall, F1-score, ROC AUC, PR AUC
- **Embedding Quality Metrics**: Silhouette score, adjusted Rand index, embedding statistics
- **Schema Matching Metrics**: Similarity-based evaluation
- **Baseline Comparison**: Compare GNN performance with baseline models

**Key Classes**:
- `GNNEvaluator`: Main evaluation class with methods for all model types

**Usage**:
```python
from training.gnn_evaluation import GNNEvaluator

evaluator = GNNEvaluator()

# Evaluate classification
metrics = evaluator.evaluate_classification(y_true, y_pred, y_proba)

# Evaluate link prediction
metrics = evaluator.evaluate_link_prediction(y_true, y_pred, y_proba)

# Evaluate embeddings
metrics = evaluator.evaluate_embeddings(embeddings, labels=labels)
```

### 2. GNN Training Module (`gnn_training.py`)

**Purpose**: Provides training capabilities for all GNN models with data preparation, training loops, and model persistence.

**Features**:
- **Node Classifier Training**: Train with validation split, label extraction, model persistence
- **Link Predictor Training**: Train with negative sampling, validation, model persistence
- **Anomaly Detector Training**: Train autoencoder for anomaly detection
- **Schema Matcher Training**: Train on schema pairs with similarity labels
- **Model Loading**: Load trained models for inference

**Key Classes**:
- `GNNTrainer`: Main training orchestrator

**Usage**:
```python
from training.gnn_training import GNNTrainer

trainer = GNNTrainer(output_dir="./gnn_models", device="cpu")

# Train node classifier
result = trainer.train_node_classifier(
    nodes, edges,
    labels=labels,
    epochs=100,
    lr=0.01,
    validation_split=0.2
)

# Train link predictor
result = trainer.train_link_predictor(
    nodes, edges,
    epochs=100,
    lr=0.01
)

# Load trained models
models = trainer.load_trained_models({
    "classifier": "path/to/classifier.pt",
    "link_predictor": "path/to/predictor.pt"
})
```

### 3. Pipeline Integration

**Training Pipeline Enhancements**:

1. **GNN Trainer Initialization**: 
   - Added `enable_gnn_training` parameter
   - Added `gnn_models_dir` parameter
   - Automatic initialization if enabled

2. **Training Methods**:
   - `train_gnn_models()`: Train multiple GNN models
   - `load_trained_gnn_models()`: Load pre-trained models
   - `_evaluate_gnn_models()`: Evaluate GNN outputs during pipeline

3. **Pipeline Integration**:
   - Training can be triggered before pipeline execution
   - Trained models are automatically loaded into GNN modules
   - Evaluation runs automatically during feature generation

**Usage**:
```python
from training.pipeline import TrainingPipeline

# Initialize with training enabled
pipeline = TrainingPipeline(
    enable_gnn=True,
    enable_gnn_training=True,
    gnn_models_dir="./gnn_models"
)

# Run pipeline with training
results = pipeline.run_full_pipeline(
    project_id="project_1",
    train_gnn_models=True,
    gnn_training_epochs=100
)

# Or train separately
training_results = pipeline.train_gnn_models(
    nodes, edges,
    train_classifier=True,
    train_link_predictor=True,
    epochs=100
)
```

### 4. Test Suite (`test_gnn_integration.py`)

**Purpose**: Comprehensive tests for GNN integration, training, and evaluation.

**Test Coverage**:
- GNN embedder initialization and generation
- Node classifier training
- Link predictor training
- Evaluation metrics (classification, link prediction)
- GNN trainer functionality
- Pipeline integration

**Running Tests**:
```bash
cd services/training
python -m pytest test_gnn_integration.py -v
```

## Dependencies

### New Dependencies
- `scikit-learn>=1.3.0`: For evaluation metrics (accuracy, precision, recall, F1, ROC AUC, etc.)

### Existing Dependencies
- `torch>=2.0.0`: PyTorch for GNN models
- `torch-geometric>=2.3.0`: Graph neural network library
- `numpy>=1.24.0`: Numerical operations

## Configuration

### Environment Variables

```bash
# Enable GNN training
ENABLE_GNN_TRAINING=true

# GNN models directory
GNN_MODELS_DIR=./gnn_models

# Device for training (auto, cpu, cuda)
GNN_DEVICE=auto
```

### Pipeline Parameters

```python
# Training parameters
train_gnn_models: bool = False  # Enable training
gnn_training_epochs: int = 100  # Number of epochs

# Trainer initialization
enable_gnn_training: Optional[bool] = None
gnn_models_dir: Optional[str] = None
```

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Weighted precision across classes
- **Recall**: Weighted recall across classes
- **F1-Score**: Weighted F1-score
- **Confusion Matrix**: Per-class performance
- **ROC AUC**: Area under ROC curve (if probabilities available)

### Link Prediction Metrics
- **Accuracy**: Overall link prediction accuracy
- **Precision**: Precision for positive links
- **Recall**: Recall for positive links
- **F1-Score**: F1-score for link prediction
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under precision-recall curve

### Embedding Quality Metrics
- **Silhouette Score**: Cluster quality (requires labels)
- **Adjusted Rand Index**: Clustering agreement (requires labels)
- **Inertia**: K-means inertia
- **Embedding Statistics**: Mean/std of embedding norms and values

## Training Workflow

1. **Data Preparation**:
   - Extract nodes and edges from knowledge graph
   - Extract or create labels for supervised learning
   - Split data into training and validation sets

2. **Model Training**:
   - Initialize GNN model
   - Train for specified epochs
   - Validate on validation set
   - Save trained model

3. **Evaluation**:
   - Evaluate on validation set
   - Compute metrics (accuracy, precision, recall, F1)
   - Compare with baseline (if available)

4. **Model Persistence**:
   - Save model to disk
   - Load model for inference
   - Integrate with pipeline

## Integration Points

### With Training Pipeline

1. **Before Pipeline Execution**:
   - Train GNN models on initial graph
   - Load trained models into GNN modules
   - Use trained models for inference during pipeline

2. **During Pipeline Execution**:
   - Generate embeddings using trained embedder
   - Classify nodes using trained classifier
   - Predict links using trained predictor
   - Evaluate outputs automatically

3. **After Pipeline Execution**:
   - Training results included in pipeline results
   - Evaluation metrics available in results
   - Models saved for future use

### With GNN Modules (Priority 1)

- **GNNEmbedder**: Can use pre-trained embeddings
- **GNNNodeClassifier**: Uses trained classifier for inference
- **GNNLinkPredictor**: Uses trained predictor for inference
- **GNNAnomalyDetector**: Uses trained detector for anomaly detection

## Example Usage

### Complete Training Workflow

```python
from training.pipeline import TrainingPipeline

# Initialize pipeline with training
pipeline = TrainingPipeline(
    output_dir="./output",
    enable_gnn=True,
    enable_gnn_training=True,
    enable_gnn_classification=True,
    enable_gnn_link_prediction=True
)

# Run pipeline with training
results = pipeline.run_full_pipeline(
    project_id="project_1",
    system_id="system_1",
    train_gnn_models=True,
    gnn_training_epochs=100
)

# Check training results
if "gnn_training" in results["steps"]:
    training_results = results["steps"]["gnn_training"]
    print(f"Models trained: {training_results.get('models_trained', [])}")
    
    for model_type, result in training_results.get("training_results", {}).items():
        if "error" not in result:
            print(f"{model_type}: accuracy={result.get('training_accuracy', 0.0):.4f}")

# Check evaluation results
if "gnn_evaluation" in results["steps"]:
    eval_results = results["steps"]["gnn_evaluation"]
    print(f"Evaluation: {eval_results}")
```

### Standalone Training

```python
from training.gnn_training import GNNTrainer
from training.gnn_evaluation import GNNEvaluator

# Initialize trainer
trainer = GNNTrainer(output_dir="./gnn_models", device="cpu")

# Train models
classifier_result = trainer.train_node_classifier(
    nodes, edges,
    labels=labels,
    epochs=100,
    validation_split=0.2
)

predictor_result = trainer.train_link_predictor(
    nodes, edges,
    epochs=100,
    validation_split=0.2
)

# Evaluate
evaluator = GNNEvaluator()
if classifier_result.get("validation_metrics"):
    metrics = classifier_result["validation_metrics"]
    print(f"Classification accuracy: {metrics.get('accuracy', 0.0):.4f}")
```

## Status

✅ **Completed**:
- Evaluation metrics module
- Training module
- Pipeline integration
- Test suite
- Model persistence
- Validation split support

## Next Steps (Priority 3)

1. **Multi-modal Learning**: Combine GNNs with semantic embeddings
2. **Hyperparameter Tuning**: Automated hyperparameter optimization
3. **Cross-validation**: K-fold cross-validation for robust evaluation
4. **Model Ensembling**: Combine multiple GNN models
5. **Transfer Learning**: Pre-trained models for different domains
6. **Real-time Training**: Incremental learning on streaming data

## Notes

- Training requires labeled data for supervised learning (classification, link prediction)
- Evaluation metrics require scikit-learn
- Models are saved in PyTorch format (`.pt` files)
- GPU support available if CUDA is enabled
- Training can be computationally intensive for large graphs

