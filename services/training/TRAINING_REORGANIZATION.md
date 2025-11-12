# Training Service Reorganization

## Overview

The training service has been reorganized into a more structured and maintainable directory layout following the same pattern as the scripts reorganization.

## New Directory Structure

```
training/
├── api/                    # API models and schemas
├── cmd/                    # Command-line tools
├── core/                   # Core training components
│   ├── domain/            # Domain-specific components
│   ├── features/          # Feature engineering
│   └── optimization/      # Optimization algorithms
├── data/                  # Data access and processing
├── deployment/            # Deployment and operations
├── evaluation/            # Evaluation and testing
├── experiments/           # Experimental code
├── gnn_modules/           # GNN-specific modules
├── gnn_spacetime/         # GNN spacetime framework
├── models/                # Model implementations
│   ├── gnn/              # GNN models
│   ├── coralnpu/         # CoralNPU models
│   └── domain/           # Domain models
├── training/              # Training algorithms and learners
├── utils/                 # Utility functions
└── main.py               # Main entry point
```

## File Mappings

### Core Components
- `intelligence_metrics.py` → `core/`
- `semantic_features.py` → `core/`
- `temporal_analysis.py` → `core/`

### Training Components
- `pattern_learning.py` → `training/`
- `pattern_learning_gnn.py` → `training/`
- `pattern_transfer.py` → `training/`
- `active_pattern_learner.py` → `training/`
- `meta_pattern_learner.py` → `training/`
- `sequence_pattern_transformer.py` → `training/`

### Models
- `graph_client.py` → `models/gnn/`
- `test_gnn_integration.py` → `models/gnn/`
- `digital_twin.py` → `models/`
- `rollback_manager.py` → `models/`
- `routing_optimizer.py` → `models/`

### Utilities
- `langsmith_tracing.py` → `utils/`

### Main Files
- `main.py` → root directory
- `pipeline.py` → root directory

## Usage

The reorganization maintains backward compatibility while providing a cleaner structure for future development.

## Migration Notes

- All imports have been updated to reflect new paths
- API endpoints remain unchanged
- Configuration files have been updated
- Tests have been updated to use new paths
