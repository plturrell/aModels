# Semantic Spacetime GNN with Narrative Intelligence

A comprehensive system for modeling dynamic graphs with temporal evolution and narrative intelligence, supporting explanatory AI, causal prediction, and anomaly detection.

## Architecture Overview

This system extends traditional GNNs with:

1. **Temporal Modeling**: Nodes and edges that evolve over time with state history
2. **Semantic Spacetime**: Combining semantic embeddings with temporal dynamics
3. **Narrative Intelligence**: Storyline-first modeling for human-understandable insights

## Core Components

### Data Structures

- **TemporalNode**: Nodes with lifespan, state history, and dynamic state
- **TemporalEdge**: Edges with temporal scope and time-varying weights
- **TemporalGraph**: Manages temporal graph snapshots and queries
- **NarrativeNode**: Extends TemporalNode with narrative roles and causal influence
- **NarrativeEdge**: Extends TemporalEdge with narrative significance
- **Storyline**: Represents coherent narrative threads
- **NarrativeGraph**: Manages storylines and narrative-aware operations

### Core Capabilities

1. **Explanatory AI** (`ExplanationGenerator`):
   - Generates human-readable explanations
   - Identifies key actors and turning points
   - Supports counterfactual reasoning

2. **Causal Prediction** (`NarrativePredictor`):
   - Predicts future narrative states
   - Generates plausible trajectory candidates
   - Scores by coherence and causal plausibility

3. **Anomaly Detection** (`NarrativeAnomalyDetector`):
   - Detects narrative violations
   - Checks character arc consistency
   - Validates causal chains

4. **Unified System** (`MultiPurposeNarrativeGNN`):
   - Switches between task modes
   - Maintains narrative consistency
   - Task-specific message weighting

## Quick Start

### Basic Usage

```python
from gnn_spacetime.narrative import (
    NarrativeGraph, MultiPurposeNarrativeGNN,
    NarrativeNode, NarrativeEdge, Storyline, NarrativeType
)

# Create narrative graph
graph = NarrativeGraph(nodes=[...], edges=[...], storylines={...})

# Initialize GNN
gnn = MultiPurposeNarrativeGNN(narrative_graph=graph)

# Generate explanation
explanation = gnn.forward(
    graph, current_time=10.0, task_mode="explain", storyline_id="merger_story"
)

# Predict future
prediction = gnn.forward(
    graph, current_time=10.0, task_mode="predict", storyline_id="merger_story"
)

# Detect anomalies
anomalies = gnn.forward(
    graph, current_time=10.0, task_mode="detect_anomalies", storyline_id="merger_story"
)
```

### Loading Data

```python
from gnn_spacetime.data.narrative_data_loader import NarrativeDataLoader
from gnn_spacetime.data.sample_data_generator import generate_synthetic_corporate_merger

# Generate sample data
events = generate_synthetic_corporate_merger(
    company_a="CompanyA",
    company_b="CompanyB",
    duration_days=180,
    num_events=20
)

# Load into narrative graph
loader = NarrativeDataLoader()
graph = loader.convert_raw_events_to_narrative_graph(events)
```

### Backward Compatibility

```python
from gnn_spacetime.data.narrative_data_loader import load_narrative_from_temporal_graph

# Convert existing TemporalGraph to NarrativeGraph
legacy_graph = load_legacy_temporal_graph()
narrative_graph = load_narrative_from_temporal_graph(legacy_graph)
```

## Evaluation

```python
from gnn_spacetime.evaluation.metrics import (
    evaluate_explanation_quality,
    evaluate_prediction_accuracy,
    evaluate_anomaly_detection
)

# Evaluate explanation
quality = evaluate_explanation_quality(
    generated_explanation="...",
    reference_explanation="...",
    key_entities=["CompanyA", "CompanyB"]
)

# Evaluate prediction
accuracy = evaluate_prediction_accuracy(
    predicted_events=[...],
    actual_events=[...]
)

# Evaluate anomaly detection
detection = evaluate_anomaly_detection(
    detected_anomalies=[...],
    ground_truth_anomalies=[...]
)
```

## Benchmarking

```python
from gnn_spacetime.benchmarks import (
    benchmark_explanation_generation,
    benchmark_prediction_accuracy,
    benchmark_runtime_performance
)

# Benchmark explanation generation
results = benchmark_explanation_generation(
    gnn, graph, time_points=[10.0, 20.0, 30.0], storyline_ids=["merger_story"]
)
```

## Testing

```bash
# Run integration tests
python -m pytest services/training/gnn_spacetime/tests/test_narrative_system.py

# Run with verbose output
python -m pytest services/training/gnn_spacetime/tests/ -v
```

## Production Readiness Checklist

- [x] **Data Structures**: Temporal and narrative data models
- [x] **Core Capabilities**: Explanation, prediction, anomaly detection
- [x] **Integration Tests**: Test harness for all modes
- [x] **Evaluation Metrics**: Quality, accuracy, detection metrics
- [x] **Data Loading**: Convert raw events to narrative graphs
- [x] **Backward Compatibility**: Convert TemporalGraph to NarrativeGraph
- [x] **Sample Data**: Synthetic data generators
- [x] **Benchmarking**: Performance measurement utilities
- [ ] **Error Handling**: Production-grade error handling
- [ ] **Logging**: Comprehensive logging for narrative decisions
- [ ] **Configuration**: Task-mode switching configuration
- [ ] **Monitoring**: Narrative quality metrics monitoring
- [ ] **Versioning**: Storyline evolution versioning

## API Documentation

### MultiPurposeNarrativeGNN

Main entry point for narrative intelligence.

**Methods:**
- `forward(graph, current_time, task_mode, storyline_id)`: Run narrative intelligence
- `narrative_aware_message_passing(...)`: Compute task-specific message weights

**Task Modes:**
- `"explain"`: Generate explanations
- `"predict"`: Predict future states
- `"detect_anomalies"`: Detect violations

### NarrativeGraph

Manages narrative graph with storylines.

**Methods:**
- `add_storyline(storyline)`: Add storyline
- `get_nodes_in_storyline(storyline_id)`: Get nodes in storyline
- `identify_key_actors(storyline_id)`: Find key actors
- `find_narrative_turning_points(storyline_id)`: Find turning points
- `build_causal_chain(storyline_id)`: Build causal chain

## Performance Considerations

- **Memory**: Narrative metadata adds ~20% overhead per node/edge
- **Compute**: Storyline coherence calculation is O(n) where n = nodes in storyline
- **Storage**: Narrative state serialization uses JSON format

## Scale Targets

- **Small**: < 1K nodes, < 5K edges, < 10 storylines
- **Medium**: 1K-10K nodes, 5K-50K edges, 10-100 storylines
- **Large**: > 10K nodes, > 50K edges, > 100 storylines

## Latency Requirements

- **Real-time**: < 100ms per query (small graphs)
- **Batch**: < 1s per query (medium graphs)
- **Offline**: < 10s per query (large graphs)

## Next Steps

1. **Synthetic Data Testing**: Validate on generated narratives
2. **Domain-Specific Validation**: Test on business/scientific/social narratives
3. **Real-World Deployment**: Gradual rollout with shadow mode
4. **Performance Optimization**: GPU acceleration, caching, batching
5. **User Interface**: API endpoints for narrative queries

