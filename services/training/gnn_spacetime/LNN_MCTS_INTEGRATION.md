# LNN and MCTS Integration for Semantic Spacetime GNNs

## Overview

This document describes the integration of **Liquid Neural Networks (LNNs)** and **Monte Carlo Tree Search (MCTS)** into the semantic spacetime GNN system, significantly enhancing temporal modeling and decision-making capabilities.

## Key Enhancements

### 1. Liquid Neural Networks (LNNs)

**Benefits:**
- **Continuous-time temporal modeling**: LNNs process data with continuous-time dynamics, making them ideal for `TemporalNode` states (`h(t)`) and `TemporalEdge` weights (`w(t)`)
- **Temporal robustness**: Knowledge transfers effectively to new time nodes and environments with complex, previously unseen changes
- **Efficiency**: Smaller networks with less computation than traditional RNNs/LSTMs
- **Interpretability**: Continuous-time nature makes decisions more interpretable

**Implementation:**
- `LiquidLayer`: Core LNN layer with continuous-time dynamics
- `LiquidStateUpdater`: Replaces RNN/LSTM/GRU for node state evolution
- `LiquidEdgeWeightUpdater`: Models time-varying edge weights

### 2. Monte Carlo Tree Search (MCTS)

**Benefits:**
- **Narrative path planning**: Plans complex sequences of events/actions in narrative space
- **What-if analysis**: Deeply explores counterfactual scenarios through rollouts
- **Balanced exploration/exploitation**: Uses UCB1 to balance trying new strategies vs. relying on known ones
- **GNN-accelerated rollouts**: GNN predicts rollout outcomes, massively speeding up search

**Implementation:**
- `NarrativeMCTS`: Base MCTS for narrative reasoning
- `GNNMCTS`: GNN-accelerated MCTS with learned rollout predictions
- `NarrativePathMCTS`: Specialized MCTS for narrative path planning

### 3. Enhanced Narrative GNN

**`EnhancedNarrativeGNN`** combines LNN and MCTS:
- LNN-based temporal state evolution
- MCTS for narrative path planning
- GNN-accelerated MCTS rollouts
- Enhanced what-if analysis

## Architecture

```
EnhancedNarrativeGNN
├── LiquidStateUpdater (LNN)
│   └── Continuous-time node state evolution: h(t)
├── LiquidEdgeWeightUpdater (LNN)
│   └── Continuous-time edge weight evolution: w(t)
├── GNNMCTS
│   ├── GNN-accelerated rollouts
│   └── Narrative path planning
└── NarrativePathMCTS
    └── Specialized narrative reasoning
```

## Usage

### Basic Usage

```python
from gnn_spacetime.narrative import EnhancedNarrativeGNN, NarrativeGraph

# Initialize with LNN and MCTS
enhanced_gnn = EnhancedNarrativeGNN(
    narrative_graph=graph,
    use_lnn=True,      # Enable LNN
    use_mcts=True,    # Enable MCTS
    lnn_time_constant=1.0,
    mcts_rollouts=100,
    mcts_exploration_c=1.414
)

# Use for standard tasks
result = enhanced_gnn.forward(
    graph=graph,
    current_time=0.0,
    task_mode="explain",
    storyline_id="merger_story"
)
```

### LNN State Updates

```python
# Update node state using LNN
updated_state = enhanced_gnn.update_node_state_lnn(
    node_id="node_1",
    messages=torch.randn(128),
    prev_state=torch.randn(64),
    time_delta=torch.tensor(0.1)
)

# Update edge weight using LNN
updated_weight = enhanced_gnn.update_edge_weight_lnn(
    source_features=torch.randn(128),
    target_features=torch.randn(128),
    relation_embedding=torch.randn(64),
    prev_weight=torch.tensor(0.7),
    time_delta=torch.tensor(0.1)
)
```

### MCTS Path Planning

```python
# Plan narrative path using MCTS
current_state = {
    "time": 0.0,
    "storyline_id": "merger_story",
    "events": []
}

action_sequence, best_value = enhanced_gnn.plan_narrative_path_mcts(
    current_state=current_state,
    storyline_id="merger_story",
    num_iterations=100
)
```

### MCTS What-If Analysis

```python
# Perform what-if analysis using MCTS
counterfactual = {
    "remove_node": "key_player_1",
    "description": "What if key player was removed?"
}

result = enhanced_gnn.what_if_analysis_mcts(
    counterfactual_condition=counterfactual,
    storyline_id="merger_story",
    num_iterations=200
)

print(f"Original value: {result['original_value']}")
print(f"Counterfactual value: {result['counterfactual_value']}")
print(f"Difference: {result['difference']}")
```

## Integration Paths

### Path 1: LNNs for Core State Evolution

Replace internal state-update mechanisms (RNNs in `TemporalNode` and `TemporalEdge`) with LNNs:

```python
# Before: RNN-based
state_updater = GRUStateUpdater(input_dim=128, hidden_dim=64)

# After: LNN-based
state_updater = LiquidStateUpdater(
    input_dim=128,
    hidden_dim=64,
    time_constant=1.0
)
```

### Path 2: MCTS for Narrative Reasoning

Integrate MCTS as a dedicated reasoning module:

```python
from gnn_spacetime.narrative import NarrativePathMCTS

mcts = NarrativePathMCTS(
    narrative_graph=graph,
    storyline_id="merger_story",
    num_rollouts=100
)

best_action, value = mcts.search(initial_state)
```

### Path 3: GNN-MCTS Synergy

Use GNN to accelerate MCTS rollouts:

```python
from gnn_spacetime.narrative import GNNMCTS

gnn_mcts = GNNMCTS(
    gnn_predictor=gnn_model.predict_state_value,
    num_rollouts=100,
    gnn_rollout_prob=0.8  # 80% use GNN, 20% random rollout
)
```

## Performance Considerations

### LNN Benefits
- **Faster inference**: Smaller networks, less computation
- **Better generalization**: Temporal robustness to novel environments
- **Smoother evolution**: Continuous-time dynamics avoid discrete step artifacts

### MCTS Benefits
- **Intelligent planning**: Explores promising paths while avoiding poor ones
- **Scalable**: Can handle complex narrative spaces
- **GNN acceleration**: 10-100x speedup when using GNN for rollouts

## Example: Full Integration

See `examples/lnn_mcts_integration_example.py` for a complete demonstration of:
1. LNN-based state updates
2. MCTS path planning
3. What-if analysis
4. Comparison with standard GNN

## Future Enhancements

1. **Reflective-MCTS (R-MCTS)**: Multi-agent debate for balanced assessments
2. **Adaptive time constants**: Learn optimal time constants per node/edge
3. **Hierarchical MCTS**: Multi-level planning for complex narratives
4. **Transfer learning**: Pre-trained LNNs for faster adaptation

## References

- Liquid Neural Networks: Continuous-time neural models with temporal robustness
- Monte Carlo Tree Search: UCB1-based exploration/exploitation
- GNN-MCTS: Hybrid approach for efficient planning

