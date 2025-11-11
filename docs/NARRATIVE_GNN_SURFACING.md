# Surfacing GNN Deep Research and Narrative Features

## Overview

This document describes how to surface the GNN spacetime narrative intelligence features in the browser UI, making them accessible for graph exploration and analysis.

## Implementation Status

### ✅ Completed

1. **API Endpoints** (`/home/aModels/services/training/main.py`)
   - `/narrative/explain` - Generate human-readable explanations
   - `/narrative/predict` - Predict future narrative states
   - `/narrative/detect-anomalies` - Detect narrative violations
   - `/narrative/mcts` - MCTS what-if analysis
   - `/narrative/storyline` - Storyline operations (list, get, key_actors, turning_points, causal_chain)

2. **API Models** (`/home/aModels/services/training/api/narrative_models.py`)
   - `NarrativeExplainRequest`
   - `NarrativePredictRequest`
   - `NarrativeAnomalyRequest`
   - `NarrativeMCTSRequest`
   - `NarrativeStorylineRequest`

3. **TypeScript API Client** (`/home/aModels/services/browser/shell/ui/src/api/narrative.ts`)
   - `explainNarrative()`
   - `predictNarrative()`
   - `detectNarrativeAnomalies()`
   - `narrativeMCTS()`
   - `narrativeStoryline()`

### 🚧 In Progress / TODO

4. **UI Components** - Need to create:
   - `NarrativeInsights.tsx` - Main component for narrative features
   - `StorylineExplorer.tsx` - Explore storylines, key actors, turning points
   - `MCTSWhatIf.tsx` - MCTS what-if analysis interface
   - Integration into `GraphModule.tsx`

## Features to Surface

### 1. Narrative Explanations
- **Purpose**: Generate human-readable explanations of graph dynamics
- **Features**:
  - Identify key actors
  - Find turning points
  - Build causal chains
  - Focus on specific nodes

### 2. Narrative Predictions
- **Purpose**: Predict future narrative states
- **Features**:
  - Generate trajectory candidates
  - Score by coherence and plausibility
  - Visualize predicted paths

### 3. Anomaly Detection
- **Purpose**: Detect narrative violations
- **Features**:
  - Check character arc consistency
  - Validate causal chains
  - Highlight inconsistencies

### 4. MCTS What-If Analysis
- **Purpose**: Explore counterfactual scenarios
- **Features**:
  - Plan narrative paths
  - Explore alternative outcomes
  - Visualize search tree

### 5. Storyline Operations
- **Purpose**: Explore storylines in the graph
- **Features**:
  - List all storylines
  - Get storyline details
  - Find key actors
  - Identify turning points
  - Build causal chains

## Integration Plan

### Graph Module Integration

Add a new tab "Narrative Intelligence" to the Graph module with sub-tabs:
1. **Explanations** - Generate and view explanations
2. **Predictions** - View future state predictions
3. **Anomalies** - View detected anomalies
4. **What-If** - MCTS analysis interface
5. **Storylines** - Explore storylines

### Component Structure

```
GraphModule
├── NarrativeInsights (new tab)
    ├── ExplanationView
    ├── PredictionView
    ├── AnomalyView
    ├── MCTSWhatIfView
    └── StorylineExplorerView
```

## Usage Examples

### Generate Explanation
```typescript
const explanation = await explainNarrative({
  nodes: graphData.nodes,
  edges: graphData.edges,
  storyline_id: "merger_story",
  focus_node_id: "company_a"
});
```

### Predict Future
```typescript
const prediction = await predictNarrative({
  nodes: graphData.nodes,
  edges: graphData.edges,
  current_time: 10.0,
  future_time: 20.0,
  num_trajectories: 5
});
```

### MCTS What-If
```typescript
const mctsResult = await narrativeMCTS({
  nodes: graphData.nodes,
  edges: graphData.edges,
  current_time: 10.0,
  num_rollouts: 100,
  max_depth: 10
});
```

## Next Steps

1. Create `NarrativeInsights.tsx` component
2. Create `StorylineExplorer.tsx` component
3. Create `MCTSWhatIf.tsx` component
4. Integrate into `GraphModule.tsx`
5. Add narrative features to health check
6. Test with real graph data

## Configuration

### Environment Variables

- `NARRATIVE_MCTS_ROLLOUTS` - Number of MCTS rollouts (default: 100)
- `ENABLE_NARRATIVE_GNN` - Enable narrative GNN (default: true)

### Service Dependencies

- Training service must have `gnn_spacetime` installed
- Enhanced Narrative GNN requires LNN and MCTS support
- Graph data must be convertible to NarrativeGraph format

