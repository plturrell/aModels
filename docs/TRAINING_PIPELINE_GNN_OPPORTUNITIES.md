# GNN Integration Opportunities for Training Pipeline

## Overview

The training pipeline (`services/training/pipeline.py`) currently uses traditional feature extraction methods. The codebase structure is well-suited for Graph Neural Network (GNN) integration, which could significantly enhance pattern learning, schema matching, and cross-system analysis.

---

## Current Pipeline Architecture

### Existing Graph Structure

The pipeline already extracts and processes graph data:

1. **Step 1: Extract Knowledge Graph**
   - Nodes: tables, columns, views, processes
   - Edges: relationships, dependencies, transformations
   - Stored in Neo4j knowledge graph

2. **Step 3: Pattern Learning**
   - Current: Traditional pattern extraction
   - Uses: `PatternLearningEngine` for column/relationship patterns

3. **Step 4: Feature Generation**
   - Current: Manual feature engineering
   - Combines: Graph structure, historical patterns, temporal insights

### Integration Points for GNNs

- **After Step 1**: Generate node/edge embeddings
- **Step 3**: Replace or enhance pattern learning with GNN-based detection
- **Step 4**: Use GNN embeddings as features instead of manual engineering
- **New Step**: Add GNN-based similarity search for cross-system matching

---

## GNN Use Cases

### 1. Node Classification

**Purpose**: Classify nodes (tables, columns) by type, domain, quality

**Application**:
- Automatically classify tables by domain (finance, risk, regulatory)
- Identify column types and semantic meaning
- Predict data quality scores for nodes
- Classify schema elements for better organization

**Implementation Approach**:
```python
# Use GraphSAGE or GCN for node classification
from torch_geometric.nn import GCNConv, GraphSAGE

class NodeClassifier(nn.Module):
    def __init__(self, num_node_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

**Integration Point**: After Step 1 (Extract), before Step 3 (Pattern Learning)

**Benefits**:
- Automatic domain detection
- Better schema organization
- Quality prediction without manual rules

---

### 2. Link Prediction

**Purpose**: Predict missing relationships or suggest new mappings

**Application**:
- Discover missing foreign key relationships
- Suggest cross-system field mappings
- Predict transformation dependencies
- Identify potential data lineage connections

**Implementation Approach**:
```python
# Use GAE (Graph Autoencoder) or GAT for link prediction
from torch_geometric.nn import GAE, GATConv

class LinkPredictor(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.encoder = GATConv(num_node_features, 64, heads=4)
        self.decoder = nn.Linear(64, 1)
    
    def forward(self, x, edge_index):
        # Encode nodes
        x = self.encoder(x, edge_index)
        # Predict edge probabilities
        return self.decoder(x)
```

**Integration Point**: Step 3 (Pattern Learning) - enhance relationship discovery

**Benefits**:
- Automatic mapping discovery
- Reduced manual mapping effort
- Better lineage completeness

---

### 3. Graph Embeddings

**Purpose**: Generate embeddings for similarity search and pattern matching

**Application**:
- Find similar schemas across systems
- Match patterns between different domains
- Semantic search for data assets
- Cross-system schema alignment

**Implementation Approach**:
```python
# Use GraphSAGE or Graph Transformer for embeddings
from torch_geometric.nn import GraphSAGE, TransformerConv

class GraphEmbedder(nn.Module):
    def __init__(self, num_node_features, embedding_dim):
        super().__init__()
        self.sage = GraphSAGE(num_node_features, embedding_dim, num_layers=3)
    
    def forward(self, x, edge_index, batch):
        # Generate node embeddings
        node_embeddings = self.sage(x, edge_index)
        # Pool to graph-level embedding
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return graph_embedding
```

**Integration Point**: Step 4 (Feature Generation) - replace manual features with embeddings

**Benefits**:
- Better feature representation
- Semantic similarity matching
- Reduced feature engineering effort

---

### 4. Anomaly Detection

**Purpose**: Detect structural anomalies in graph patterns

**Application**:
- Identify unusual schema structures
- Detect data quality issues from graph structure
- Find outliers in cross-system patterns
- Alert on structural inconsistencies

**Implementation Approach**:
```python
# Use Graph Autoencoder for anomaly detection
from torch_geometric.nn import GAE, GCNConv

class AnomalyDetector(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.encoder = GCNConv(num_node_features, 32)
        self.decoder = GCNConv(32, num_node_features)
    
    def forward(self, x, edge_index):
        # Encode
        z = self.encoder(x, edge_index)
        # Reconstruct
        x_recon = self.decoder(z, edge_index)
        # Anomaly score = reconstruction error
        anomaly_score = F.mse_loss(x, x_recon, reduction='none')
        return anomaly_score
```

**Integration Point**: New step after Step 3 (Pattern Learning)

**Benefits**:
- Automatic anomaly detection
- Structural quality assessment
- Early problem identification

---

### 5. Schema Matching

**Purpose**: Use GNNs for cross-system schema alignment

**Application**:
- Match schemas between Murex and SAP
- Align fields across different systems
- Discover semantic equivalences
- Automate mapping rule generation

**Implementation Approach**:
```python
# Use Graph Attention Networks for matching
from torch_geometric.nn import GATConv, global_mean_pool

class SchemaMatcher(nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.gat1 = GATConv(num_node_features, 64, heads=4)
        self.gat2 = GATConv(64 * 4, 128, heads=1)
        self.match_head = nn.Linear(128, 1)
    
    def forward(self, x1, edge_index1, x2, edge_index2):
        # Embed both schemas
        emb1 = self.gat2(self.gat1(x1, edge_index1), edge_index1)
        emb2 = self.gat2(self.gat1(x2, edge_index2), edge_index2)
        # Compute similarity
        similarity = cosine_similarity(emb1, emb2)
        return similarity
```

**Integration Point**: New service for cross-system mapping

**Benefits**:
- Automated schema alignment
- Reduced manual mapping work
- Better cross-system integration

---

## Recommended GNN Architecture

### For Node Classification & Embeddings
- **GraphSAGE**: Good for inductive learning, handles new nodes
- **GCN**: Simple and effective for node classification
- **GAT**: Attention mechanism for important relationships

### For Link Prediction
- **GAE (Graph Autoencoder)**: Learn embeddings and predict links
- **GAT**: Attention-based link prediction

### For Schema Matching
- **Graph Transformer**: Handle complex schema structures
- **GAT**: Attention for matching important schema elements

### For Anomaly Detection
- **GAE**: Reconstruction error for anomaly scoring
- **GraphSAGE**: Learn normal patterns, detect deviations

---

## Implementation Roadmap

### Phase 1: Graph Embeddings (Foundation)
1. Create `services/training/gnn_embeddings.py`
2. Implement GraphSAGE-based embedding generation
3. Integrate into Step 4 (Feature Generation)
4. Replace manual features with embeddings

### Phase 2: Node Classification
1. Add node classification model
2. Train on labeled schema data
3. Integrate domain/type classification
4. Use for automatic schema organization

### Phase 3: Link Prediction
1. Implement link prediction model
2. Train on existing relationships
3. Use for mapping discovery
4. Integrate with mapping rule agent

### Phase 4: Schema Matching
1. Create schema matching service
2. Use GNN for cross-system alignment
3. Generate mapping suggestions
4. Integrate with mapping workflows

### Phase 5: Anomaly Detection
1. Implement anomaly detection model
2. Train on normal schema patterns
3. Detect structural anomalies
4. Integrate with quality monitoring

---

## Integration with Existing Code

### Files to Modify

1. **`services/training/pipeline.py`**
   - Add GNN embedding step after extraction
   - Replace manual feature generation with embeddings
   - Add GNN-based pattern learning

2. **`services/training/pattern_learning.py`**
   - Enhance with GNN-based pattern detection
   - Use embeddings for similarity matching

3. **New Files to Create**:
   - `services/training/gnn_embeddings.py` - Embedding generation
   - `services/training/gnn_classifier.py` - Node classification
   - `services/training/gnn_link_predictor.py` - Link prediction
   - `services/training/gnn_schema_matcher.py` - Schema matching
   - `services/training/gnn_anomaly_detector.py` - Anomaly detection

### Dependencies

```python
# Required packages
torch>=2.0.0
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.0
```

---

## Data Preparation

### Graph Format

Convert existing graph data to PyTorch Geometric format:

```python
from torch_geometric.data import Data

# Convert nodes to features
node_features = extract_node_features(nodes)  # Shape: [num_nodes, num_features]

# Convert edges to edge_index
edge_index = extract_edge_index(edges)  # Shape: [2, num_edges]

# Create PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index)
```

### Node Features

Extract features from existing node properties:
- Column: data_type, nullable, default_value
- Table: column_count, primary_key_count, foreign_key_count
- View: definition_length, column_count

### Edge Features

Extract features from existing edge properties:
- Relationship type (HAS_COLUMN, REFERENCES, etc.)
- Transformation metadata
- Quality metrics

---

## Training Pipeline Integration

### Modified Pipeline Flow

```
Step 1: Extract Knowledge Graph
  ↓
Step 1a: Generate GNN Embeddings (NEW)
  ↓
Step 2: Query Glean (optional)
  ↓
Step 3: Learn Patterns (enhanced with GNN)
  ↓
Step 3a: Node Classification (NEW)
  ↓
Step 3b: Link Prediction (NEW)
  ↓
Step 3c: Anomaly Detection (NEW)
  ↓
Step 4: Generate Features (using embeddings)
  ↓
Step 5: Prepare Dataset
```

---

## Example: Node Classification Integration

```python
# In services/training/pipeline.py

def _generate_gnn_embeddings(self, graph_data):
    """Generate GNN embeddings for graph nodes."""
    from .gnn_embeddings import GraphEmbedder
    
    # Convert graph to PyTorch Geometric format
    data = self._convert_to_pyg_format(graph_data)
    
    # Initialize embedder
    embedder = GraphEmbedder(
        num_node_features=data.x.shape[1],
        embedding_dim=128
    )
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = embedder(data.x, data.edge_index)
    
    return embeddings

def _classify_nodes(self, graph_data, embeddings):
    """Classify nodes using GNN."""
    from .gnn_classifier import NodeClassifier
    
    # Load trained classifier
    classifier = NodeClassifier.load_from_checkpoint(...)
    
    # Classify nodes
    classifications = classifier.predict(embeddings)
    
    return classifications
```

---

## Benefits Summary

1. **Better Feature Representation**: GNN embeddings capture structural patterns better than manual features
2. **Automatic Discovery**: Link prediction and schema matching reduce manual work
3. **Scalability**: GNNs handle large graphs efficiently
4. **Generalization**: Models learn patterns that apply across systems
5. **Continuous Learning**: Models improve as more data is added

---

## Next Steps

1. **Start with Graph Embeddings**: Easiest to integrate, provides foundation
2. **Add Node Classification**: Useful for automatic organization
3. **Implement Link Prediction**: High value for mapping discovery
4. **Build Schema Matcher**: Critical for cross-system integration
5. **Add Anomaly Detection**: Completes the GNN integration

---

## Related Documentation

- [Training Pipeline](../services/training/pipeline.py)
- [Pattern Learning](../services/training/pattern_learning.py)
- [SAP BDC Integration](./SAP_BDC_INTEGRATION.md)

