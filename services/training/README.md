# Training Service

This service provides training pipeline integration with Glean Catalog and pattern learning capabilities.

## Features

- **Glean Catalog Integration**: Query historical data from Glean Catalog for training
- **Temporal Pattern Analysis**: Analyze information theory metrics over time
- **Column Type Pattern Learning**: Learn column type distribution patterns and transitions
- **Relationship Pattern Learning**: Learn patterns in table relationships and data flow
- **Metadata Entropy Pattern Learning**: Learn patterns in information theory metrics
- **Training Data Ingestion**: Transform Glean data into training format
- **Pattern Prediction**: Predict schema evolution and relationship types

## Usage

### Basic Glean Integration

```python
from services.training import GleanTrainingClient, ingest_glean_data_for_training

# Query historical data
client = GleanTrainingClient()
nodes = client.query_historical_nodes(project_id="sgmi", days_back=30)
edges = client.query_historical_edges(project_id="sgmi", days_back=30)

# Get information theory metrics over time
metrics = client.query_information_theory_metrics(project_id="sgmi", days_back=30)

# Get column type patterns
patterns = client.query_column_type_patterns(project_id="sgmi", days_back=30)
```

### Training Data Ingestion

```python
# Ingest Glean data for training
glean_data = ingest_glean_data_for_training(
    project_id="sgmi",
    system_id="sgmi-system",
    days_back=30,
    output_dir="./training_data/glean"
)

# Access ingested data
nodes = glean_data["nodes"]
edges = glean_data["edges"]
metrics = glean_data["metrics"]
column_patterns = glean_data["column_patterns"]
```

### Pattern Learning

```python
from services.training import PatternLearningEngine

# Learn patterns from knowledge graph and Glean data
engine = PatternLearningEngine()
patterns = engine.learn_patterns(
    nodes=nodes,
    edges=edges,
    metrics=metrics,
    glean_data=glean_data
)

# Predict schema evolution
predictions = engine.predict_schema_evolution("VARCHAR", context={"table": "users"})

# Predict relationship types
rel_predictions = engine.predict_relationship_type("table", "column")
```

### Integration with Training Scripts

```bash
# Enable Glean integration in training
python tools/scripts/train_relational_transformer.py \
    --config config.yaml \
    --mode pretrain \
    --glean-enable \
    --glean-project-id sgmi \
    --glean-system-id sgmi-system \
    --glean-days-back 30 \
    --glean-output-dir ./training_data/glean
```

### Extract Service Integration

```python
from services.training import ExtractServiceClient

# Get knowledge graph from Extract service
client = ExtractServiceClient()
graph_data = client.get_knowledge_graph(
    project_id="sgmi",
    system_id="sgmi-system",
    json_tables=["data/training/sgmi/json_with_changes.json"],
    hive_ddls=["data/training/sgmi/hive-ddl/sgmisit_all_tables_statement.hql"],
    control_m_files=["data/training/sgmi/sgmi-controlm/catalyst migration prod 640.xml"]
)

# Query knowledge graph with Cypher
results = client.query_knowledge_graph(
    query="MATCH (n:Table) RETURN n LIMIT 10"
)
```

### Training Pipeline

```python
from services.training import TrainingPipeline

# Run complete training pipeline
pipeline = TrainingPipeline()
results = pipeline.run_full_pipeline(
    project_id="sgmi",
    system_id="sgmi-system",
    json_tables=["data/training/sgmi/json_with_changes.json"],
    hive_ddls=["data/training/sgmi/hive-ddl/sgmisit_all_tables_statement.hql"],
    control_m_files=["data/training/sgmi/sgmi-controlm/catalyst migration prod 640.xml"],
    glean_days_back=30,
    enable_glean=True
)
```

### Temporal Pattern Analysis

Temporal pattern analysis integrates with **all timestamping systems** in the aModels platform:

- **Postgres**: Queries `updated_at_utc` from `glean_nodes` and `glean_edges` tables
- **Neo4j**: Queries nodes/edges with `updated_at` timestamps and `metrics_calculated_at` properties
- **Glean Catalog**: Uses `exported_at` from export manifests for historical trends
- **json_with_changes.json**: Parses historical change data with timestamps

```python
from services.training import TemporalPatternLearner, SchemaEvolutionAnalyzer
from services.training import ExtractServiceClient, GleanTrainingClient

# Initialize clients for full temporal integration
extract_client = ExtractServiceClient(extract_service_url="http://localhost:19080")
glean_client = GleanTrainingClient(db_name="glean_db")

# Analyze schema evolution from change history
analyzer = SchemaEvolutionAnalyzer()
with open("data/training/sgmi/json_with_changes.json", 'r') as f:
    json_with_changes = json.load(f)

evolution_analysis = analyzer.analyze_change_history(
    json_with_changes,
    project_id="sgmi",
    system_id="sgmi-system"
)

# Learn temporal patterns from ALL sources (Postgres, Neo4j, Glean, JSON)
learner = TemporalPatternLearner(
    extract_client=extract_client,  # For Neo4j queries
    glean_client=glean_client,       # For Glean queries
    postgres_dsn=os.getenv("POSTGRES_CATALOG_DSN")  # For Postgres queries
)

temporal_patterns = learner.learn_temporal_patterns(
    json_with_changes=json_with_changes,
    glean_metrics=glean_metrics,
    project_id="sgmi",
    system_id="sgmi-system"
)

# Temporal patterns now include:
# - evolution_patterns: From json_with_changes.json
# - temporal_metrics: From Glean export manifests
# - postgres_temporal: From Postgres updated_at_utc queries
# - neo4j_temporal: From Neo4j updated_at and metrics_calculated_at
# - combined_insights: Unified insights from all sources

# Predict future changes
predictions = analyzer.predict_future_changes(
    current_schema=current_schema,
    analysis=evolution_analysis
)
```

### Training Evaluation and Metrics Export

Enhanced evaluation with pattern-specific metrics and ARC-AGI style intelligence evaluation:

```python
from services.training import (
    evaluate_training_results,
    export_training_metrics_to_glean,
    DomainIntelligenceEvaluator
)

# Create intelligence evaluator and test cases
evaluator = DomainIntelligenceEvaluator()
test_cases = evaluator.create_domain_test_cases(
    knowledge_graph=graph_data,
    learned_patterns=learned_patterns
)

# Evaluate training results with intelligence metrics
evaluation = evaluate_training_results(
    model_metrics={"loss": 0.25, "accuracy": 0.85},
    training_context=training_context,
    checkpoint_path="./checkpoints/model.pt",
    model_predictions=model_predictions,
    test_cases=test_cases,
    enable_intelligence_metrics=True
)

# Access intelligence metrics
intelligence = evaluation["intelligence_metrics"]
print(f"Intelligence Level: {intelligence['intelligence_level']}/5")
print(f"Domain Expertise: {intelligence['domain_expertise']:.2f}")
print(f"Learning Rate: {evaluation['learning_rate']['patterns_per_epoch']:.2f} patterns/epoch")

# Pattern-specific metrics
pattern_metrics = evaluation["pattern_specific_metrics"]
print(f"Column Type Accuracy: {pattern_metrics['column_type_accuracy']:.2f}")
print(f"Relationship Accuracy: {pattern_metrics['relationship_accuracy']:.2f}")

# Export to Glean
from services.training import GleanTrainingClient
glean_client = GleanTrainingClient()
export_info = export_training_metrics_to_glean(
    evaluation=evaluation,
    glean_client=glean_client,
    output_dir="./training_metrics"
)
```

**Intelligence Levels (ARC-AGI Style):**
- **Level 1: Pattern Recognition** - Can recognize basic patterns in data
- **Level 2: Pattern Generalization** - Can generalize to unseen patterns
- **Level 3: Compositional Reasoning** - Can compose patterns to solve new problems
- **Level 4: Transfer Learning** - Can transfer knowledge across domains
- **Level 5: Abstract Reasoning** - Can reason about abstract concepts

**Learning Rate Metrics:**
- `patterns_per_epoch`: Average patterns learned per training epoch
- `convergence_rate`: How quickly model converges (0-1)
- `adaptation_rate`: How quickly model adapts to new data

## Configuration

Environment variables:

- `GLEAN_DB_NAME`: Glean database name (default: `amodels`)
- `GLEAN_SCHEMA_PATH`: Path to Glean schema file
- `GLEAN_QUERY_API_URL`: REST API endpoint for Glean queries (optional)
- `GLEAN_USE_CLI`: Use Glean CLI (default: `true`)

## API Reference

### GleanTrainingClient

#### `query_historical_nodes(project_id, system_id, days_back, limit)`
Query historical node data from Glean Catalog.

#### `query_historical_edges(project_id, system_id, days_back, limit)`
Query historical edge data from Glean Catalog.

#### `query_export_manifests(project_id, system_id, days_back, limit)`
Query historical export manifests for training metadata.

#### `query_information_theory_metrics(project_id, system_id, days_back)`
Query information theory metrics over time for pattern analysis.

Returns:
- `metadata_entropy_trend`: List of (date, entropy) tuples
- `kl_divergence_trend`: List of (date, kl_div) tuples
- `column_count_trend`: List of (date, count) tuples
- `averages`: Average values over time period

#### `query_column_type_patterns(project_id, system_id, days_back)`
Query column type distribution patterns over time.

Returns:
- `type_distributions`: List of (date, distribution) tuples
- `common_types`: Most common column types
- `type_transitions`: Patterns of type changes

### ingest_glean_data_for_training()

Ingest Glean data into training pipeline format.

Returns dictionary with:
- `nodes`: List of historical nodes
- `edges`: List of historical edges
- `metrics`: Information theory metrics over time
- `column_patterns`: Column type distribution patterns
- `output_files`: List of generated output files (if output_dir provided)

### PatternLearningEngine

Main engine for learning patterns from knowledge graphs and Glean data.

#### `learn_patterns(nodes, edges, metrics, glean_data)`
Learn all patterns from knowledge graph and Glean data.

Returns dictionary with:
- `column_patterns`: Column type patterns and transitions
- `relationship_patterns`: Relationship patterns and chains
- `metrics_patterns`: Metadata entropy patterns and trends
- `summary`: Summary of learned patterns

#### `predict_schema_evolution(current_type, context)`
Predict likely schema evolution patterns.

Returns list of (predicted_type, probability) tuples.

#### `predict_relationship_type(source_type, target_type)`
Predict likely relationship type between nodes.

Returns list of (edge_label, probability) tuples.

### ColumnTypePatternLearner

Learns patterns in column type distributions and transitions.

### RelationshipPatternLearner

Learns patterns in table relationships and data flow.

### MetadataEntropyPatternLearner

Learns patterns in metadata entropy and information theory metrics.

### TemporalPatternLearner

Learns temporal patterns from schema evolution and Glean historical data.

#### `learn_temporal_patterns(json_with_changes, glean_metrics, project_id, system_id)`
Learn temporal patterns from change history and Glean metrics.

Returns dictionary with:
- `evolution_patterns`: Patterns from change history
- `temporal_metrics`: Patterns from Glean metrics
- `combined_insights`: Combined insights from both sources

### SchemaEvolutionAnalyzer

Analyzes schema evolution patterns from change history.

#### `analyze_change_history(json_with_changes, project_id, system_id)`
Analyze schema change history from json_with_changes.json.

Returns dictionary with:
- `change_patterns`: Patterns of changes over time
- `temporal_sequences`: Sequences of changes
- `evolution_statistics`: Statistics about evolution

#### `predict_future_changes(current_schema, analysis)`
Predict likely future schema changes based on learned patterns.

Returns list of predicted changes with probabilities.

### ExtractServiceClient

Client for interacting with the Extract service.

#### `get_knowledge_graph(project_id, system_id, json_tables, hive_ddls, control_m_files)`
Get knowledge graph from Extract service.

Returns dictionary with nodes, edges, metrics, and quality assessment.

#### `query_knowledge_graph(query, params)`
Execute a Cypher query against the Neo4j knowledge graph.

Returns query results with columns and data.

#### `health_check()`
Check if Extract service is available.

### Training Evaluation

#### `evaluate_training_results(model_metrics, training_context, checkpoint_path)`
Evaluate training results and generate evaluation metrics.

Returns dictionary with:
- `model_metrics`: Original model metrics
- `training_quality`: Quality assessment of training
- `pattern_coverage`: How well patterns were learned
- `recommendations`: Recommendations for improvement

#### `export_training_metrics_to_glean(evaluation, glean_client, output_dir)`
Export training evaluation metrics to Glean Catalog.

Returns dictionary with export information.

