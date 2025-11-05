# Phase 1 Implementation: Domain Configuration with Differential Privacy

## Overview

This document describes the Phase 1 implementation of domain-specific training data filtering and domain config updates with differential privacy integration.

## Features Implemented

### 1. Domain Association During Extraction ✅

**Location**: `services/extract/domain_detector.go`, `services/extract/main.go`, `services/extract/neo4j.go`

**What it does**:
- Loads domain configurations from LocalAI on extraction service startup
- Detects appropriate domains for extracted data using keyword matching
- Associates `agent_id` and `domain` properties with nodes and edges
- Stores domain metadata in Neo4j as separate properties for efficient querying

**How it works**:
1. `DomainDetector` loads domains from LocalAI `/v1/domains` endpoint
2. During extraction, nodes/edges are analyzed for domain keywords
3. Best matching domain is selected based on keyword/tag scores
4. `agent_id` and `domain` are stored in Neo4j node/edge properties

**Configuration**:
- `LOCALAI_URL`: URL of LocalAI service (default: `http://localai:8080`)

### 2. Domain-Specific Training Data Filtering ✅

**Location**: `services/training/domain_filter.py`, `services/training/pipeline.py`

**What it does**:
- Filters training data by domain with differential privacy protection
- Applies Laplacian noise to numeric values to protect sensitive information
- Tracks privacy budget per domain to prevent over-querying
- Auto-detects domain from extracted data if not specified

**Differential Privacy**:
- Uses Laplacian mechanism: `Lap(sensitivity / ε)`
- Configurable privacy levels: `low` (ε=2.0), `medium` (ε=1.0), `high` (ε=0.5)
- Privacy budget tracking per domain (max 100 queries per domain)
- Noise applied to:
  - Numeric node/edge properties
  - Distribution values
  - Feature data
  - Performance metrics

**Usage**:
```python
from services.training import TrainingPipeline

# Initialize with domain filtering and privacy
pipeline = TrainingPipeline(
    enable_domain_filtering=True,
    privacy_level="medium"  # or "low", "high"
)

# Run pipeline - domain filtering applied automatically
results = pipeline.run_full_pipeline(
    project_id="sgmi",
    json_tables=["data/training/sgmi/json_with_changes.json"],
    hive_ddls=["data/training/sgmi/hive-ddl/sgmisit_all_tables_statement.hql"]
)
```

**Privacy Configuration**:
- `PRIVACY_LEVEL`: `low`, `medium`, or `high` (default: `medium`)
- `PRIVACY_EPSILON`: Override epsilon value (optional)
- `PRIVACY_DELTA`: Override delta value (optional, default: 1e-5)

### 3. Automated Domain Config Updates ✅

**Location**: `services/localai/scripts/load_domains_from_training.py`

**What it does**:
- Automatically updates domain configurations in PostgreSQL after training
- Applies differential privacy to performance metrics before storage
- Links domain configs to training runs via `training_run_id`
- Tracks model versions and performance metrics

**Differential Privacy for Metrics**:
- Laplacian noise added to performance metrics:
  - Accuracy: 1% sensitivity
  - Latency: 10ms sensitivity
  - Tokens/second: 1 token/s sensitivity
  - Loss values: 0.1% sensitivity
- Configurable via environment variables

**Usage**:
```bash
# Set privacy configuration
export APPLY_DIFFERENTIAL_PRIVACY=true
export PRIVACY_EPSILON=1.0

# Run after training completes
python services/localai/scripts/load_domains_from_training.py
```

**Configuration**:
- `APPLY_DIFFERENTIAL_PRIVACY`: Enable/disable privacy (default: `true`)
- `PRIVACY_EPSILON`: Privacy budget ε (default: `1.0`)
- `POSTGRES_DSN`: PostgreSQL connection string
- `TRAINING_RUN_ID`: Training run identifier
- `MODEL_VERSION`: Model version identifier

## Privacy Guarantees

### Differential Privacy Definition

Differential privacy provides a mathematical guarantee that the output of a query does not reveal information about any individual data point. The privacy loss is quantified by:

- **ε (epsilon)**: Privacy budget - smaller values mean stronger privacy
- **δ (delta)**: Failure probability - typically set to 1e-5 or smaller

### Privacy Levels

| Level | ε (epsilon) | δ (delta) | Noise Scale | Use Case |
|-------|-------------|-----------|-------------|----------|
| Low | 2.0 | 1e-4 | 0.05 | Development/testing |
| Medium | 1.0 | 1e-5 | 0.1 | Production (default) |
| High | 0.5 | 1e-6 | 0.2 | Sensitive data |

### Privacy Budget Tracking

Each domain has a privacy budget that limits the number of queries:
- **Max queries per domain**: 100 (configurable)
- **Budget consumption**: 1.0 / max_queries per query
- **Budget reset**: Manual (requires restart or configuration update)

### Noise Addition

**Laplacian Mechanism**:
```
noise = Lap(sensitivity / ε)
private_value = original_value + noise
```

**Sensitivity Values**:
- Node/edge counts: 1.0
- Distribution values: 1.0 (normalized after noise)
- Accuracy: 0.01 (1%)
- Latency: 10.0 (10ms)
- Tokens/second: 1.0
- Loss values: 0.001 (0.1%)

## Integration Points

### 1. Extraction → Domain Association

```
Extract Service
    ↓
DomainDetector.LoadDomains() (from LocalAI)
    ↓
DomainDetector.AssociateDomainsWithNodes()
DomainDetector.AssociateDomainsWithEdges()
    ↓
Neo4j.SaveGraph() (with agent_id and domain properties)
```

### 2. Training → Domain Filtering

```
Training Pipeline
    ↓
_extract_knowledge_graph() (with domain filtering)
    ↓
DomainFilter.filter_by_domain() (with DP)
    ↓
_generate_training_features()
    ↓
DomainFilter.filter_features_by_domain() (with DP)
    ↓
Training dataset (domain-filtered, privacy-protected)
```

### 3. Training → Domain Config Update

```
Training completes
    ↓
load_domains_from_training.py
    ↓
apply_privacy_to_metrics() (Laplacian noise)
    ↓
PostgreSQL domain_configs table
    ↓
Redis sync (if enabled)
```

## Configuration

### Environment Variables

**Extract Service**:
```bash
LOCALAI_URL=http://localai:8080  # LocalAI service URL
```

**Training Service**:
```bash
ENABLE_DOMAIN_FILTERING=true     # Enable domain filtering
PRIVACY_LEVEL=medium             # Privacy level: low, medium, high
PRIVACY_EPSILON=1.0              # Override epsilon (optional)
PRIVACY_DELTA=1e-5               # Override delta (optional)
LOCALAI_URL=http://localai:8080 # LocalAI service URL
```

**Domain Config Updates**:
```bash
APPLY_DIFFERENTIAL_PRIVACY=true  # Enable privacy for metrics
PRIVACY_EPSILON=1.0              # Privacy budget
POSTGRES_DSN=postgres://...      # PostgreSQL connection
TRAINING_RUN_ID=run_001          # Training run ID
MODEL_VERSION=v1                 # Model version
```

### Docker Compose

**Extract Service** (already configured):
```yaml
extract:
  environment:
    - LOCALAI_URL=${LOCALAI_URL:-http://localai:8080}
```

**Training Service** (add to docker-compose):
```yaml
trainer:
  environment:
    - ENABLE_DOMAIN_FILTERING=true
    - PRIVACY_LEVEL=medium
    - LOCALAI_URL=http://localai:8080
```

## Privacy Audit

### Privacy Stats

The domain filter provides privacy statistics:
```python
stats = domain_filter.get_privacy_stats()
# Returns:
# {
#   "domains": {
#     "domain_id": {
#       "queries_used": 10,
#       "queries_remaining": 90,
#       "budget_utilization": 10.0
#     }
#   },
#   "privacy_config": {
#     "epsilon": 1.0,
#     "delta": 1e-5,
#     "noise_scale": 0.1,
#     "privacy_level": "medium"
#   }
# }
```

### Privacy Budget Monitoring

Monitor privacy budget usage:
- Check `privacy_stats` in training results
- Log privacy budget warnings when approaching limits
- Track queries per domain in domain filter

## Best Practices

1. **Privacy Level Selection**:
   - Use `low` for development/testing
   - Use `medium` for production (default)
   - Use `high` for sensitive data

2. **Privacy Budget Management**:
   - Monitor budget usage regularly
   - Reset budgets periodically if needed
   - Adjust `max_queries` per domain based on usage

3. **Sensitivity Configuration**:
   - Adjust sensitivity values based on data characteristics
   - Use domain-specific sensitivity if needed
   - Validate noise addition doesn't break model training

4. **Domain Filtering**:
   - Enable domain filtering for production
   - Auto-detect domain when possible
   - Manually specify domain for critical training runs

## Testing

### Test Domain Filtering

```python
from services.training.domain_filter import DomainFilter, PrivacyConfig

# Initialize filter
config = PrivacyConfig(privacy_level="medium")
filter = DomainFilter(localai_url="http://localai:8080", privacy_config=config)

# Filter nodes/edges
nodes = [{"id": "n1", "properties": {"domain": "0x5678-SQLAgent"}}]
edges = [{"source": "n1", "target": "n2", "properties": {"domain": "0x5678-SQLAgent"}}]

filtered_nodes, filtered_edges = filter.filter_by_domain(nodes, edges, domain_id="0x5678-SQLAgent")
```

### Test Privacy

```python
# Check privacy stats
stats = filter.get_privacy_stats()
assert stats["domains"]["0x5678-SQLAgent"]["queries_used"] == 1

# Verify noise addition
original_value = 0.85
noisy_value = filter._add_noise(original_value)
assert abs(noisy_value - original_value) < 0.5  # Noise should be reasonable
```

## Future Enhancements

1. **Adaptive Privacy Budgets**: Automatically adjust budgets based on domain usage
2. **Compositional Privacy**: Track privacy loss across multiple queries
3. **Domain-Specific Sensitivity**: Different sensitivity values per domain
4. **Privacy Auditing**: Comprehensive audit logs for privacy operations
5. **Automatic Budget Reset**: Scheduled budget resets based on time periods

## References

- [Differential Privacy Explained](https://en.wikipedia.org/wiki/Differential_privacy)
- [Laplacian Mechanism](https://en.wikipedia.org/wiki/Laplace_distribution)
- [Privacy Budget Management](https://www.microsoft.com/en-us/research/publication/the-algorithmic-foundations-of-differential-privacy/)

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-04  
**Phase**: Phase 1 Implementation Complete

