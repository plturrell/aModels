# Phase 2 Implementation: Domain-Specific Training & Deployment

## Overview

Phase 2 builds on Phase 1 to provide:
1. Domain-specific model training workflows
2. Performance metrics collection and analytics
3. Automatic deployment triggers
4. Model version tracking
5. Performance monitoring dashboard

## Features Implemented

### 1. Domain-Specific Model Training ✅

**Location**: `services/training/domain_trainer.py`

**What it does**:
- Trains or fine-tunes models for specific domains
- Uses domain-filtered training data from Phase 1
- Supports both fine-tuning and training from scratch
- Generates training run IDs and tracks model versions
- Automatically evaluates models and checks deployment thresholds

**Usage**:
```python
from services.training import DomainTrainer

trainer = DomainTrainer(
    localai_url="http://localai:8080",
    postgres_dsn="postgres://...",
    checkpoint_dir="./checkpoints",
    model_output_dir="./models/domain_models"
)

# Train domain-specific model
results = trainer.train_domain_model(
    domain_id="0x5678-SQLAgent",
    training_data_path="./training_data/domain_filtered/sql_agent_data.json",
    fine_tune=True
)

# Results include:
# - training_run_id
# - model_path
# - evaluation metrics
# - should_deploy flag
# - deployment result (if threshold met)
```

**Configuration**:
- `ENABLE_DOMAIN_TRAINING`: Enable domain training in pipeline (default: `false`)
- `DEPLOY_ACCURACY_THRESHOLD`: Minimum accuracy for deployment (default: `0.85`)
- `DEPLOY_LATENCY_THRESHOLD`: Maximum latency in ms (default: `500`)
- `DEPLOY_LOSS_THRESHOLD`: Maximum training loss (default: `0.3`)
- `DEPLOY_VAL_LOSS_THRESHOLD`: Maximum validation loss (default: `0.35`)

### 2. Domain Performance Metrics Collection ✅

**Location**: `services/training/domain_metrics.py`

**What it does**:
- Collects comprehensive metrics per domain
- Aggregates performance history from PostgreSQL
- Calculates trends over time
- Compares metrics across domains
- Exports data for dashboard visualization

**Usage**:
```python
from services.training import DomainMetricsCollector

collector = DomainMetricsCollector(
    localai_url="http://localai:8080",
    postgres_dsn="postgres://..."
)

# Collect metrics for a domain
metrics = collector.collect_domain_metrics(
    domain_id="0x5678-SQLAgent",
    time_window_days=30
)

# Compare multiple domains
comparison = collector.get_domain_comparison([
    "0x5678-SQLAgent",
    "0x3579-VectorProcessingAgent"
])

# Export dashboard data
collector.export_metrics_dashboard(
    output_path="./dashboard_data/metrics.json",
    domain_ids=["0x5678-SQLAgent"]
)
```

**Metrics Collected**:
- Performance: accuracy, latency, training/validation loss
- Usage: request counts, query patterns
- Quality: error rates, response quality
- Trends: direction and magnitude of changes

### 3. Automatic Deployment Triggers ✅

**Location**: `services/training/auto_deploy.py`

**What it does**:
- Checks if training results meet deployment thresholds
- Automatically updates domain configurations
- Syncs to Redis for fast access
- Triggers LocalAI reload (if API available)
- Tracks deployment status and versions

**Usage**:
```python
from services.training import AutoDeploymentTrigger

trigger = AutoDeploymentTrigger(
    localai_url="http://localai:8080",
    postgres_dsn="postgres://...",
    redis_url="redis://..."
)

# Check and deploy if threshold met
result = trigger.check_and_deploy(
    domain_id="0x5678-SQLAgent",
    training_run_id="0x5678-SQLAgent_20241104_120000",
    metrics={
        "accuracy": 0.87,
        "latency_ms": 120,
        "training_loss": 0.023,
        "validation_loss": 0.028
    },
    model_path="./checkpoints/0x5678-SQLAgent/model.pt"
)

if result["deployed"]:
    print(f"✅ Model deployed: {result['model_version']}")
else:
    print(f"⚠️  Deployment skipped: {result['reason']}")
```

**Deployment Flow**:
1. Check metrics against thresholds
2. Generate model version
3. Update PostgreSQL domain_configs
4. Sync to Redis
5. Trigger LocalAI reload

### 4. Model Version Tracking ✅

**Version Format**: `{domain_id}-v{YYYYMMDD_HHMMSS}`

**Version Storage**:
- PostgreSQL: `domain_configs.model_version`
- Model files: `{domain_id}/v{version}.pt`
- Training results: `{domain_id}/{training_run_id}/results.json`

**Version History**:
- Tracked in PostgreSQL with version increment
- Performance metrics linked to each version
- Training run IDs link versions to training data

### 5. Performance Monitoring Dashboard ✅

**Location**: `services/training/domain_dashboard.py`

**What it does**:
- Web-based dashboard for domain performance
- Real-time metrics display
- Domain comparison and rankings
- Trend visualization
- REST API for metrics access

**Usage**:
```python
from services.training import DomainDashboard, DomainMetricsCollector

collector = DomainMetricsCollector()
dashboard = DomainDashboard(
    metrics_collector=collector,
    port=8085
)

# Start dashboard server
dashboard.serve()
```

**Access**:
- Dashboard: `http://localhost:8085/`
- Metrics API: `http://localhost:8085/api/metrics?domain_id={domain_id}`
- All domains: `http://localhost:8085/api/metrics`

## Integration with Training Pipeline

The training pipeline now includes:

```python
from services.training import TrainingPipeline

pipeline = TrainingPipeline(
    enable_domain_filtering=True,
    privacy_level="medium"
)

# Enable domain training
os.environ["ENABLE_DOMAIN_TRAINING"] = "true"

# Run pipeline - includes domain training and deployment
results = pipeline.run_full_pipeline(
    project_id="sgmi",
    json_tables=["data/training/sgmi/json_with_changes.json"]
)

# Check results
if results["steps"]["domain_training"]["status"] == "success":
    training_result = results["steps"]["domain_training"]
    if training_result["should_deploy"]:
        print(f"✅ Model deployed: {training_result['deployment']['model_version']}")
```

## Configuration

### Environment Variables

**Domain Training**:
```bash
ENABLE_DOMAIN_TRAINING=true          # Enable domain-specific training
DEPLOY_ACCURACY_THRESHOLD=0.85       # Minimum accuracy for deployment
DEPLOY_LATENCY_THRESHOLD=500         # Maximum latency (ms)
DEPLOY_LOSS_THRESHOLD=0.3            # Maximum training loss
DEPLOY_VAL_LOSS_THRESHOLD=0.35       # Maximum validation loss
```

**Metrics Collection**:
```bash
METRICS_TIME_WINDOW_DAYS=30          # Time window for metrics
DASHBOARD_PORT=8085                  # Dashboard server port
```

**Database**:
```bash
POSTGRES_DSN=postgres://...          # PostgreSQL connection
REDIS_URL=redis://...                # Redis connection (optional)
```

### Docker Compose

**Training Service**:
```yaml
trainer:
  environment:
    - ENABLE_DOMAIN_TRAINING=true
    - DEPLOY_ACCURACY_THRESHOLD=0.85
    - POSTGRES_DSN=postgres://postgres:postgres@postgres:5432/amodels
    - REDIS_URL=redis://redis:6379/0
```

## Deployment Workflow

### Automatic Deployment

1. **Training completes** → Domain-specific model trained
2. **Evaluation** → Metrics collected and evaluated
3. **Threshold check** → Compare metrics to deployment thresholds
4. **Auto-deploy** → If thresholds met:
   - Update domain config in PostgreSQL
   - Sync to Redis
   - Trigger LocalAI reload
5. **Version tracking** → Model version saved with metrics

### Manual Deployment

If automatic deployment is disabled or threshold not met:

```python
from services.training import DomainTrainer

trainer = DomainTrainer()

# Manually deploy a model
results = trainer.train_domain_model(
    domain_id="0x5678-SQLAgent",
    training_data_path="./data.json",
    fine_tune=True
)

# Check deployment status
if results["should_deploy"]:
    print("✅ Model ready for deployment")
else:
    print("⚠️  Model does not meet deployment threshold")
```

## Metrics Dashboard

### Features

- **Domain Cards**: Individual domain performance metrics
- **Rankings**: Cross-domain comparison
- **Trends**: Performance direction over time
- **API Access**: REST API for programmatic access

### Metrics Displayed

- Accuracy: Model prediction accuracy
- Latency: Response time in milliseconds
- Training Loss: Training loss value
- Validation Loss: Validation loss value
- Trends: Improving, degrading, or stable

### Dashboard Access

```bash
# Start dashboard
python -m services.training.domain_dashboard

# Access in browser
open http://localhost:8085
```

## Best Practices

1. **Threshold Configuration**:
   - Set realistic thresholds based on domain requirements
   - Adjust thresholds per domain if needed
   - Monitor threshold hit rates

2. **Model Versioning**:
   - Use semantic versioning for production
   - Keep version history in PostgreSQL
   - Link versions to training runs

3. **Metrics Collection**:
   - Collect metrics regularly (daily/weekly)
   - Track trends over time
   - Compare across domains

4. **Deployment**:
   - Use automatic deployment for development
   - Manual review for production
   - Monitor deployment success rates

## Testing

### Test Domain Training

```python
from services.training import DomainTrainer

trainer = DomainTrainer()

# Test training
results = trainer.train_domain_model(
    domain_id="test-domain",
    training_data_path="./test_data.json",
    fine_tune=True
)

assert results["training_run_id"] is not None
assert results["model_path"] is not None
assert "evaluation" in results
```

### Test Metrics Collection

```python
from services.training import DomainMetricsCollector

collector = DomainMetricsCollector()

# Test metrics collection
metrics = collector.collect_domain_metrics("test-domain", time_window_days=7)

assert "performance" in metrics
assert "trends" in metrics
```

### Test Deployment

```python
from services.training import AutoDeploymentTrigger

trigger = AutoDeploymentTrigger()

# Test deployment check
result = trigger.check_and_deploy(
    domain_id="test-domain",
    training_run_id="test-run",
    metrics={"accuracy": 0.9, "latency_ms": 100},
    model_path="./test_model.pt"
)

assert "deployed" in result
```

## Future Enhancements

1. **A/B Testing**: Deploy multiple model versions and compare
2. **Rollback**: Automatic rollback if performance degrades
3. **Canary Deployment**: Gradual rollout to percentage of requests
4. **Advanced Analytics**: ML-based performance prediction
5. **Alerting**: Notifications when metrics cross thresholds

## References

- Phase 1 Documentation: `docs/domain-configuration-phase1-dp-integration.md`
- Domain Configuration Review: `docs/domain-configuration-process-review.md`
- Training Pipeline: `services/training/pipeline.py`

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-04  
**Phase**: Phase 2 Implementation Complete

