# Phase 3 Implementation: Advanced Domain Management & Optimization

## Overview

Phase 3 builds on Phases 1 and 2 to provide:
1. Advanced routing optimization with learning
2. Domain lifecycle management (create, update, archive, delete)
3. A/B testing for domain models
4. Automatic rollback on performance degradation
5. Domain-specific optimizations (caching, batching)
6. Routing analytics and insights

## Features Implemented

### 1. Advanced Routing Optimization with Learning ✅

**Location**: `services/training/routing_optimizer.py`

**What it does**:
- Learns optimal routing decisions from performance feedback
- Adjusts routing weights based on actual performance metrics
- Combines base routing scores with learned weights
- Tracks routing decision history and performance

**Usage**:
```python
from services.training import RoutingOptimizer

optimizer = RoutingOptimizer(
    postgres_dsn="postgres://...",
    learning_rate=0.1
)

# Record routing decision and outcome
optimizer.record_routing_decision(
    domain_id="0x5678-SQLAgent",
    query="SELECT * FROM users",
    decision_confidence=0.85,
    actual_metrics={
        "accuracy": 0.92,
        "latency_ms": 120
    }
)

# Optimize weights
weights = optimizer.optimize_routing_weights(["0x5678-SQLAgent"])

# Get optimal domain
optimal = optimizer.get_optimal_domain(
    candidate_domains=["0x5678-SQLAgent", "0x3579-VectorProcessingAgent"],
    query="SELECT * FROM users",
    base_scores={"0x5678-SQLAgent": 0.8, "0x3579-VectorProcessingAgent": 0.6}
)
```

**Configuration**:
- `ROUTING_LEARNING_RATE`: Learning rate for weight updates (default: `0.1`)

### 2. Domain Lifecycle Management ✅

**Location**: `services/localai/pkg/domain/lifecycle_manager.go`, `api.go`

**What it does**:
- Create new domains programmatically
- Update existing domain configurations
- Archive domains (safe removal while keeping history)
- Delete domains (permanent removal with force flag)
- List all domains with status

**Usage**:
```bash
# Create domain
curl -X POST http://localai:8080/v1/domains/create \
  -H "Content-Type: application/json" \
  -d '{
    "domain_id": "0x9999-NewAgent",
    "config": {
      "name": "New Agent",
      "agent_id": "0x9999-NewAgent",
      "layer": "layer1",
      "team": "DataTeam",
      "model_path": "/models/new-model.gguf",
      "keywords": ["new", "agent"],
      "max_tokens": 2048
    }
  }'

# Update domain
curl -X PUT http://localai:8080/v1/domains/0x9999-NewAgent \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "max_tokens": 4096
    }
  }'

# Archive domain
curl -X POST http://localai:8080/v1/domains/0x9999-NewAgent/archive \
  -H "Content-Type: application/json" \
  -d '{"reason": "Replaced by newer model"}'

# List domains
curl http://localai:8080/v1/domains/list
```

**Python API**:
```python
from services.localai.pkg.domain import lifecycle_manager

lm = lifecycle_manager.NewLifecycleManager(...)

# Create
lm.CreateDomain(ctx, "0x9999-NewAgent", config, metadata)

# Update
lm.UpdateDomain(ctx, "0x9999-NewAgent", updated_config, metadata)

# Archive
lm.ArchiveDomain(ctx, "0x9999-NewAgent", "Replaced by newer model")

# List
statuses = lm.ListDomains(ctx)
```

### 3. A/B Testing for Domain Models ✅

**Location**: `services/training/ab_testing.py`

**What it does**:
- Create A/B tests comparing two model variants
- Route traffic between variants using consistent hashing
- Track metrics per variant (accuracy, latency, loss)
- Determine winner with statistical significance
- Automatically deploy winning variant

**Usage**:
```python
from services.training import ABTestManager

ab_manager = ABTestManager(
    postgres_dsn="postgres://...",
    redis_url="redis://..."
)

# Create A/B test
ab_test = ab_manager.create_ab_test(
    domain_id="0x5678-SQLAgent",
    variant_a={
        "model_path": "/models/sql-agent-v1.pt",
        "model_version": "v1.0.0"
    },
    variant_b={
        "model_path": "/models/sql-agent-v2.pt",
        "model_version": "v2.0.0"
    },
    traffic_split=0.5,  # 50% to variant B
    duration_days=7
)

# Route request
variant, config = ab_manager.route_request(
    domain_id="0x5678-SQLAgent",
    request_id="req_123"
)

# Record metrics
ab_manager.record_metric(
    test_id=ab_test["test_id"],
    variant="variant_b",
    metric_name="accuracy",
    value=0.92
)

# Get results
results = ab_manager.get_ab_test_results(ab_test["test_id"])

# Conclude test and deploy winner
conclusion = ab_manager.conclude_ab_test(
    test_id=ab_test["test_id"],
    deploy_winner=True
)
```

**Configuration**:
- A/B tests stored in PostgreSQL `ab_tests` table
- Traffic split: 0.0-1.0 (percentage to variant B)
- Statistical significance: Simplified t-test (production would use proper tests)

### 4. Automatic Rollback on Performance Degradation ✅

**Location**: `services/training/rollback_manager.py`

**What it does**:
- Monitors performance metrics after deployment
- Compares current metrics to baseline (previous version)
- Automatically rolls back if degradation detected
- Logs rollback events for audit trail
- Configurable rollback thresholds

**Usage**:
```python
from services.training import RollbackManager

rollback_manager = RollbackManager(
    postgres_dsn="postgres://...",
    redis_url="redis://...",
    localai_url="http://localai:8080"
)

# Check and rollback if needed
result = rollback_manager.check_and_rollback(
    domain_id="0x5678-SQLAgent",
    current_metrics={
        "accuracy": 0.80,  # 5% drop from 0.85 baseline
        "latency_ms": 600,  # 1.5x increase from 400ms baseline
        "error_rate": 0.05
    }
)

if result["rollback_triggered"]:
    print(f"⚠️  Rolled back: {result['reason']}")
    print(f"Rollback version: {result['rollback_result']['rollback_version']}")

# Get rollback history
history = rollback_manager.get_rollback_history(
    domain_id="0x5678-SQLAgent",
    limit=10
)
```

**Rollback Thresholds**:
- `accuracy_degradation`: 5% drop triggers rollback
- `latency_increase`: 1.5x increase triggers rollback
- `error_rate_increase`: 10% error rate triggers rollback
- `min_samples`: 50 minimum samples before rollback

**Configuration**:
```python
rollback_manager.rollback_thresholds = {
    "accuracy_degradation": 0.05,
    "latency_increase": 1.5,
    "error_rate_increase": 0.1,
    "min_samples": 50,
}
```

### 5. Domain-Specific Optimizations ✅

**Location**: `services/training/domain_optimizer.py`

**What it does**:
- Query response caching per domain
- Request batching for improved throughput
- Domain-specific optimization configurations
- Cache TTL management
- Batch queue management

**Usage**:
```python
from services.training import DomainOptimizer

optimizer = DomainOptimizer(
    redis_url="redis://...",
    cache_ttl=3600  # 1 hour
)

# Configure domain optimizations
optimizer.configure_domain_optimizations(
    domain_id="0x5678-SQLAgent",
    config={
        "cache_enabled": True,
        "cache_ttl": 3600,
        "batch_enabled": True,
        "batch_size": 10,
        "batch_timeout": 5  # seconds
    }
)

# Check cache
cached = optimizer.get_cached_response(
    domain_id="0x5678-SQLAgent",
    query="SELECT * FROM users"
)

if cached:
    return cached  # Cache hit

# Process request and cache response
response = process_request(query)
optimizer.cache_response(
    domain_id="0x5678-SQLAgent",
    query="SELECT * FROM users",
    response=response,
    ttl=3600
)

# Batch requests
if optimizer.add_to_batch(domain_id, request):
    # Request added to batch, wait for more
    pass
else:
    # Batch ready, process
    batch = optimizer.get_batch(domain_id)
    process_batch(batch)
```

**Optimization Features**:
- **Caching**: Redis-backed with in-memory fallback
- **Batching**: Configurable batch size and timeout
- **Per-domain config**: Each domain can have different optimization settings

### 6. Routing Analytics and Insights ✅

**Location**: `services/training/routing_optimizer.py`

**What it does**:
- Track routing decision history
- Analyze performance per domain
- Calculate routing confidence statistics
- Provide insights for routing optimization

**Usage**:
```python
from services.training import RoutingOptimizer

optimizer = RoutingOptimizer()

# Get routing analytics
analytics = optimizer.get_routing_analytics(domain_id="0x5678-SQLAgent")

# Returns:
# {
#   "domains": {
#     "0x5678-SQLAgent": {
#       "routing_weight": 0.85,
#       "decision_count": 1250,
#       "average_accuracy": 0.92,
#       "average_latency": 120.5,
#       "average_confidence": 0.88
#     }
#   },
#   "overall": {
#     "total_decisions": 1250,
#     "average_confidence": 0.88
#   }
# }
```

## Integration with Training Pipeline

The training pipeline now includes:

```python
from services.training import TrainingPipeline

pipeline = TrainingPipeline()

# Phase 3 components are automatically initialized:
# - ab_test_manager: For A/B testing
# - rollback_manager: For automatic rollback
# - routing_optimizer: For routing optimization
# - domain_optimizer: For domain-specific optimizations

# Run pipeline - includes rollback checking
results = pipeline.run_full_pipeline(...)

# Check rollback results
if results["steps"]["rollback_check"]["rollback_triggered"]:
    print(f"⚠️  Rollback triggered: {results['steps']['rollback_check']['reason']}")
```

## Configuration

### Environment Variables

**Routing Optimization**:
```bash
ROUTING_LEARNING_RATE=0.1          # Learning rate for routing weights
```

**Domain Optimizations**:
```bash
DOMAIN_CACHE_TTL=3600               # Cache TTL in seconds
REDIS_URL=redis://...               # Redis for caching
```

**A/B Testing**:
```bash
POSTGRES_DSN=postgres://...         # PostgreSQL for A/B test storage
REDIS_URL=redis://...               # Redis for traffic splitting
```

**Rollback**:
```bash
POSTGRES_DSN=postgres://...         # PostgreSQL for rollback tracking
LOCALAI_URL=http://localai:8080     # LocalAI for reload
```

### Database Schema

**A/B Tests Table**:
```sql
CREATE TABLE ab_tests (
    test_id VARCHAR(255) PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    test_config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    start_date TIMESTAMP DEFAULT NOW(),
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Rollback Events Table**:
```sql
CREATE TABLE rollback_events (
    id SERIAL PRIMARY KEY,
    domain_id VARCHAR(255) NOT NULL,
    reason TEXT,
    from_version VARCHAR(255),
    to_version VARCHAR(255),
    current_metrics JSONB,
    baseline_metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Routing Weights Table**:
```sql
CREATE TABLE routing_weights (
    domain_id VARCHAR(255) PRIMARY KEY,
    weight FLOAT NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

## Best Practices

### A/B Testing

1. **Test Duration**: Run tests for at least 7 days to get sufficient samples
2. **Traffic Split**: Start with 50/50 split, adjust based on risk tolerance
3. **Metrics**: Track accuracy, latency, and error rates
4. **Statistical Significance**: Ensure minimum sample size (30+ per variant)
5. **Deployment**: Automatically deploy winner if statistically significant

### Rollback Management

1. **Thresholds**: Set conservative thresholds initially, adjust based on experience
2. **Monitoring**: Monitor metrics closely after deployment
3. **Baseline**: Ensure baseline metrics are reliable before deployment
4. **History**: Review rollback history regularly to identify patterns
5. **Alerts**: Set up alerts for rollback events

### Routing Optimization

1. **Learning Rate**: Start with low learning rate (0.1), adjust based on performance
2. **History**: Keep sufficient history (1000+ decisions) for reliable weights
3. **Combination**: Use 60/40 split between base scores and learned weights
4. **Monitoring**: Track routing analytics regularly
5. **Updates**: Update weights periodically, not on every request

### Domain Optimizations

1. **Caching**: Enable caching for queries with deterministic results
2. **TTL**: Set appropriate TTL based on data freshness requirements
3. **Batching**: Use batching for high-throughput domains
4. **Configuration**: Configure optimizations per domain based on workload
5. **Monitoring**: Track cache hit rates and batch efficiency

## API Endpoints

### Domain Lifecycle Management

- `POST /v1/domains/create` - Create new domain
- `PUT /v1/domains/{domain_id}` - Update domain
- `POST /v1/domains/{domain_id}/archive` - Archive domain
- `DELETE /v1/domains/{domain_id}?force=true` - Delete domain
- `GET /v1/domains/list` - List all domains with status

## Testing

### Test A/B Testing

```python
from services.training import ABTestManager

ab_manager = ABTestManager()

# Create test
test = ab_manager.create_ab_test(
    domain_id="test-domain",
    variant_a={"model_path": "/models/v1.pt"},
    variant_b={"model_path": "/models/v2.pt"},
    traffic_split=0.5
)

# Test routing
variant, config = ab_manager.route_request("test-domain", "req1")
assert variant in ["variant_a", "variant_b"]

# Test metrics
ab_manager.record_metric(test["test_id"], variant, "accuracy", 0.9)
results = ab_manager.get_ab_test_results(test["test_id"])
assert "winner" in results
```

### Test Rollback

```python
from services.training import RollbackManager

rollback_manager = RollbackManager()

# Test rollback check
result = rollback_manager.check_and_rollback(
    domain_id="test-domain",
    current_metrics={"accuracy": 0.75}  # Below threshold
)

assert "rollback_triggered" in result
```

### Test Routing Optimization

```python
from services.training import RoutingOptimizer

optimizer = RoutingOptimizer()

# Record decisions
optimizer.record_routing_decision(
    domain_id="test-domain",
    query="test query",
    decision_confidence=0.8,
    actual_metrics={"accuracy": 0.9, "latency_ms": 100}
)

# Optimize weights
weights = optimizer.optimize_routing_weights(["test-domain"])
assert "test-domain" in weights
```

## Future Enhancements

1. **Canary Deployment**: Gradual rollout to percentage of requests
2. **Multi-variant Testing**: Support for A/B/C/D testing
3. **Advanced Statistics**: Proper statistical tests (t-test, chi-square)
4. **Predictive Rollback**: ML-based prediction of performance degradation
5. **Auto-optimization**: Automatic optimization parameter tuning
6. **Cross-domain Learning**: Learn from similar domains

## References

- Phase 1 Documentation: `docs/domain-configuration-phase1-dp-integration.md`
- Phase 2 Documentation: `docs/domain-configuration-phase2-implementation.md`
- Domain Configuration Review: `docs/domain-configuration-process-review.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-11-04  
**Phase**: Phase 3 Implementation Complete

