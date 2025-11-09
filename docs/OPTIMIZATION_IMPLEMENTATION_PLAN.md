# Optimization Implementation Plan

**Based on**: Service Integration Review  
**Date**: 2025-01-XX  
**Priority**: High, Medium, Low

---

## Phase 1: High Priority Optimizations (2-3 weeks)

### 1.1 LocalAI ↔ Postgres Integration

**Status**: Code exists but unused  
**Impact**: High - Enables model cache persistence, inference logging, performance tracking  
**Effort**: Medium (3-5 days)

#### Implementation Steps

1. **Integrate PostgresConfigStore into LocalAI Startup**
   - File: `services/localai/cmd/vaultgemma-server/main.go`
   - Load domain configs from Postgres if `POSTGRES_DSN` is set
   - Fallback to JSON file if Postgres unavailable
   - Add health check for Postgres connection

2. **Persist Model Cache State**
   - File: `services/localai/pkg/server/model_cache.go`
   - Save cache state to Postgres on model load/unload
   - Restore cache state on startup
   - Track model memory usage in Postgres

3. **Log Inference Requests**
   - File: `services/localai/pkg/server/vaultgemma_server.go`
   - Create `inference_logs` table in Postgres
   - Log: domain, model, prompt_length, response_length, latency, tokens, timestamp
   - Batch writes for performance

4. **Store Performance Metrics**
   - File: `services/localai/pkg/server/model_cache.go`
   - Track: loading_time, memory_usage, access_count, last_access
   - Store in `model_performance_metrics` table
   - Enable performance analysis queries

#### Database Schema

```sql
-- Inference logs table
CREATE TABLE IF NOT EXISTS inference_logs (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) NOT NULL,
    model_name VARCHAR(255),
    prompt_length INTEGER,
    response_length INTEGER,
    latency_ms INTEGER,
    tokens_generated INTEGER,
    tokens_prompt INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    workflow_id VARCHAR(255),
    user_id VARCHAR(255)
);

CREATE INDEX idx_inference_logs_domain ON inference_logs(domain);
CREATE INDEX idx_inference_logs_created_at ON inference_logs(created_at);
CREATE INDEX idx_inference_logs_workflow_id ON inference_logs(workflow_id);

-- Model performance metrics table
CREATE TABLE IF NOT EXISTS model_performance_metrics (
    id SERIAL PRIMARY KEY,
    domain VARCHAR(255) NOT NULL,
    model_name VARCHAR(255),
    loading_time_ms INTEGER,
    memory_usage_mb INTEGER,
    access_count BIGINT DEFAULT 0,
    last_access TIMESTAMP,
    avg_latency_ms INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_model_performance_domain ON model_performance_metrics(domain);
CREATE INDEX idx_model_performance_last_access ON model_performance_metrics(last_access);

-- Model cache state table
CREATE TABLE IF NOT EXISTS model_cache_state (
    domain VARCHAR(255) PRIMARY KEY,
    model_type VARCHAR(50), -- safetensors, gguf, transformers
    model_path TEXT,
    loaded_at TIMESTAMP,
    memory_mb INTEGER,
    access_count BIGINT DEFAULT 0,
    last_access TIMESTAMP,
    cache_data JSONB, -- Additional cache metadata
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### Success Criteria
- ✅ Domain configs loadable from Postgres
- ✅ Model cache state persisted and restored
- ✅ Inference requests logged with <5ms overhead
- ✅ Performance metrics queryable

---

### 1.2 Connection Pooling

**Status**: No pooling implemented  
**Impact**: High - 20-40% latency reduction, 30-50% resource reduction  
**Effort**: Medium (4-6 days)

#### Implementation Steps

1. **Create Shared Connection Pool Manager**
   - New file: `services/shared/pkg/connectionpool/pool.go`
   - HTTP connection pool with configurable size
   - gRPC connection pool with keepalive
   - Connection reuse across services

2. **Update LocalAI GPU Router**
   - File: `services/localai/pkg/gpu/gpu_router.go`
   - Use shared HTTP client from pool
   - Configure connection pool size (default: 10)
   - Add connection pool metrics

3. **Update Graph Service**
   - File: `services/graph/pkg/workflows/gpu_processor.go`
   - Use shared HTTP client for GPU orchestrator
   - File: `services/graph/pkg/workflows/orchestration_processor.go`
   - Use shared HTTP client for LocalAI
   - Pool gRPC connections to Postgres

4. **Update GPU Orchestrator**
   - File: `services/gpu-orchestrator/gpu_orchestrator/gpu_orchestrator.go`
   - Use shared HTTP client for DeepAgents
   - Use shared HTTP client for Graph service

#### Connection Pool Configuration

```go
// services/shared/pkg/connectionpool/pool.go
type HTTPPoolConfig struct {
    MaxIdleConns        int           // Default: 100
    MaxIdleConnsPerHost int           // Default: 10
    IdleConnTimeout     time.Duration // Default: 90s
    MaxConnsPerHost     int           // Default: 50
}

type GRPCPoolConfig struct {
    MaxConns        int           // Default: 10
    KeepAliveTime   time.Duration // Default: 30s
    KeepAliveTimeout time.Duration // Default: 5s
}
```

#### Success Criteria
- ✅ Connection reuse rate >80%
- ✅ Latency reduction 20-40% for cross-service calls
- ✅ Memory usage reduction 30-50%
- ✅ No connection leaks

---

### 1.3 Workflow-Aware GPU Allocation

**Status**: Partial (workflow context not fully utilized)  
**Impact**: High - 20-30% resource utilization improvement  
**Effort**: Medium (3-4 days)

#### Implementation Steps

1. **Enhance GPU Orchestrator with Workflow Context**
   - File: `services/gpu-orchestrator/gpu_orchestrator/gpu_orchestrator.go`
   - Accept workflow_id, workflow_priority, workflow_dependencies
   - Consider workflow priority in allocation decisions
   - Track workflow-based allocations

2. **Pass Workflow Metadata from Graph Service**
   - File: `services/graph/pkg/workflows/gpu_processor.go`
   - Extract workflow_id, priority from state
   - Include in GPU allocation request
   - Pass workflow dependencies if available

3. **LocalAI Workflow Context Integration**
   - File: `services/localai/pkg/server/vaultgemma_server.go`
   - Accept workflow_id, workflow_priority in requests
   - Use workflow context for domain routing
   - Pass workflow_id to GPU orchestrator

4. **Priority-Based Scheduling**
   - File: `services/gpu-orchestrator/scheduler/scheduler.go`
   - Implement priority queue for allocations
   - High-priority workflows get dedicated resources
   - Low-priority workflows can share resources

#### Workflow Context Data Structure

```go
type WorkflowContext struct {
    WorkflowID      string
    Priority        int  // 1-10, higher = more important
    Dependencies    []string
    EstimatedDuration time.Duration
    RequiredGPUs    int
    MinMemoryMB     int64
}
```

#### Success Criteria
- ✅ Workflow priority considered in allocation
- ✅ High-priority workflows get resources faster
- ✅ Resource utilization improved 20-30%
- ✅ Workflow dependencies tracked

---

## Phase 2: Medium Priority Optimizations (3-4 weeks)

### 2.1 Cross-Service Caching

**Status**: No shared cache  
**Impact**: Medium-High - 50-80% latency reduction for cached requests  
**Effort**: High (5-7 days)

#### Implementation Steps

1. **Redis Integration for Shared Metadata**
   - Create: `services/shared/pkg/cache/redis.go`
   - Cache: Model registry, domain configs, workflow state
   - TTL-based expiration
   - Cache invalidation on updates

2. **Postgres-Backed Model Cache**
   - File: `services/localai/pkg/server/model_cache.go`
   - Load cache state from Postgres on startup
   - Persist cache state periodically
   - Use Postgres as cache backend

3. **Workflow Result Caching**
   - File: `services/graph/pkg/workflows/unified_processor.go`
   - Cache workflow results by input hash
   - TTL: 1 hour default
   - Cache key: hash(workflow_type + input)

#### Cache Strategy

- **Model Registry**: Redis, TTL: 24h, invalidate on model update
- **Domain Configs**: Redis + Postgres, TTL: 1h, invalidate on config change
- **Workflow Results**: Redis, TTL: 1h, cache by input hash
- **GPU Allocation Cache**: Redis, TTL: 5m, cache allocation decisions

#### Success Criteria
- ✅ Cache hit rate >60% for model metadata
- ✅ Latency reduction 50-80% for cached requests
- ✅ Load reduction 60-70% on downstream services

---

### 2.2 Request Batching

**Status**: Individual requests  
**Impact**: Medium - 3-5x throughput improvement  
**Effort**: Medium (4-5 days)

#### Implementation Steps

1. **GPU Allocation Batching**
   - File: `services/localai/pkg/gpu/gpu_router.go`
   - Batch multiple model GPU requests
   - Single API call for multiple allocations
   - Batch size: 5-10 requests

2. **Graph → LocalAI Aggregation**
   - File: `services/graph/pkg/workflows/orchestration_processor.go`
   - Batch multiple inference requests
   - Aggregate chat completions in workflow
   - Batch size: 3-5 requests

3. **Postgres Batch Writes**
   - File: `services/extract/schema_replication.go`
   - Batch schema replication writes
   - File: `services/localai/pkg/domain/postgres_config.go`
   - Batch inference log writes
   - Batch size: 50-100 records

#### Batching Configuration

```go
type BatchConfig struct {
    MaxBatchSize    int           // Default: 10
    MaxWaitTime     time.Duration // Default: 100ms
    FlushInterval   time.Duration // Default: 1s
}
```

#### Success Criteria
- ✅ Throughput improvement 3-5x for batched operations
- ✅ Latency reduction 30-50% for batched requests
- ✅ Network overhead reduction 40-60%

---

### 2.3 Performance Telemetry

**Status**: Limited observability  
**Impact**: Medium - 50-70% debugging time reduction  
**Effort**: Medium (4-5 days)

#### Implementation Steps

1. **Unified Prometheus Metrics**
   - Add metrics to all services
   - Cross-service correlation IDs
   - Unified metric namespaces

2. **Postgres Telemetry Storage**
   - Create: `services/postgres/migrations/telemetry_schema.sql`
   - Store: Performance metrics, allocation history, workflow execution
   - Enable historical analysis

3. **Graph Service Aggregation**
   - File: `services/graph/cmd/graph-server/main.go`
   - Aggregate telemetry from all services
   - Provide unified telemetry API
   - Enable cross-service analytics

#### Telemetry Schema

```sql
CREATE TABLE IF NOT EXISTS service_telemetry (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(255) NOT NULL,
    operation VARCHAR(255),
    latency_ms INTEGER,
    success BOOLEAN,
    error_message TEXT,
    correlation_id VARCHAR(255),
    workflow_id VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_telemetry_service ON service_telemetry(service_name);
CREATE INDEX idx_telemetry_correlation ON service_telemetry(correlation_id);
CREATE INDEX idx_telemetry_workflow ON service_telemetry(workflow_id);
CREATE INDEX idx_telemetry_created_at ON service_telemetry(created_at);
```

#### Success Criteria
- ✅ All services expose Prometheus metrics
- ✅ Cross-service correlation working
- ✅ Telemetry queryable from Postgres
- ✅ Debugging time reduced 50-70%

---

## Phase 3: Low Priority Enhancements (4-6 weeks)

### 3.1 Bidirectional Graph ↔ LocalAI

**Status**: One-way only  
**Impact**: Low-Medium - Context-aware inference  
**Effort**: High (5-7 days)

#### Implementation Steps

1. **LocalAI → Graph Integration**
   - File: `services/localai/pkg/server/vaultgemma_server.go`
   - Update knowledge graphs with inference results
   - Query knowledge graphs for context
   - Add Graph service client

2. **Context-Aware Routing**
   - File: `services/localai/pkg/domain/domain_manager.go`
   - Use knowledge graph context for domain routing
   - Enrich prompts with graph context
   - Improve domain detection accuracy

#### Success Criteria
- ✅ LocalAI can update knowledge graphs
- ✅ Context-aware domain routing working
- ✅ Improved inference quality

---

### 3.2 Model Registry Sharing

**Status**: Registry exists but not shared  
**Impact**: Low-Medium - Better GPU allocation  
**Effort**: Medium (3-4 days)

#### Implementation Steps

1. **Model Metadata API**
   - File: `services/localai/pkg/server/vaultgemma_server.go`
   - Expose `/v1/models/metadata` endpoint
   - Return model registry information
   - Enable querying by model characteristics

2. **GPU Orchestrator Integration**
   - File: `services/gpu-orchestrator/workload_analyzer/workload_analyzer.go`
   - Query LocalAI for model metadata
   - Use model characteristics for allocation
   - Cache model metadata

#### Success Criteria
- ✅ Model metadata API functional
- ✅ GPU orchestrator uses model metadata
- ✅ Better allocation decisions

---

### 3.3 Model Lifecycle Management

**Status**: Independent model loading  
**Impact**: Low-Medium - Predictive loading  
**Effort**: High (6-8 days)

#### Implementation Steps

1. **GPU Orchestrator Awareness**
   - File: `services/gpu-orchestrator/gpu_orchestrator/gpu_orchestrator.go`
   - Receive model loading/unloading events
   - Coordinate GPU allocation with lifecycle
   - Predictive GPU allocation

2. **Graph Service Tracking**
   - File: `services/graph/pkg/workflows/unified_processor.go`
   - Track model usage patterns
   - Predict model loading needs
   - Coordinate with LocalAI

3. **Postgres Performance History**
   - Use existing `model_performance_metrics` table
   - Enable predictive loading
   - Optimize model selection

#### Success Criteria
- ✅ Model lifecycle events tracked
- ✅ Predictive loading working
- ✅ Startup time reduced 60-80% for frequent models

---

## Implementation Checklist

### Phase 1 (Weeks 1-3)
- [ ] LocalAI Postgres integration
- [ ] Connection pooling implementation
- [ ] Workflow-aware GPU allocation
- [ ] Testing and validation

### Phase 2 (Weeks 4-7)
- [ ] Cross-service caching
- [ ] Request batching
- [ ] Performance telemetry
- [ ] Testing and validation

### Phase 3 (Weeks 8-13)
- [ ] Bidirectional Graph ↔ LocalAI
- [ ] Model registry sharing
- [ ] Model lifecycle management
- [ ] Testing and validation

---

## Risk Mitigation

### Technical Risks
1. **Postgres Performance**: Use connection pooling, batch writes, indexes
2. **Cache Invalidation**: Implement TTL and event-based invalidation
3. **Backward Compatibility**: Maintain JSON file fallback for domain configs

### Operational Risks
1. **Deployment**: Gradual rollout with feature flags
2. **Monitoring**: Enhanced observability before optimization
3. **Rollback**: Keep old code paths for quick rollback

---

## Success Metrics Dashboard

### Latency Metrics
- P50, P95, P99 latencies for all service calls
- Target: 30-50% reduction

### Resource Metrics
- Memory usage per service
- Connection pool utilization
- Target: 40% reduction

### Throughput Metrics
- Requests per second
- Concurrent connections
- Target: 2-3x improvement

### Cache Metrics
- Cache hit rate
- Cache size
- Target: >60% hit rate

### GPU Metrics
- GPU utilization
- Allocation efficiency
- Target: 25-35% improvement

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-XX  
**Next Review**: After Phase 1 completion

