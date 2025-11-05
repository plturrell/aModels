# aModels System Review & Testing Roadmap

## Executive Summary

**Status**: Phases 1-3 & 7-9 Complete ✅  
**Next Priority**: Comprehensive Integration Testing  
**Goal**: Verify end-to-end functionality of domain-aware system

---

## Current System Architecture

### Core Services

1. **LocalAI Service** (`services/localai/`)
   - Domain management (Phases 1-3)
   - Model serving (GGUF, Transformers, VaultGemma)
   - Domain lifecycle API (`/v1/domains/create`, `/v1/domains/list`)
   - Redis/PostgreSQL configuration storage

2. **Extract Service** (`services/extract/`)
   - Domain detection during extraction (Phase 1)
   - Knowledge graph generation with domain metadata
   - SAP RPT OSS integration
   - Semantic schema analysis (Phase 8.1)
   - Cross-system extraction (Phase 8.3)
   - Model fusion (Phase 8.2)
   - Self-healing (Phase 9.2)

3. **Training Service** (`services/training/`)
   - Domain filtering with differential privacy (Phase 1)
   - Domain-specific model training (Phase 2)
   - Metrics collection (Phase 2)
   - A/B testing (Phase 3)
   - Rollback management (Phase 3)
   - Routing optimization (Phase 3)
   - Pattern learning (Phase 7)
   - Auto-tuning (Phase 9.1)
   - Auto-pipeline orchestration (Phase 9.3)

4. **Search Service** (`services/search/`)
   - Semantic search with LocalAI embeddings
   - Hybrid search capabilities

5. **DeepAgents Service** (`services/deepagents/`)
   - Agent orchestration with LocalAI
   - Planning and task decomposition

6. **Orchestration Service** (`services/orchestration/`)
   - Auto-pipeline with domain awareness (Phase 9.3)
   - Model registry
   - A/B test management

7. **Analytics Service** (`services/analytics/`)
   - Predictive analytics (Phase 9.4)
   - Domain performance predictions

### Storage Systems

1. **PostgreSQL**
   - Domain configurations (`domain_configs`)
   - Training runs
   - Metrics history
   - A/B test results
   - Rollback events

2. **Redis**
   - Domain config cache
   - Traffic splitting for A/B tests
   - Optimization caching

3. **Neo4j**
   - Knowledge graphs
   - Domain associations (nodes/edges)
   - Data lineage

4. **Elasticsearch**
   - Document indexing
   - Search capabilities

---

## Completed Phases Summary

### ✅ Phase 1: Domain Association & Differential Privacy
- Domain detection during extraction
- Domain-specific training filtering
- Automated domain config updates
- Differential privacy integration

### ✅ Phase 2: Domain Training & Deployment
- Domain-specific model training
- Performance metrics collection
- Automatic deployment triggers
- Performance dashboard

### ✅ Phase 3: Advanced Management
- A/B testing
- Automatic rollback
- Learning-based routing optimization
- Domain lifecycle management
- Domain-specific optimizations

### ✅ Phase 7: Pattern Learning (Domain-Aware)
- GNN-based pattern learning with domain models
- Meta-pattern learning (layer/team-specific)
- Sequence pattern learning with domain conditioning
- Active pattern learning with domain filtering

### ✅ Phase 8: Extraction & Intelligence (Domain-Aware)
- Semantic schema analysis with domain manager
- Model fusion with domain-optimized weights
- Cross-system extraction with domain normalization
- Pattern transfer with domain similarity

### ✅ Phase 9: Automation (Domain-Aware)
- Auto-tuning with domain-specific studies
- Self-healing with domain health monitoring
- Auto-pipeline with domain orchestration
- Predictive analytics with domain predictions

---

## Critical Integration Points to Test

### 1. Domain Detection & Association Flow
```
Extract Request → Domain Detector → LocalAI /v1/domains → Domain Config → Neo4j Storage
```

**Test Cases**:
- [ ] Extract service detects domain from text content
- [ ] Domain detector loads configs from LocalAI
- [ ] Extracted nodes/edges tagged with domain_id and agent_id
- [ ] Domain metadata stored in Neo4j
- [ ] Domain detection works with Redis config
- [ ] Domain detection falls back to file-based config

### 2. Domain-Filtered Training Pipeline
```
Training Request → Domain Filter → Domain-Specific Data → Training → Metrics → Auto-Deploy
```

**Test Cases**:
- [ ] Domain filter filters training data by keywords
- [ ] Differential privacy applied to filtered data
- [ ] Domain-specific model training succeeds
- [ ] Metrics collected per domain
- [ ] Auto-deployment triggers when thresholds met
- [ ] Domain config updated in PostgreSQL
- [ ] Domain config synced to Redis

### 3. Domain-Aware Routing & Selection
```
Query → Domain Router → Domain Config → Model Selection → Response
```

**Test Cases**:
- [ ] Router selects domain based on query keywords
- [ ] Domain health checked before routing
- [ ] Domain-optimized weights used in model fusion
- [ ] Fallback to default domain when selected domain unhealthy
- [ ] Routing optimizer learns from performance feedback

### 4. A/B Testing & Rollback
```
New Model → A/B Test → Traffic Split → Metrics → Winner Selection → Rollback Check
```

**Test Cases**:
- [ ] A/B test created for new domain model
- [ ] Traffic split correctly (consistent hashing)
- [ ] Metrics tracked per variant
- [ ] Winner selected with statistical significance
- [ ] Automatic rollback triggers on degradation
- [ ] Previous version restored correctly

### 5. Pattern Learning Integration
```
Knowledge Graph → Domain Filter → Pattern Learning → Domain-Specific Patterns → Storage
```

**Test Cases**:
- [ ] GNN learns domain-specific patterns
- [ ] Meta-patterns grouped by domain layer/team
- [ ] Sequence patterns conditioned on domain
- [ ] Active learning filters by domain
- [ ] Patterns stored with domain metadata

### 6. Cross-Domain Intelligence
```
Source Domain → Domain Similarity → Pattern Transfer → Target Domain
```

**Test Cases**:
- [ ] Domain similarity calculated correctly
- [ ] Pattern transfer adapts based on similarity
- [ ] Cross-domain learning improves target domain
- [ ] Domain configs used for similarity calculation

### 7. Auto-Pipeline Orchestration
```
New Data → Domain Detection → Domain Training → Domain Deployment → A/B Test
```

**Test Cases**:
- [ ] Pipeline detects new data
- [ ] Domain auto-detected from data
- [ ] Domain-specific training triggered
- [ ] Model deployed with domain context
- [ ] A/B test started for domain
- [ ] Model registry tracks domain versions

### 8. Predictive Analytics
```
Historical Metrics → Domain Config → Prediction → Recommendations
```

**Test Cases**:
- [ ] Domain performance predicted correctly
- [ ] Domain data quality predicted
- [ ] Domain training needs forecasted
- [ ] Recommendations based on domain layer/keywords

---

## Testing Plan

### Phase 1: Unit Tests (Week 1)

#### 1.1 Domain Detection Tests
**File**: `testing/test_domain_detection.py`
- Test domain keyword matching
- Test domain config loading from LocalAI
- Test domain config fallback to file
- Test domain association with nodes/edges

#### 1.2 Domain Filter Tests
**File**: `testing/test_domain_filter.py`
- Test keyword-based filtering
- Test differential privacy application
- Test privacy budget tracking
- Test domain-specific feature extraction

#### 1.3 Domain Trainer Tests
**File**: `testing/test_domain_trainer.py`
- Test domain-specific training
- Test training run ID generation
- Test model version tracking
- Test deployment threshold checks

#### 1.4 Domain Metrics Tests
**File**: `testing/test_domain_metrics.py`
- Test metrics collection from PostgreSQL
- Test metrics collection from LocalAI
- Test trend calculation
- Test cross-domain comparison

### Phase 2: Integration Tests (Week 2)

#### 2.1 End-to-End Extraction Flow
**File**: `testing/test_extraction_flow.py`
```python
def test_extraction_with_domain_detection():
    # 1. Send extraction request
    # 2. Verify domain detected
    # 3. Verify nodes/edges tagged with domain
    # 4. Verify Neo4j storage has domain metadata
```

#### 2.2 End-to-End Training Flow
**File**: `testing/test_training_flow.py`
```python
def test_domain_training_pipeline():
    # 1. Extract knowledge graph
    # 2. Apply domain filtering
    # 3. Train domain-specific model
    # 4. Collect metrics
    # 5. Check auto-deployment
    # 6. Verify config updates
```

#### 2.3 A/B Testing Flow
**File**: `testing/test_ab_testing_flow.py`
```python
def test_ab_testing_complete():
    # 1. Create A/B test
    # 2. Send queries (traffic split)
    # 3. Collect metrics
    # 4. Select winner
    # 5. Deploy winner
```

#### 2.4 Rollback Flow
**File**: `testing/test_rollback_flow.py`
```python
def test_automatic_rollback():
    # 1. Deploy new model
    # 2. Simulate performance degradation
    # 3. Verify rollback triggered
    # 4. Verify previous version restored
```

### Phase 3: Pattern Learning Tests (Week 3)

#### 3.1 GNN Pattern Learning
**File**: `testing/test_gnn_pattern_learning.py`
- Test domain-specific GNN model creation
- Test domain feature extraction
- Test pattern learning with domain context
- Test pattern storage per domain

#### 3.2 Meta-Pattern Learning
**File**: `testing/test_meta_pattern_learning.py`
- Test layer-specific meta-patterns
- Test team-specific meta-patterns
- Test cross-domain pattern transfer
- Test domain similarity calculation

#### 3.3 Sequence Pattern Learning
**File**: `testing/test_sequence_pattern_learning.py`
- Test domain-conditioned sequences
- Test SAP RPT domain embeddings
- Test temporal pattern extraction
- Test domain-specific sequence models

#### 3.4 Active Pattern Learning
**File**: `testing/test_active_pattern_learning.py`
- Test domain-filtered active learning
- Test domain keyword validation
- Test domain-specific taxonomy
- Test rare pattern discovery per domain

### Phase 4: Extraction & Intelligence Tests (Week 4)

#### 4.1 Semantic Schema Analysis
**File**: `testing/test_semantic_schema_analysis.py`
- Test domain-aware column analysis
- Test domain keyword matching
- Test domain tag similarity
- Test domain config integration

#### 4.2 Model Fusion
**File**: `testing/test_model_fusion.py`
- Test domain-optimized weights
- Test domain-specific weight caching
- Test weight optimization per domain layer
- Test ensemble with domain context

#### 4.3 Cross-System Extraction
**File**: `testing/test_cross_system_extraction.py`
- Test domain-normalized patterns
- Test domain keyword confidence boosting
- Test domain-specific normalization rules
- Test cross-system pattern extraction

#### 4.4 Pattern Transfer
**File**: `testing/test_pattern_transfer.py`
- Test domain similarity calculation
- Test pattern adaptation
- Test cross-domain transfer confidence
- Test routing optimizer integration

### Phase 5: Automation Tests (Week 5)

#### 5.1 Auto-Tuning
**File**: `testing/test_auto_tuning.py`
- Test domain-specific Optuna studies
- Test domain-aware hyperparameter constraints
- Test domain-specific architecture selection
- Test domain config loading

#### 5.2 Self-Healing
**File**: `testing/test_self_healing.py`
- Test domain health monitoring
- Test domain health score calculation
- Test domain-aware circuit breakers
- Test domain-specific fallbacks

#### 5.3 Auto-Pipeline
**File**: `testing/test_auto_pipeline.py`
- Test domain-aware training orchestration
- Test domain-specific deployment
- Test domain-aware A/B testing
- Test model registry with domain context

#### 5.4 Predictive Analytics
**File**: `testing/test_predictive_analytics.py`
- Test domain performance prediction
- Test domain quality prediction
- Test domain training needs forecast
- Test domain-aware recommendations

### Phase 6: End-to-End System Tests (Week 6)

#### 6.1 Complete Domain Lifecycle
**File**: `testing/test_complete_domain_lifecycle.py`
```python
def test_complete_domain_lifecycle():
    # 1. Create domain via API
    # 2. Extract data → domain detected
    # 3. Train domain model
    # 4. Deploy with A/B test
    # 5. Monitor performance
    # 6. Optimize routing
    # 7. Predict future needs
    # 8. Archive domain
```

#### 6.2 Multi-Domain Workflow
**File**: `testing/test_multi_domain_workflow.py`
- Test multiple domains simultaneously
- Test cross-domain learning
- Test domain priority handling
- Test domain resource allocation

#### 6.3 Failure Scenarios
**File**: `testing/test_failure_scenarios.py`
- Test LocalAI unavailable
- Test PostgreSQL unavailable
- Test Redis unavailable
- Test Neo4j unavailable
- Test domain config corruption
- Test rollback scenarios

#### 6.4 Performance & Load Tests
**File**: `testing/test_performance.py`
- Test concurrent domain requests
- Test large knowledge graphs
- Test high-volume training
- Test A/B test traffic splitting
- Test routing optimization latency

---

## Test Infrastructure Setup

### Prerequisites

1. **Docker Compose Environment**
   ```bash
   cd infrastructure/docker/brev
   docker compose up -d
   ```

2. **Test Data**
   - Sample knowledge graphs
   - Sample domain configurations
   - Sample training data
   - Sample queries

3. **Test Database State**
   - Clean PostgreSQL (domain_configs table)
   - Clean Redis
   - Clean Neo4j
   - Test domain configs loaded

### Test Environment Variables

```bash
# Core Services
LOCALAI_URL=http://localai:8080
EXTRACT_SERVICE_URL=http://extract-service:19080
TRAINING_SERVICE_URL=http://training-service:8080
SEARCH_SERVICE_URL=http://search-service:8090

# Storage
POSTGRES_DSN=postgresql://user:pass@postgres:5432/amodels
REDIS_URL=redis://redis:6379/0
NEO4J_URI=bolt://neo4j:7687

# Test Configuration
ENABLE_DOMAIN_FILTERING=true
ENABLE_DOMAIN_TRAINING=true
ENABLE_AB_TESTING=true
PRIVACY_EPSILON=1.0
```

---

## Test Execution Strategy

### Daily Smoke Tests
Run quick tests to verify basic functionality:
```bash
./testing/run_smoke_tests.sh
```

### Weekly Integration Tests
Run full integration test suite:
```bash
./testing/run_integration_tests.sh
```

### Pre-Deployment Tests
Run before deploying to production:
```bash
./testing/run_pre_deployment_tests.sh
```

### Continuous Integration
Add to CI/CD pipeline:
```yaml
# .github/workflows/test.yml
- name: Run Integration Tests
  run: |
    docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
    sleep 60
    ./testing/run_all_tests.sh
```

---

## Success Criteria

### Minimum Viable Testing (MVP)
- ✅ All services start successfully
- ✅ Domain detection works end-to-end
- ✅ Domain training completes successfully
- ✅ A/B testing functions correctly
- ✅ Rollback triggers on degradation

### Production-Ready Testing
- ✅ All unit tests pass (>90% coverage)
- ✅ All integration tests pass
- ✅ Performance tests meet SLA
- ✅ Failure scenarios handled gracefully
- ✅ Load tests pass

---

## Next Steps

### Immediate (This Week)

1. **Update Existing Tests**
   - [ ] Update `test_localai_integration_suite.py` with domain endpoints
   - [ ] Add domain detection tests
   - [ ] Add domain filtering tests

2. **Create New Test Files**
   - [ ] `test_domain_detection.py`
   - [ ] `test_domain_filter.py`
   - [ ] `test_domain_trainer.py`
   - [ ] `test_domain_metrics.py`

3. **Set Up Test Environment**
   - [ ] Create test Docker Compose file
   - [ ] Create test data fixtures
   - [ ] Set up test databases

### Short Term (Next 2 Weeks)

1. **Integration Tests**
   - [ ] End-to-end extraction flow
   - [ ] End-to-end training flow
   - [ ] A/B testing flow
   - [ ] Rollback flow

2. **Pattern Learning Tests**
   - [ ] GNN pattern learning
   - [ ] Meta-pattern learning
   - [ ] Sequence pattern learning
   - [ ] Active pattern learning

### Medium Term (Next Month)

1. **Complete Test Coverage**
   - [ ] All Phase 7-9 components tested
   - [ ] All integration points verified
   - [ ] Performance benchmarks established

2. **CI/CD Integration**
   - [ ] Automated test runs
   - [ ] Test result reporting
   - [ ] Failure notifications

---

## Known Issues & Gaps

### Configuration Issues
- [ ] Some services may need domain config paths updated
- [ ] Redis connection may need verification
- [ ] PostgreSQL schema may need migrations

### Integration Gaps
- [ ] Domain lifecycle API endpoints need testing
- [ ] Cross-domain learning needs verification
- [ ] Predictive analytics needs validation

### Documentation Gaps
- [ ] API documentation for domain endpoints
- [ ] Deployment guide for domain system
- [ ] Troubleshooting guide

---

## Risk Assessment

### High Risk Areas
1. **Domain Detection Accuracy** - May misclassify domains
2. **Differential Privacy** - May affect model performance
3. **A/B Testing Traffic Split** - May not be truly random
4. **Rollback Logic** - May not trigger correctly
5. **Cross-Domain Learning** - May transfer incorrectly

### Mitigation Strategies
1. **Accuracy Testing** - Test with known domain data
2. **Privacy Testing** - Verify privacy budget usage
3. **Statistical Testing** - Verify traffic distribution
4. **Failure Testing** - Simulate degradation scenarios
5. **Transfer Testing** - Validate pattern adaptation

---

## Conclusion

The aModels platform has extensive domain-aware functionality implemented across Phases 1-3 and 7-9. The next critical step is comprehensive testing to ensure:

1. **Correctness**: All features work as designed
2. **Integration**: Components work together seamlessly
3. **Performance**: System meets performance requirements
4. **Reliability**: System handles failures gracefully
5. **Scalability**: System handles multiple domains

**Priority**: Start with Phase 1 tests (domain detection and filtering) as these are foundational to all other features.

---

**Document Version**: 1.0  
**Created**: 2025-01-XX  
**Status**: Testing Roadmap  
**Next Review**: After Phase 1 tests complete

