# SGMI Data Flow Rating Framework

## Overview

This document defines the rating framework for evaluating the SGMI data flow from extraction through storage to training, including AgentFlow and Open Deep Research integration.

## Rating Scale

All criteria are rated on a scale of 1-5:
- **1 (Poor)**: Critical issues, system unusable
- **2 (Below Average)**: Major issues, significant problems
- **3 (Average)**: Works but with notable limitations
- **4 (Good)**: Works well with minor issues
- **5 (Excellent)**: Works perfectly, exceeds expectations

## Rating Criteria

### 1. Data Completeness

**Definition**: All extracted nodes and edges are preserved across all storage systems (Postgres, Redis, Neo4j).

**Evaluation Points**:
- Node count matches across all systems
- Edge count matches across all systems
- Node properties preserved correctly
- Edge properties preserved correctly
- Information theory metrics propagated

**Scoring Rubric**:
- **5**: 100% data preservation, all properties intact
- **4**: 95-99% data preservation, minor property loss
- **3**: 90-94% data preservation, some properties missing
- **2**: 80-89% data preservation, significant data loss
- **1**: <80% data preservation, critical data loss

**Measurement Method**:
```sql
-- Postgres
SELECT COUNT(*) FROM glean_nodes WHERE properties_json->>'project_id' = 'sgmi';
SELECT COUNT(*) FROM glean_edges;

-- Neo4j
MATCH (n:Node) WHERE n.properties_json CONTAINS 'sgmi' RETURN COUNT(n);
MATCH ()-[r:RELATIONSHIP]->() RETURN COUNT(r);

-- Redis
SCAN 0 MATCH schema:node:* COUNT 1000
SCAN 0 MATCH schema:edge:* COUNT 1000
```

### 2. Performance

**Definition**: System performance measured by latency and throughput at each phase.

**Evaluation Points**:
- Extraction time (target: <5s for SGMI dataset)
- Postgres persistence time (target: <2s)
- Redis persistence time (target: <2s)
- Neo4j persistence time (target: <2s)
- Training data retrieval time (target: <1s)
- AgentFlow execution time (target: <10s)
- Deep research query time (target: <5s)

**Scoring Rubric**:
- **5**: All targets met, excellent performance
- **4**: Most targets met, minor performance issues
- **3**: Some targets met, noticeable performance degradation
- **2**: Few targets met, significant performance issues
- **1**: No targets met, system unusably slow

**Measurement Method**:
- Extract timing: `run_sgmi_full_graph.sh` with timestamps
- Storage timing: Log timestamps in persistence functions
- Training timing: Log timestamps in graph_client methods
- AgentFlow timing: Log timestamps in processor
- Deep research timing: HTTP response times

### 3. Error Handling

**Definition**: System resilience, error recovery, and graceful degradation.

**Evaluation Points**:
- Graceful degradation when services unavailable
- Retry logic for transient failures
- Error logging with context
- Transaction rollback on failures
- User-friendly error messages
- Error recovery mechanisms

**Scoring Rubric**:
- **5**: Excellent error handling, automatic recovery
- **4**: Good error handling, most errors handled gracefully
- **3**: Basic error handling, some errors not handled
- **2**: Poor error handling, frequent failures
- **1**: No error handling, system crashes on errors

**Evaluation Checklist**:
- [ ] Extract service handles missing files gracefully
- [ ] Postgres failures don't stop Redis/Neo4j updates
- [ ] Neo4j transaction rollback on errors
- [ ] Training service falls back to extract service
- [ ] AgentFlow retries on transient failures
- [ ] Deep research handles timeouts gracefully
- [ ] All errors logged with context

### 4. Integration Quality

**Definition**: Quality of service integration, API design, and service boundaries.

**Evaluation Points**:
- Clean API design
- Proper service boundaries
- Well-documented interfaces
- Consistent data formats
- Minimal service coupling
- Clear integration patterns

**Scoring Rubric**:
- **5**: Excellent integration, clean APIs, well-documented
- **4**: Good integration, minor coupling issues
- **3**: Acceptable integration, some coupling
- **2**: Poor integration, tight coupling
- **1**: No integration, services tightly coupled

**Evaluation Checklist**:
- [ ] Extract service has clean REST API
- [ ] Graph service client has clear interface
- [ ] Training service uses proper abstractions
- [ ] AgentFlow integration via well-defined state
- [ ] Deep research has clear API contract
- [ ] Data formats consistent across services
- [ ] Integration patterns documented

### 5. Data Consistency

**Definition**: Data consistency across Postgres, Redis, and Neo4j storage systems.

**Evaluation Points**:
- Same node counts across systems
- Same edge counts across systems
- Synchronized updates
- Consistent property values
- No data drift over time
- Validation mechanisms

**Scoring Rubric**:
- **5**: Perfect consistency, validation in place
- **4**: Good consistency, minor discrepancies
- **3**: Acceptable consistency, some drift
- **2**: Poor consistency, noticeable discrepancies
- **1**: No consistency, significant data drift

**Measurement Method**:
```python
# Validation script checks
postgres_nodes = query_postgres("SELECT COUNT(*) FROM glean_nodes")
redis_nodes = count_redis_keys("schema:node:*")
neo4j_nodes = query_neo4j("MATCH (n:Node) RETURN COUNT(n)")

consistency_score = 1 - abs(postgres_nodes - neo4j_nodes) / max(postgres_nodes, neo4j_nodes)
```

### 6. Training Readiness

**Definition**: Data quality and structure suitable for GNN training.

**Evaluation Points**:
- Valid graph structure (no orphaned nodes)
- Sufficient data volume for training
- Quality metrics available
- Proper node/edge types
- Rich property data
- Temporal data available

**Scoring Rubric**:
- **5**: Excellent data quality, ready for training
- **4**: Good data quality, minor issues
- **3**: Acceptable data quality, some limitations
- **2**: Poor data quality, significant issues
- **1**: Data not suitable for training

**Evaluation Checklist**:
- [ ] Graph structure valid (connected components)
- [ ] Minimum nodes/edges for training (e.g., >1000 nodes)
- [ ] Information theory metrics present
- [ ] Node types properly classified
- [ ] Edge relationships meaningful
- [ ] Properties rich enough for embeddings
- [ ] Temporal data available for time-series analysis

## Overall Rating Calculation

### Weighted Average

Each criterion is weighted based on importance:

1. **Data Completeness**: 25% (Critical - data loss is unacceptable)
2. **Performance**: 20% (Important - affects user experience)
3. **Error Handling**: 20% (Important - affects reliability)
4. **Integration Quality**: 15% (Moderate - affects maintainability)
5. **Data Consistency**: 10% (Moderate - affects correctness)
6. **Training Readiness**: 10% (Moderate - affects downstream use)

### Formula

```
Overall Rating = (Data Completeness × 0.25) +
                 (Performance × 0.20) +
                 (Error Handling × 0.20) +
                 (Integration Quality × 0.15) +
                 (Data Consistency × 0.10) +
                 (Training Readiness × 0.10)
```

### Rating Interpretation

- **4.5 - 5.0**: Excellent - Production ready
- **3.5 - 4.4**: Good - Minor improvements needed
- **2.5 - 3.4**: Average - Significant improvements needed
- **1.5 - 2.4**: Below Average - Major improvements required
- **1.0 - 1.4**: Poor - System not ready for production

## Rating Process

### Step 1: Data Collection
1. Run SGMI extraction end-to-end
2. Collect metrics from all systems
3. Measure performance at each phase
4. Test error scenarios
5. Validate data consistency

### Step 2: Evaluation
1. Rate each criterion independently
2. Document evidence for each rating
3. Identify specific issues
4. Note positive aspects

### Step 3: Calculation
1. Calculate weighted average
2. Determine overall rating
3. Identify priority improvements
4. Create improvement roadmap

### Step 4: Documentation
1. Document ratings with evidence
2. Create improvement recommendations
3. Prioritize fixes
4. Track progress over time

## Example Rating

### Sample Evaluation

**Data Completeness**: 4/5
- Evidence: 98% node preservation, 97% edge preservation
- Issue: Some properties lost in Redis serialization

**Performance**: 3/5
- Evidence: Extraction 6s (target 5s), Postgres 2.5s (target 2s)
- Issue: Postgres batch size needs optimization

**Error Handling**: 4/5
- Evidence: Graceful degradation works, retry logic present
- Issue: Some error messages lack context

**Integration Quality**: 4/5
- Evidence: Clean APIs, good documentation
- Issue: Some service coupling in training service

**Data Consistency**: 3/5
- Evidence: 95% consistency, minor drift
- Issue: No automatic validation mechanism

**Training Readiness**: 4/5
- Evidence: Valid structure, sufficient volume, metrics present
- Issue: Some node types need better classification

**Overall Rating**: 3.7/5 (Good - Minor improvements needed)

## Continuous Improvement

### Regular Reviews
- Monthly rating reviews
- Track rating trends over time
- Identify regressions early
- Celebrate improvements

### Rating Targets
- **Q1**: Achieve 3.5+ overall rating
- **Q2**: Achieve 4.0+ overall rating
- **Q3**: Achieve 4.5+ overall rating
- **Q4**: Maintain 4.5+ overall rating

### Improvement Tracking
- Track ratings in metrics dashboard
- Link improvements to specific changes
- Measure impact of optimizations
- Share progress with team

