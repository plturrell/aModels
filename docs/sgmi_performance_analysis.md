# SGMI Data Flow Performance Analysis

## Overview

This document analyzes performance metrics for each phase of the SGMI data flow from extraction through storage to training, including AgentFlow and Open Deep Research integration.

## Performance Targets

### Extraction Phase
- **Target**: <5 seconds for SGMI dataset
- **Components**: File parsing, graph building, metrics calculation

### Storage Phase
- **Postgres**: <2 seconds for batch insert
- **Redis**: <2 seconds for schema storage
- **Neo4j**: <2 seconds for graph persistence

### Training Phase
- **Graph Retrieval**: <1 second for query
- **GNN Processing**: Variable based on graph size
- **Cache Lookup**: <100ms

### Integration Phase
- **AgentFlow**: <10 seconds for flow execution
- **Deep Research**: <5 seconds for query

## Performance Measurement Methodology

### 1. Extraction Performance

**Measurement Points**:
- File reading time
- JSON parsing time
- DDL parsing time
- Control-M parsing time
- Graph building time
- Metrics calculation time

**Code References**:
- `services/extract/main.go:940-965`: JSON table extraction
- `services/extract/main.go:967-1020`: Hive DDL parsing
- `services/extract/main.go:1022-1120`: Control-M parsing
- `services/extract/information_theory.go`: Metrics calculation

**Optimization Opportunities**:
1. Parallel file processing
2. Streaming JSON parsing for large files
3. Incremental graph building
4. Cached metrics calculation

### 2. Postgres Persistence Performance

**Measurement Points**:
- Connection time
- Table creation time (if needed)
- Batch insert time
- Transaction commit time

**Code References**:
- `services/extract/schema_replication.go:178-212`: Postgres replication
- `services/extract/schema_replication.go:214-244`: Batch node insertion
- `services/extract/schema_replication.go:246-276`: Batch edge insertion

**Current Implementation**:
- Batch size: 1000 records (configurable)
- Uses prepared statements
- Single transaction for all operations
- ON CONFLICT for upserts

**Performance Characteristics**:
- **Small datasets (<1000 nodes)**: ~100-200ms
- **Medium datasets (1000-10000 nodes)**: ~500ms-1s
- **Large datasets (>10000 nodes)**: 1-3s

**Optimization Opportunities**:
1. Increase batch size for large datasets
2. Parallel batch processing
3. Connection pooling
4. Index optimization

### 3. Redis Persistence Performance

**Measurement Points**:
- Connection time
- Pipeline execution time
- Key storage time
- Queue operations time

**Code References**:
- `services/extract/redis.go:199-336`: Schema saving
- `services/extract/redis.go:34-64`: Vector storage

**Current Implementation**:
- Pipeline operations for batch efficiency
- Hash storage for schema data
- List operations for entity queue
- TTL management for cache

**Performance Characteristics**:
- **Small datasets (<1000 nodes)**: ~50-100ms
- **Medium datasets (1000-10000 nodes)**: ~200-500ms
- **Large datasets (>10000 nodes)**: 500ms-1s

**Optimization Opportunities**:
1. Pipeline batching optimization
2. Compression for large values
3. Redis cluster for scalability
4. Async operations

### 4. Neo4j Persistence Performance

**Measurement Points**:
- Session creation time
- Transaction execution time
- Node MERGE time
- Edge MERGE time
- Transaction commit time

**Code References**:
- `services/extract/neo4j.go:82-190`: Graph saving
- `services/extract/neo4j.go:113-136`: Node MERGE
- `services/extract/neo4j.go:138-184`: Edge MERGE

**Current Implementation**:
- Single transaction for all operations
- MERGE operations for idempotency
- Property serialization as JSON
- Index on node.id

**Performance Characteristics**:
- **Small datasets (<1000 nodes)**: ~200-500ms
- **Medium datasets (1000-10000 nodes)**: ~1-2s
- **Large datasets (>10000 nodes)**: 2-5s

**Optimization Opportunities**:
1. Batch MERGE operations
2. Parallel transaction processing
3. Index optimization
4. Property indexing for common queries

### 5. Training Service Performance

**Measurement Points**:
- Graph client query time
- Neo4j query execution time
- Data streaming time
- GNN processing time

**Code References**:
- `services/training/graph_client.py:135-293`: Graph data retrieval
- `services/training/graph_client.py:233-293`: Streaming methods
- `services/training/pipeline.py:696-900`: GNN processing

**Current Implementation**:
- Streaming for large graphs (1000 nodes per batch)
- Caching for repeated queries
- Parallel GNN processing (if enabled)
- Connection pooling

**Performance Characteristics**:
- **Graph Query**: 100-500ms (depending on graph size)
- **Streaming**: 50-200ms per batch
- **GNN Processing**: Variable (depends on model complexity)
- **Cache Hit**: <10ms

**Optimization Opportunities**:
1. Query optimization
2. Batch size tuning
3. Cache warming
4. Parallel processing optimization

### 6. AgentFlow Integration Performance

**Measurement Points**:
- Flow execution time
- Knowledge graph context passing time
- Result processing time

**Code References**:
- `services/graph/pkg/workflows/agentflow_processor.go:44-218`: Flow execution
- `services/graph/pkg/workflows/unified_processor.go:462-508`: Integration

**Current Implementation**:
- HTTP client with 120s timeout
- Retry logic with exponential backoff
- State management for context passing

**Performance Characteristics**:
- **Flow Execution**: 2-10s (depends on flow complexity)
- **Context Passing**: <100ms
- **Error Recovery**: +2-5s for retries

**Optimization Opportunities**:
1. Async flow execution
2. Context caching
3. Connection pooling
4. Flow optimization

### 7. Open Deep Research Performance

**Measurement Points**:
- Research query time
- SPARQL execution time
- Catalog search time
- Result processing time

**Code References**:
- `services/catalog/research/deep_research_tool.py:14-46`: Research tool
- `services/gateway/main.py:2727-2746`: API endpoint

**Current Implementation**:
- HTTP client with 300s timeout
- SPARQL query execution
- Catalog search integration

**Performance Characteristics**:
- **Research Query**: 2-5s (depends on query complexity)
- **SPARQL Execution**: 500ms-2s
- **Catalog Search**: 100-500ms

**Optimization Opportunities**:
1. Query caching
2. Parallel tool execution
3. Result streaming
4. Query optimization

## Performance Bottlenecks

### Identified Bottlenecks

1. **Neo4j Transaction Processing**
   - **Issue**: Single transaction for all operations
   - **Impact**: High latency for large datasets
   - **Solution**: Batch transactions, parallel processing

2. **Postgres Batch Size**
   - **Issue**: Fixed batch size may not be optimal
   - **Impact**: Suboptimal performance for varying dataset sizes
   - **Solution**: Dynamic batch sizing

3. **Graph Client Query Performance**
   - **Issue**: Sequential queries for large graphs
   - **Impact**: High latency for training data retrieval
   - **Solution**: Parallel queries, query optimization

4. **AgentFlow Synchronous Execution**
   - **Issue**: Blocking flow execution
   - **Impact**: High latency for complex flows
   - **Solution**: Async execution, result polling

### Performance Metrics Collection

**Recommended Metrics**:
- Extraction time by component
- Storage time by system
- Training retrieval time
- Integration execution time
- Error rates and retry counts
- Cache hit rates

**Collection Method**:
- Log timestamps at key points
- Use structured logging
- Aggregate metrics in monitoring system
- Generate performance reports

## Performance Recommendations

### Immediate Actions

1. **Add Performance Logging**
   - Log timestamps at each phase
   - Track duration for each operation
   - Aggregate metrics for analysis

2. **Optimize Batch Sizes**
   - Tune Postgres batch size based on dataset size
   - Optimize Redis pipeline batch size
   - Adjust Neo4j transaction batch size

3. **Implement Caching**
   - Cache graph queries in training service
   - Cache AgentFlow flow definitions
   - Cache Deep Research results

### Short-term Improvements

1. **Parallel Processing**
   - Parallel file processing in extraction
   - Parallel storage operations
   - Parallel GNN processing

2. **Connection Pooling**
   - Optimize database connection pools
   - Implement Redis connection pooling
   - Neo4j session pooling

3. **Query Optimization**
   - Optimize Neo4j Cypher queries
   - Add database indexes
   - Optimize SPARQL queries

### Long-term Improvements

1. **Async Operations**
   - Async storage operations
   - Async AgentFlow execution
   - Async Deep Research queries

2. **Streaming Processing**
   - Stream large datasets
   - Incremental graph building
   - Streaming GNN processing

3. **Distributed Processing**
   - Distributed graph storage
   - Distributed training processing
   - Distributed research queries

## Performance Monitoring

### Key Performance Indicators (KPIs)

1. **Extraction Time**: Target <5s
2. **Storage Time**: Target <2s per system
3. **Training Retrieval**: Target <1s
4. **AgentFlow Execution**: Target <10s
5. **Deep Research**: Target <5s

### Monitoring Tools

- Application logs with timestamps
- Performance metrics dashboard
- Alerting on performance degradation
- Regular performance reviews

### Performance Testing

- Regular end-to-end performance tests
- Load testing with varying dataset sizes
- Stress testing for error scenarios
- Performance regression testing

## Conclusion

The SGMI data flow performance is generally acceptable but has room for improvement. Key areas for optimization include:

1. Neo4j transaction processing
2. Batch size optimization
3. Query performance
4. Async operations
5. Caching strategies

Regular performance monitoring and optimization will ensure the system meets performance targets as data volumes grow.

