# High-Priority Improvements Configuration Guide

This document describes the configuration options for all 6 high-priority improvements.

## 1. Data Validation Before Storage

### Configuration
No additional configuration required. Validation runs automatically before all storage operations.

### Environment Variables
- `VALIDATION_STRICT_MODE` (default: `false`): If `true`, rejects all data with validation errors instead of filtering

### Metrics
Access via `/metrics/improvements` endpoint:
- `validation.total_validated`: Total number of validation operations
- `validation.nodes_validated`: Total nodes validated
- `validation.edges_validated`: Total edges validated
- `validation.nodes_rejected`: Total nodes rejected
- `validation.edges_rejected`: Total edges rejected
- `validation.avg_validation_time`: Average validation time

## 2. Retry Logic for Storage Operations

### Configuration
Retry behavior is configurable per storage system:

#### Postgres Retry
- `POSTGRES_RETRY_MAX_ATTEMPTS` (default: `3`): Maximum retry attempts
- `POSTGRES_RETRY_INITIAL_BACKOFF` (default: `200ms`): Initial backoff duration
- `POSTGRES_RETRY_MAX_BACKOFF` (default: `1s`): Maximum backoff duration

#### Redis Retry
- `REDIS_RETRY_MAX_ATTEMPTS` (default: `3`): Maximum retry attempts
- `REDIS_RETRY_INITIAL_BACKOFF` (default: `100ms`): Initial backoff duration
- `REDIS_RETRY_MAX_BACKOFF` (default: `500ms`): Maximum backoff duration

#### Neo4j Retry
- `NEO4J_RETRY_MAX_ATTEMPTS` (default: `3`): Maximum retry attempts
- `NEO4J_RETRY_INITIAL_BACKOFF` (default: `300ms`): Initial backoff duration
- `NEO4J_RETRY_MAX_BACKOFF` (default: `2s`): Maximum backoff duration

### Metrics
- `retry.total_retries`: Total retry operations
- `retry.successful_retries`: Successful retries
- `retry.failed_retries`: Failed retries
- `retry.success_rate`: Success rate percentage
- `retry.avg_retry_time`: Average retry time

## 3. Automatic Consistency Validation

### Configuration
- `CONSISTENCY_CHECK_ENABLED` (default: `true`): Enable/disable consistency checks
- `CONSISTENCY_CHECK_VARIANCE_THRESHOLD` (default: `5%`): Maximum allowed variance percentage
- `CONSISTENCY_CHECK_INTERVAL` (default: `after_replication`): When to run checks (`after_replication`, `on_demand`, `scheduled`)

### Metrics
- `consistency.total_checks`: Total consistency checks performed
- `consistency.consistent_checks`: Number of consistent checks
- `consistency.inconsistent_checks`: Number of inconsistent checks
- `consistency.consistency_rate`: Consistency rate percentage
- `consistency.avg_node_variance`: Average node count variance
- `consistency.avg_edge_variance`: Average edge count variance
- `consistency.avg_check_time`: Average check time

## 4. Unified Data Access Layer

### Configuration
- `ENABLE_UNIFIED_DATA_ACCESS` (default: `true`): Enable unified data access layer
- `UNIFIED_DATA_ACCESS_PRIMARY_STORAGE` (default: `neo4j`): Primary storage system (`neo4j`, `postgres`, `redis`)

### Storage Configuration
- `POSTGRES_DSN`: Postgres connection string
- `REDIS_URL`: Redis connection URL
- `NEO4J_URI`: Neo4j connection URI
- `NEO4J_USERNAME`: Neo4j username
- `NEO4J_PASSWORD`: Neo4j password

### Metrics
Access via training service cache metrics:
- Cache hit/miss rates
- Query performance metrics
- Storage system availability

## 5. Neo4j Transaction Processing Optimization

### Configuration
- `NEO4J_BATCH_SIZE` (default: `1000`): Batch size for nodes/edges
- `NEO4J_BATCH_SIZE_LARGE` (default: `500`): Batch size for datasets >10K nodes
- `NEO4J_BATCH_SIZE_THRESHOLD` (default: `10000`): Threshold for switching to smaller batches

### Metrics
- `neo4j_batch.total_batches`: Total batches processed
- `neo4j_batch.total_nodes`: Total nodes saved
- `neo4j_batch.total_edges`: Total edges saved
- `neo4j_batch.avg_batch_size`: Average batch size
- `neo4j_batch.avg_batch_time`: Average batch processing time
- `neo4j_batch.batch_errors`: Number of batch errors
- `neo4j_batch.error_rate`: Error rate percentage

## 6. Comprehensive Caching Strategy

### Configuration
- `GNN_CACHE_TTL` (default: `3600`): Default cache TTL in seconds
- `GNN_CACHE_MAX_MEMORY_SIZE` (default: `1000`): Maximum in-memory cache entries
- `GNN_CACHE_ENABLE_PERSISTENT` (default: `true`): Enable persistent file-based cache
- `GNN_CACHE_DIR` (default: `./gnn_cache`): Cache directory
- `REDIS_URL`: Redis URL for distributed caching

### Cache Types
- `embedding`: GNN embeddings
- `query`: Query results
- `graph_data`: Graph data
- `classification`: Node classification results
- `link_prediction`: Link prediction results

### Metrics
Access via training service:
- `cache.hits`: Cache hits
- `cache.misses`: Cache misses
- `cache.hit_rate`: Cache hit rate percentage
- `cache.sets`: Cache sets
- `cache.invalidations`: Cache invalidations
- `cache.expired`: Expired entries

## Monitoring Endpoints

### Extract Service
- `GET /metrics/improvements`: Get all improvement metrics

### Training Service
- `GET /gnn/cache/stats`: Get cache statistics
- `GET /gnn/cache/metrics`: Get detailed cache metrics

## Performance Tuning

### Recommended Settings for Large Datasets (>100K nodes)
```bash
export NEO4J_BATCH_SIZE=500
export NEO4J_BATCH_SIZE_LARGE=250
export POSTGRES_RETRY_MAX_ATTEMPTS=5
export GNN_CACHE_TTL=7200
export GNN_CACHE_MAX_MEMORY_SIZE=2000
```

### Recommended Settings for Small Datasets (<10K nodes)
```bash
export NEO4J_BATCH_SIZE=1000
export POSTGRES_RETRY_MAX_ATTEMPTS=3
export GNN_CACHE_TTL=3600
export GNN_CACHE_MAX_MEMORY_SIZE=1000
```

## Troubleshooting

### High Validation Rejection Rate
- Check data source quality
- Review validation error logs
- Adjust `VALIDATION_STRICT_MODE` if needed

### High Retry Failure Rate
- Check storage system connectivity
- Review retry configuration
- Increase `*_RETRY_MAX_ATTEMPTS` if needed

### Consistency Issues
- Check storage system synchronization
- Review consistency check logs
- Adjust `CONSISTENCY_CHECK_VARIANCE_THRESHOLD` if needed

### Low Cache Hit Rate
- Increase `GNN_CACHE_TTL`
- Review cache invalidation patterns
- Check cache size limits

### Neo4j Batch Performance Issues
- Reduce `NEO4J_BATCH_SIZE` for large datasets
- Check Neo4j connection pool settings
- Review transaction timeout settings

