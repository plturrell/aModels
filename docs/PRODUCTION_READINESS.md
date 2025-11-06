# Production Readiness Guide

This document outlines the production readiness features implemented for the aModels platform.

## Table of Contents

1. [Error Handling & Retry Logic](#error-handling--retry-logic)
2. [Rate Limiting](#rate-limiting)
3. [Authentication & Authorization](#authentication--authorization)
4. [Monitoring & Alerting](#monitoring--alerting)
5. [Performance Optimization](#performance-optimization)
6. [Database Indexes](#database-indexes)

---

## Error Handling & Retry Logic

### Features

- **Centralized Error Handling**: All API errors are handled consistently
- **Retry Logic**: Automatic retry for transient failures
- **Panic Recovery**: Automatic panic recovery to prevent service crashes
- **Structured Error Responses**: JSON error responses with details

### Implementation

Error handling is provided via `ErrorHandler` middleware:

```go
errorHandler := api.NewErrorHandler(logger)
handler := errorHandler.RecoveryMiddleware(mux)
```

### Error Response Format

```json
{
  "error": {
    "code": 500,
    "message": "Internal Server Error",
    "details": "Error details",
    "retry_after": 60,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Retryable Errors

The following errors are automatically retryable:
- Network timeouts
- Connection errors
- HTTP 500, 502, 503, 504 errors
- HTTP 429 (rate limit) errors

### Configuration

Default retry configuration:
- Max Retries: 3
- Initial Delay: 100ms
- Max Delay: 5s
- Backoff Multiplier: 2.0

---

## Rate Limiting

### Features

- **Per-IP Rate Limiting**: Individual rate limits per client IP
- **Global Rate Limiting**: Service-wide rate limits
- **Configurable Limits**: Customizable rate limits per endpoint
- **Automatic Cleanup**: Old limiters are cleaned up periodically

### Implementation

Rate limiting uses the token bucket algorithm:

```go
rateLimiter := api.DefaultRateLimiter() // 1000 req/min global, 100 req/min per IP
handler := api.RateLimitMiddleware(rateLimiter, 100.0/60.0, 10)(mux)
```

### Default Limits

- **Global**: 1000 requests per minute
- **Per IP**: 100 requests per minute
- **Burst**: 10 requests

### Rate Limit Response

When rate limit is exceeded:

```json
{
  "error": {
    "code": 429,
    "message": "Too Many Requests",
    "retry_after": 60,
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

HTTP Headers:
```
Retry-After: 60
```

### Customization

Rate limits can be customized per endpoint:

```go
// Higher limit for search endpoint
searchLimiter := api.RateLimitMiddleware(rateLimiter, 200.0/60.0, 20)
mux.HandleFunc("/api/discover/search", searchLimiter(searchHandler))
```

---

## Authentication & Authorization

### Features

- **Token-Based Authentication**: Bearer token authentication
- **Optional Enforcement**: Can be enabled/disabled via environment variable
- **Role-Based Access**: Role-based access control (future enhancement)

### Implementation

Authentication is enabled via environment variable:

```bash
ENABLE_AUTH=true
```

### Token Format

```
Authorization: Bearer <token>
```

### Current Implementation

- Default test token: `test-token` (for development)
- Production: Load tokens from secure configuration

### Future Enhancements

- OAuth2 integration
- JWT token validation
- Role-based access control (RBAC)
- API key management

---

## Monitoring & Alerting

### Features

- **Prometheus Metrics**: Comprehensive metrics collection
- **Request Monitoring**: Track request latency, status codes, sizes
- **Automatic Alerting**: Alerts for high latency and server errors
- **Custom Metrics**: Support for custom business metrics

### Metrics

#### HTTP Request Metrics

- `catalog_request_duration_seconds`: Request duration histogram
- `catalog_requests_total`: Total request count
- `catalog_request_size_bytes`: Request size histogram
- `catalog_response_size_bytes`: Response size histogram

#### Alert Metrics

- `catalog_alerts_total`: Total alert count by type

#### Database Metrics

- `catalog_neo4j_query_duration_seconds`: Neo4j query duration
- `catalog_neo4j_connection_pool_size`: Connection pool size

#### Quality Metrics

- `catalog_data_element_quality_score`: Data element quality scores

### Alert Conditions

Alerts are automatically triggered for:

1. **High Latency**: Requests exceeding 2 seconds
2. **Server Errors**: HTTP 5xx status codes
3. **Error Rate**: Error rate exceeding 5% (future)

### Alert Configuration

Default alert configuration:
- Max Latency: 2 seconds
- Error Threshold: 5%
- Check Interval: 30 seconds

### Prometheus Integration

Metrics are exposed at `/metrics` endpoint:

```bash
curl http://localhost:8084/metrics
```

### Alerting System Integration

In production, alerts can be forwarded to:
- PagerDuty
- Slack
- Email
- Custom webhooks

---

## Performance Optimization

### Database Connection Pooling

- **Neo4j**: Connection pooling configured
- **PostgreSQL**: Connection pooling via pgx
- **Monitoring**: Connection pool metrics available

### Caching

- **Redis Cache**: In-memory caching for frequently accessed data
- **Cache Metrics**: Hit/miss ratios tracked
- **TTL Configuration**: Configurable cache expiration

### Query Optimization

#### Recommendations

1. **Index Frequently Queried Fields**: See Database Indexes section
2. **Use Connection Pooling**: Already configured
3. **Batch Operations**: Use batch endpoints for bulk operations
4. **Pagination**: Use limit/offset for large result sets

#### Performance Best Practices

1. **Avoid N+1 Queries**: Batch related queries
2. **Use Projections**: Only fetch required fields
3. **Leverage Caching**: Cache expensive computations
4. **Monitor Query Performance**: Use Prometheus metrics

### Database Query Optimization

#### Neo4j

- Use indexes for frequently queried properties
- Use `PROFILE` to analyze query performance
- Consider using `LIMIT` and `SKIP` for pagination
- Use `EXPLAIN` to understand query plans

#### PostgreSQL

- Ensure indexes on foreign keys
- Use `EXPLAIN ANALYZE` to identify slow queries
- Consider materialized views for complex aggregations
- Monitor query execution time

---

## Database Indexes

### Recommended Indexes

#### PostgreSQL Indexes

**Data Product Versions Table** (`data_product_versions`):
```sql
CREATE INDEX idx_data_product_versions_product_id ON data_product_versions(product_id);
CREATE INDEX idx_data_product_versions_version ON data_product_versions(version);
CREATE INDEX idx_data_product_versions_created_at ON data_product_versions(created_at);
```

**Discoverability Tables**:
```sql
-- Tags
CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_tags_category ON tags(category);
CREATE INDEX idx_tags_parent_tag_id ON tags(parent_tag_id);

-- Product Tags
CREATE INDEX idx_product_tags_product_id ON product_tags(product_id);
CREATE INDEX idx_product_tags_tag_id ON product_tags(tag_id);
CREATE INDEX idx_product_tags_created_at ON product_tags(created_at);

-- Search History
CREATE INDEX idx_search_history_query ON search_history(query);
CREATE INDEX idx_search_history_timestamp ON search_history(timestamp);

-- Product Usage Stats
CREATE INDEX idx_product_usage_stats_updated_at ON product_usage_stats(updated_at);

-- Access Requests
CREATE INDEX idx_access_requests_product_id ON access_requests(product_id);
CREATE INDEX idx_access_requests_status ON access_requests(status);
CREATE INDEX idx_access_requests_requested_at ON access_requests(requested_at);
```

**Audit Trail Tables**:
```sql
-- LangExtract Audit Trail
CREATE INDEX idx_langextract_audit_trail_timestamp ON langextract_audit_trail(timestamp);
CREATE INDEX idx_langextract_audit_trail_user_id ON langextract_audit_trail(user_id);
CREATE INDEX idx_langextract_audit_trail_operation_type ON langextract_audit_trail(operation_type);
```

**Regulatory Specs Tables**:
```sql
-- Regulatory Schemas
CREATE INDEX idx_regulatory_schemas_regulatory_type ON regulatory_schemas(regulatory_type);
CREATE INDEX idx_regulatory_schemas_version ON regulatory_schemas(version);
CREATE INDEX idx_regulatory_schemas_created_at ON regulatory_schemas(created_at);
CREATE UNIQUE INDEX idx_regulatory_schemas_type_version ON regulatory_schemas(regulatory_type, version);
```

#### Neo4j Indexes

**Node Labels**:
```cypher
CREATE INDEX ON :DataElement(identifier);
CREATE INDEX ON :DataElement(name);
CREATE INDEX ON :DataProduct(id);
CREATE INDEX ON :Tag(name);
CREATE INDEX ON :Tag(category);
```

**Relationships**:
```cypher
CREATE INDEX ON :HAS_TAG(timestamp);
CREATE INDEX ON :HAS_VERSION(timestamp);
```

### Index Maintenance

#### Monitoring

- Monitor index usage via PostgreSQL `pg_stat_user_indexes`
- Monitor Neo4j index usage via `db.indexes()`
- Review slow query logs regularly

#### Maintenance

- Rebuild indexes periodically (PostgreSQL: `REINDEX`)
- Analyze tables for query planner (PostgreSQL: `ANALYZE`)
- Monitor index bloat and rebuild when necessary

---

## Deployment Checklist

### Pre-Deployment

- [ ] Configure rate limits for production
- [ ] Set up authentication tokens
- [ ] Configure monitoring alerts
- [ ] Create database indexes
- [ ] Set up connection pooling
- [ ] Configure caching (Redis)
- [ ] Set up log aggregation
- [ ] Configure error tracking (Sentry, etc.)

### Post-Deployment

- [ ] Monitor error rates
- [ ] Monitor latency metrics
- [ ] Check alert configuration
- [ ] Verify rate limiting is working
- [ ] Monitor database performance
- [ ] Review cache hit rates
- [ ] Check connection pool usage

---

## Configuration

### Environment Variables

```bash
# Authentication
ENABLE_AUTH=true

# Rate Limiting
RATE_LIMIT_GLOBAL_RPS=16.67  # 1000 req/min
RATE_LIMIT_PER_IP_RPS=1.67   # 100 req/min
RATE_LIMIT_BURST=10

# Monitoring
ALERT_MAX_LATENCY_MS=2000
ALERT_ERROR_THRESHOLD=5.0

# Database
POSTGRES_MAX_CONNECTIONS=100
NEO4J_MAX_CONNECTIONS=50

# Caching
REDIS_CACHE_TTL=3600  # 1 hour
```

---

## Troubleshooting

### High Error Rate

1. Check Prometheus metrics for error patterns
2. Review application logs
3. Check database connection pool
4. Verify external service availability
5. Review rate limiting settings

### High Latency

1. Check database query performance
2. Review cache hit rates
3. Check external API response times
4. Review connection pool usage
5. Consider scaling horizontally

### Rate Limit Issues

1. Verify rate limit configuration
2. Check if legitimate traffic is being throttled
3. Consider increasing limits for specific endpoints
4. Review IP detection (X-Forwarded-For header)

---

## Future Enhancements

- [ ] Circuit breaker pattern
- [ ] Distributed rate limiting (Redis-based)
- [ ] OAuth2/JWT authentication
- [ ] Advanced alerting (PagerDuty, Slack integration)
- [ ] Automatic scaling based on metrics
- [ ] Request tracing (OpenTelemetry)
- [ ] Advanced caching strategies
- [ ] Database query optimization advisor

