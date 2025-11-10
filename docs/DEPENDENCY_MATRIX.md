# Third-Party Library Dependency Matrix

This document provides a comprehensive overview of all third-party library dependencies used across the services, including versions, compatibility information, and update procedures.

## Overview

This matrix tracks all third-party libraries used in the aModels services, ensuring version consistency, compatibility, and providing guidance for updates.

## Library Versions

### Arrow (Apache Arrow)

| Service | Version | Package | Status |
|---------|---------|---------|--------|
| extract | v18.4.1 | `github.com/apache/arrow-go/v18` | ✅ Standardized |
| postgres | v18.4.1 | `github.com/apache/arrow-go/v18` | ✅ Standardized |
| graph | v18.4.1 | `github.com/apache/arrow-go/v18` | ✅ Standardized |
| agentflow | v18.4.1 | `github.com/apache/arrow-go/v18` | ✅ Standardized |

**Usage**: Arrow Flight for high-performance inter-service data transfer.

**Replace Directive Pattern**:
```go
replace github.com/apache/arrow-go/v18 => ../../infrastructure/third_party/go-arrow
```

**Update Procedure**:
1. Update version in all `go.mod` files
2. Update imports from `github.com/apache/arrow/go/v16` to `github.com/apache/arrow-go/v18`
3. Run `go mod tidy` in each service
4. Test Arrow Flight protocol compatibility

### Elasticsearch

| Service | Language | Version | Package | Status |
|---------|----------|---------|---------|--------|
| search-inference | Go | v7.17.10 | `github.com/elastic/go-elasticsearch/v7` | ✅ Documented |
| python_service | Python | v8.x | `elasticsearch` | ✅ Documented |

**Note**: Version difference is intentional:
- Go service uses v7 for compatibility with Go 1.18
- Python service uses v8 for modern Python library compatibility
- Both are compatible with Elasticsearch clusters v7.x and v8.x

**Update Procedure**:
- Go: Update `go.mod` and test compatibility
- Python: Update `requirements.txt` or `pyproject.toml`

### Goose (Database Migrations)

| Service | Version | Package | Status |
|---------|---------|---------|--------|
| extract | v3.21.1 | `github.com/pressly/goose/v3` | ✅ Standardized |
| catalog | v3.21.1 | `github.com/pressly/goose/v3` | ✅ Standardized |

**Replace Directive Pattern**:
```go
replace github.com/pressly/goose/v3 => ./third_party/goose
```

### LangChain / LangGraph

| Service | Language | Packages | Status |
|---------|----------|----------|--------|
| deepagents | Python | `langchain>=0.3.0`, `langchain-core>=0.3.0`, `langgraph>=0.2.0` | ✅ Configured |
| graph | Python | `langchain`, `langgraph` | ✅ Configured |

**Usage**: LLM-powered applications, narrative intelligence, agent orchestration.

**Update Procedure**:
- Update `requirements.txt` or `pyproject.toml`
- Test agent workflows after updates
- Check for breaking changes in LangChain/LangGraph changelogs

### Orchestration

| Service | Version | Package | Status |
|---------|---------|---------|--------|
| extract | local | `github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration` | ✅ Standardized |
| graph | local | `github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration` | ✅ Standardized |

**Replace Directive Pattern**:
```go
replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration => ../../infrastructure/third_party/orchestration
```

## Compatibility Matrix

### Arrow Flight Compatibility

| Client Version | Server Version | Compatible |
|----------------|----------------|------------|
| v18.4.1 | v18.4.1 | ✅ Yes |
| v18.x | v18.x | ✅ Yes |
| v16.x | v18.x | ❌ No (breaking changes) |

### Elasticsearch Compatibility

| Client Version | Cluster Version | Compatible |
|----------------|-----------------|------------|
| Go v7 | ES v7.x | ✅ Yes |
| Go v7 | ES v8.x | ✅ Yes (backward compatible) |
| Python v8 | ES v7.x | ✅ Yes (backward compatible) |
| Python v8 | ES v8.x | ✅ Yes |

## Replace Directive Patterns

### Standard Patterns

1. **Third-party libraries in infrastructure**:
   ```go
   replace github.com/SAP/go-hdb => ../../infrastructure/third_party/go-hdb
   replace github.com/apache/arrow-go/v18 => ../../infrastructure/third_party/go-arrow
   ```

2. **Local services**:
   ```go
   replace github.com/plturrell/aModels/services/postgres => ../postgres
   ```

3. **Orchestration library**:
   ```go
   replace github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration => ../../infrastructure/third_party/orchestration
   ```

## Update Procedures

### Major Version Updates

1. **Review Breaking Changes**: Check library changelog for breaking changes
2. **Update Dependencies**: Update `go.mod` or `requirements.txt`
3. **Update Code**: Fix any API changes
4. **Test Thoroughly**: Run integration tests
5. **Update Documentation**: Update this matrix and service READMEs

### Minor/Patch Updates

1. **Update Dependencies**: Update version numbers
2. **Run Tests**: Ensure no regressions
3. **Update Matrix**: Update version numbers in this document

## Breaking Change Tracking

### Arrow v16 → v18

- **Breaking**: Import path changed from `github.com/apache/arrow/go/v16` to `github.com/apache/arrow-go/v18`
- **Migration**: Update all import statements
- **Status**: ✅ Completed

### Elasticsearch v7 → v8 (Go)

- **Status**: Not migrated (intentionally using v7 for Go 1.18 compatibility)
- **Documentation**: See `services/search/search-inference/README.md`

## Shared Libraries

### Connection Pooling

- **Location**: `services/shared/pkg/pools/flight_pool.go`
- **Usage**: Arrow Flight connection pooling
- **Dependencies**: `github.com/apache/arrow-go/v18/arrow/flight`

### Retry Logic

- **Location**: `services/shared/pkg/retry/retry.go`
- **Usage**: Generic retry with exponential backoff
- **Dependencies**: None (standard library only)

### Circuit Breaker

- **Location**: `services/shared/pkg/circuitbreaker/breaker.go`
- **Usage**: Circuit breaker pattern for resilience
- **Dependencies**: `github.com/sony/gobreaker`

### Caching

- **Location**: `services/shared/pkg/cache/cache.go`
- **Usage**: Multi-level cache (memory + Redis)
- **Dependencies**: `github.com/redis/go-redis/v9`

### LLM Pool (Python)

- **Location**: `services/shared/python/llm_pool.py`
- **Usage**: LLM client pooling, rate limiting, caching
- **Dependencies**: `redis` (optional), `langchain`

## Monitoring and Metrics

All third-party library interactions are instrumented with Prometheus metrics:

- Connection pool utilization
- Retry attempt counts
- Circuit breaker state transitions
- Cache hit/miss rates
- API response times (p50, p95, p99)
- Error rates by library

See `services/shared/pkg/metrics/third_party_metrics.go` for details.

## Health Checks

Health checks are available for:

- Arrow Flight servers
- Elasticsearch clusters
- Connection pools
- LLM services

See `services/shared/pkg/health/third_party_health.go` for implementation.

## Testing

Integration tests are available for:

- Connection pools (`services/shared/pkg/pools/flight_pool_test.go`)
- Retry logic (`services/shared/pkg/retry/retry_test.go`)
- Circuit breakers (`services/shared/pkg/circuitbreaker/breaker_test.go`)
- Caching (`services/shared/pkg/cache/cache_test.go`)

## Maintenance Schedule

- **Weekly**: Review for security updates
- **Monthly**: Review for minor version updates
- **Quarterly**: Review for major version updates and breaking changes

## Contact

For questions or issues with third-party library integration, contact the platform team or create an issue in the repository.

