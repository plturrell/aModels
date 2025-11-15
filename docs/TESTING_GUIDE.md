# aModels Service Testing Guide

Complete guide for testing all aModels services individually and as an integrated system.

## Quick Start

### Run All Tests

```bash
# Full system test (starts services and runs all tests)
./scripts/testing/test_full_system.sh yes

# Test assuming services are already running
./scripts/testing/test_full_system.sh no
```

### Test Individual Service

```bash
./scripts/testing/test_service_individual.sh <service_name>
```

### Run Health Checks

```bash
./scripts/system/health-check.sh
```

## Test Scripts Overview

### 1. Individual Service Tests

**Script**: `scripts/testing/test_service_individual.sh`

Tests a single service to verify:
- Port is accessible
- Health endpoint responds
- Basic connectivity

**Usage**:
```bash
./scripts/testing/test_service_individual.sh redis
./scripts/testing/test_service_individual.sh localai
./scripts/testing/test_service_individual.sh catalog
```

**Available Services**:
- Infrastructure: `redis`, `postgres`, `neo4j`, `elasticsearch`, `gitea`
- Core: `localai`, `catalog`
- Application: `extract`, `graph`, `search`, `deepagents`, `runtime`, `orchestration`, `training`, `regulatory_audit`, `telemetry_exporter`, `gateway`

### 2. Test All Services Individually

**Script**: `scripts/testing/test_all_services_individual.sh`

Runs individual tests for all services sequentially.

**Usage**:
```bash
./scripts/testing/test_all_services_individual.sh
```

**Output**: 
- Test results for each service
- Summary with pass/fail counts
- Logs in `logs/testing/`

### 3. Functional Tests

**Script**: `scripts/testing/test_service_functionality.sh`

Tests actual functionality of services beyond health checks:
- API endpoints
- Data operations
- Service-specific features

**Usage**:
```bash
./scripts/testing/test_service_functionality.sh localai
./scripts/testing/test_service_functionality.sh catalog
```

**What It Tests**:
- **LocalAI**: Models endpoint, completions endpoint
- **Catalog**: Health, API endpoints
- **Extract**: Health, extraction endpoints
- **Graph**: Health, graph queries
- **Search**: Health, search queries
- **DeepAgents**: Health, agent operations
- **Runtime**: Health, analytics
- **Orchestration**: Health, workflows
- **Training**: Health, training jobs
- **Regulatory Audit**: Health, audits
- **Gateway**: Health, routing
- **Telemetry Exporter**: Health, export operations

### 4. Integration Tests

**Script**: `scripts/testing/test_service_integration.sh`

Tests service-to-service communication and workflows.

**Usage**:
```bash
./scripts/testing/test_service_integration.sh
```

**Integration Tests**:
1. Catalog → Extract
2. Graph → LocalAI
3. Search → Elasticsearch → LocalAI
4. DeepAgents → Extract → LocalAI
5. Runtime → Catalog
6. Orchestration → Multiple Services
7. Extract → Gitea
8. Telemetry Exporter → Extract
9. End-to-end: Extract → Catalog → Graph
10. End-to-end: Search → Elasticsearch → LocalAI

### 5. Full System Test

**Script**: `scripts/testing/test_full_system.sh`

Comprehensive test that:
- Checks for port conflicts
- Checks system resources
- Optionally starts all services
- Runs health checks
- Runs individual tests
- Runs functional tests
- Runs integration tests

**Usage**:
```bash
# Start services and test
./scripts/testing/test_full_system.sh yes

# Test assuming services are running
./scripts/testing/test_full_system.sh no
```

## Health Check System

### Enhanced Health Check Script

**Script**: `scripts/system/health-check.sh`

Comprehensive health checking for all services.

**Usage**:
```bash
./scripts/system/health-check.sh
```

**Features**:
- Checks all registered services
- Detailed checks for critical services
- Docker container health
- System resource monitoring
- Color-coded output

**Services Checked**:
- Infrastructure: Redis, PostgreSQL, Neo4j, Elasticsearch, Gitea
- Core: LocalAI, Transformers, Catalog
- Application: Extract, Graph, Search, DeepAgents, Runtime, Orchestration, Training, Regulatory Audit, Telemetry Exporter, Gateway
- Special: PostgreSQL Lang (gRPC)

## Testing Workflow

### Recommended Testing Order

1. **Pre-flight Checks**
   ```bash
   # Check for port conflicts
   ./scripts/testing/test_full_system.sh no | grep -i conflict
   
   # Check system resources
   free -h
   df -h
   ```

2. **Start Services** (if needed)
   ```bash
   PROFILE=full ./scripts/system/start-system.sh start
   ```

3. **Health Checks**
   ```bash
   ./scripts/system/health-check.sh
   ```

4. **Individual Service Tests**
   ```bash
   ./scripts/testing/test_all_services_individual.sh
   ```

5. **Functional Tests**
   ```bash
   for service in localai catalog extract graph search; do
       ./scripts/testing/test_service_functionality.sh $service
   done
   ```

6. **Integration Tests**
   ```bash
   ./scripts/testing/test_service_integration.sh
   ```

7. **Full System Test**
   ```bash
   ./scripts/testing/test_full_system.sh no
   ```

## Test Results

### Log Locations

All test logs are written to: `logs/testing/`

**Log Files**:
- `startup.log` - Service startup output
- `health_check.log` - Health check results
- `individual_tests.log` - Individual test results
- `functional_tests.log` - Functional test results
- `integration_tests.log` - Integration test results
- `<service>.log` - Per-service test logs

### Interpreting Results

**Success Indicators**:
- ✓ Green checkmarks
- "PASS" status
- Exit code 0

**Failure Indicators**:
- ✗ Red X marks
- "FAIL" status
- Exit code non-zero
- Error messages in logs

## Troubleshooting

### Service Not Starting

1. Check logs: `logs/startup/` or `logs/testing/startup.log`
2. Check port conflicts: `lsof -i :<port>` or `netstat -tlnp | grep <port>`
3. Check dependencies: Ensure required services are running
4. Check resources: `free -h`, `df -h`

### Health Check Failing

1. Verify service is running: `./scripts/testing/test_service_individual.sh <service>`
2. Check health endpoint manually: `curl http://localhost:<port>/health`
3. Check service logs for errors
4. Verify dependencies are healthy

### Integration Test Failing

1. Verify all services in the workflow are running
2. Check service-to-service connectivity
3. Verify network configuration
4. Check service logs for connection errors

### Port Conflicts

1. Identify conflicting services: `lsof -i :<port>`
2. Check service configuration for port settings
3. Update service registry with correct ports
4. Restart services with correct configuration

## Best Practices

1. **Run tests in order**: Individual → Functional → Integration → Full System
2. **Check logs**: Always review test logs for detailed error information
3. **Start fresh**: Stop all services before running full system test
4. **Verify dependencies**: Ensure infrastructure services are running first
5. **Monitor resources**: Check system resources before running full test suite

## Continuous Integration

### CI/CD Integration

The test scripts can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Service Tests
  run: |
    ./scripts/testing/test_full_system.sh no
    
- name: Check Test Results
  run: |
    if [ $? -ne 0 ]; then
      echo "Tests failed"
      exit 1
    fi
```

### Scheduled Health Checks

Set up cron job for regular health checks:

```bash
# Add to crontab
0 */6 * * * /path/to/scripts/system/health-check.sh >> /path/to/logs/health-cron.log 2>&1
```

## Additional Resources

- **Service Inventory**: `docs/SERVICE_INVENTORY.md`
- **Testing Report**: `docs/SERVICE_TESTING_REPORT.md`
- **Service Startup Guide**: `docs/SERVICES_STARTUP.md`
- **Service Configuration**: `config/services.yaml`

## Support

For issues or questions:
1. Check test logs in `logs/testing/`
2. Review service-specific documentation
3. Check service startup logs
4. Review health check output
