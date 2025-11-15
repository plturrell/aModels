# Service Testing Scripts

Quick reference guide for testing aModels services.

## Quick Start

### Test a Single Service
```bash
./test_service_individual.sh <service_name>
```

### Test All Services
```bash
./test_all_services_individual.sh
```

### Run Health Checks
```bash
../system/health-check.sh
```

### Run Full System Test
```bash
./test_full_system.sh [yes|no]
# yes = start services first
# no = assume services are running
```

## Available Services

### Infrastructure
- `redis` - Redis cache
- `postgres` or `postgresql` - PostgreSQL database
- `neo4j` - Neo4j graph database
- `elasticsearch` - Elasticsearch search engine
- `gitea` - Gitea git repository

### Core Services
- `localai` - LocalAI model server
- `catalog` - Catalog service

### Application Services
- `extract` - Extract service
- `graph` - Graph service
- `search` - Search service
- `deepagents` - DeepAgents service
- `runtime` - Runtime analytics
- `orchestration` - Orchestration service
- `training` - Training service
- `regulatory_audit` or `regulatory` - Regulatory audit
- `telemetry_exporter` or `telemetry` - Telemetry exporter
- `gateway` - API Gateway

## Test Scripts

### 1. test_service_individual.sh
Tests a single service in isolation.

**Usage:**
```bash
./test_service_individual.sh redis
./test_service_individual.sh localai
```

**What it tests:**
- Port accessibility
- Health endpoint
- Basic connectivity
- Docker container status (if applicable)

### 2. test_all_services_individual.sh
Tests all services sequentially.

**Usage:**
```bash
./test_all_services_individual.sh
```

**Output:**
- Individual test results for each service
- Summary with pass/fail counts
- Logs in `../../logs/testing/`

### 3. test_service_functionality.sh
Tests actual functionality of services.

**Usage:**
```bash
./test_service_functionality.sh localai
./test_service_functionality.sh catalog
```

**What it tests:**
- API endpoints
- Data operations
- Service-specific features

### 4. test_service_integration.sh
Tests service-to-service communication.

**Usage:**
```bash
./test_service_integration.sh
```

**What it tests:**
- Service dependencies
- Service-to-service communication
- End-to-end workflows

### 5. test_full_system.sh
Complete system test suite.

**Usage:**
```bash
# Start services and test
./test_full_system.sh yes

# Test assuming services are running
./test_full_system.sh no
```

**What it does:**
- Port conflict detection
- System resource checks
- Optional service startup
- Runs all test suites

## Test Results

All test logs are written to: `../../logs/testing/`

**Log Files:**
- `startup.log` - Service startup
- `health_check.log` - Health check results
- `individual_tests.log` - Individual test results
- `functional_tests.log` - Functional test results
- `integration_tests.log` - Integration test results
- `<service>.log` - Per-service logs

## Docker Support

All test scripts support services running in Docker containers. The scripts will:
1. Check for Docker containers first
2. Use Docker exec for container-based checks
3. Fall back to direct port checks for native services

## Examples

### Test Infrastructure Services
```bash
for service in redis postgres neo4j elasticsearch gitea; do
    ./test_service_individual.sh $service
done
```

### Test Core Services
```bash
./test_service_individual.sh localai
./test_service_individual.sh catalog
```

### Run Complete Test Suite
```bash
# 1. Health check
../system/health-check.sh

# 2. Individual tests
./test_all_services_individual.sh

# 3. Functional tests
for service in localai catalog extract; do
    ./test_service_functionality.sh $service
done

# 4. Integration tests
./test_service_integration.sh

# 5. Full system test
./test_full_system.sh no
```

## Troubleshooting

### Service Not Found
- Check if service is running: `docker ps` or check ports
- Verify service name spelling
- Check service logs

### Port Not Accessible
- Services in Docker may not be accessible from host
- Tests will try Docker exec first, then fall back to port checks
- Check Docker network configuration

### Test Fails
- Check service logs
- Verify dependencies are running
- Check system resources
- Review test log files in `../../logs/testing/`

## See Also

- `../../docs/TESTING_GUIDE.md` - Complete testing guide
- `../../docs/SERVICE_INVENTORY.md` - Service inventory
- `../../docs/SERVICE_TESTING_REPORT.md` - Testing report
- `../system/health-check.sh` - Health check script

