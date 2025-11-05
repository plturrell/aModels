# Test Execution Guide

## Quick Start

### Option 1: Run All Tests (Recommended)
```bash
cd /home/aModels
./testing/run_all_tests_fixed.sh
```

### Option 2: Run Tests from Docker Container
```bash
docker exec -it training-shell bash
cd /workspace
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
python3 testing/test_domain_detection.py
```

### Option 3: Run Individual Test Suites
```bash
# Week 1: Foundation Tests
python3 testing/test_domain_detection.py
python3 testing/test_domain_filter.py
python3 testing/test_domain_trainer.py
python3 testing/test_domain_metrics.py

# Week 2: Integration Tests
python3 testing/test_extraction_flow.py
python3 testing/test_training_flow.py
python3 testing/test_ab_testing_flow.py
python3 testing/test_rollback_flow.py

# Week 3: Phase 7-9 Tests
python3 testing/test_pattern_learning.py
python3 testing/test_extraction_intelligence.py
python3 testing/test_automation.py

# Week 4: Performance Tests
python3 testing/test_performance.py
python3 testing/test_load.py
python3 testing/test_concurrent_requests.py
python3 testing/performance_benchmark.py
```

## Environment Variables

### For Host Machine (localhost)
```bash
export LOCALAI_URL="http://localhost:8081"
export EXTRACT_SERVICE_URL="http://localhost:19080"
export TRAINING_SERVICE_URL="http://localhost:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
export REDIS_URL="redis://localhost:6379/0"
export NEO4J_URI="bolt://localhost:7687"
```

### For Docker Container (service names)
```bash
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"
```

## Prerequisites

1. **Start Services**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
   ```

2. **Wait for Services**
   ```bash
   sleep 30  # Wait for services to be ready
   ```

3. **Verify Services**
   ```bash
   # From host
   curl http://localhost:8081/health
   
   # From Docker
   docker exec localai wget -q -O- http://localhost:8080/health
   ```

## Expected Test Results

### Tests Pass When:
- ✅ Services are running and accessible
- ✅ Domain modules are importable
- ✅ Database connections work (if configured)
- ✅ All functionality works correctly

### Tests Skip When:
- ⏭️ Service is not running (with warning)
- ⏭️ Module not found (when running outside service)
- ⏭️ Database not configured

### Tests Fail When:
- ❌ Service is misconfigured
- ❌ Module import fails
- ❌ Required functionality missing
- ❌ Integration points broken

## Troubleshooting

### Issue: Connection Refused
**Solution**: Check if services are running and accessible
```bash
docker compose -f infrastructure/docker/brev/docker-compose.yml ps
curl http://localhost:8081/health
```

### Issue: Module Not Found
**Solution**: Ensure Python path includes service directories
```bash
export PYTHONPATH="/home/aModels/services/training:$PYTHONPATH"
```

### Issue: Tests Fail from Host
**Solution**: Run tests from Docker container where services are accessible
```bash
docker exec -it training-shell bash
cd /workspace
./testing/run_tests_from_docker.sh
```

## Test Coverage Summary

- **Week 1**: 26 foundation tests
- **Week 2**: 26 integration tests
- **Week 3**: 24 Phase 7-9 tests
- **Week 4**: 21 performance/load tests

**Total: 97 comprehensive tests**

## Next Steps After Tests

1. Review test results
2. Fix any failures
3. Address skipped tests (if services need to be started)
4. Re-run tests after fixes
5. Check performance benchmarks against baselines

