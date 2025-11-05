# Running All Tests - Complete Guide

## Current Status

✅ **All 21 test files are syntactically valid**
✅ **Test infrastructure is complete**
⚠️ **Some tests require services to be running**

## Test Files Summary

### Week 1: Foundation Tests
- `test_domain_detection.py` (377 lines) - 6 tests
- `test_domain_filter.py` (412 lines) - 7 tests  
- `test_domain_trainer.py` (362 lines) - 7 tests
- `test_domain_metrics.py` (321 lines) - 6 tests

### Week 2: Integration Tests
- `test_helpers.py` (241 lines) - Shared utilities
- `test_extraction_flow.py` (484 lines) - 7 tests
- `test_training_flow.py` (420 lines) - 7 tests
- `test_ab_testing_flow.py` (392 lines) - 6 tests
- `test_rollback_flow.py` (376 lines) - 6 tests

### Week 3: Phase 7-9 Tests
- `test_pattern_learning.py` (419 lines) - 8 tests
- `test_extraction_intelligence.py` (351 lines) - 8 tests
- `test_automation.py` (339 lines) - 8 tests

### Week 4: Performance Tests
- `test_performance.py` (344 lines) - 6 tests
- `test_load.py` (368 lines) - 5 tests
- `test_concurrent_requests.py` (386 lines) - 5 tests
- `performance_benchmark.py` (339 lines) - 5 benchmarks

**Total: 7,000+ lines of test code, 97+ tests**

## Quick Start

### 1. Validate All Test Files
```bash
cd /home/aModels
python3 testing/validate_all_tests.py
```

### 2. Check Services Status
```bash
docker compose -f infrastructure/docker/brev/docker-compose.yml ps
```

### 3. Run Tests

#### Option A: From Host (localhost URLs)
```bash
export LOCALAI_URL="http://localhost:8081"
export EXTRACT_SERVICE_URL="http://localhost:19080"
python3 testing/test_domain_detection.py
```

#### Option B: From Docker Container (service names)
```bash
docker exec -it training-shell bash
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:19080"
cd /workspace
python3 testing/test_domain_detection.py
```

## Test Results Analysis

### ✅ Tests That Work (Module-Level)
- All test files are syntactically valid
- Module imports work (when dependencies available)
- Test structure is correct
- Error handling is in place

### ⚠️ Tests That Need Services
- **Domain detection tests**: Need LocalAI running
- **Extraction flow tests**: Need Extract service running
- **Training flow tests**: Need Training service running
- **Integration tests**: Need multiple services

### Expected Behavior
- Tests **skip gracefully** when services unavailable
- Tests **pass** when services are running correctly
- Tests provide **clear error messages**

## Service Requirements

| Test Suite | Required Services |
|------------|-------------------|
| Domain Detection | LocalAI |
| Domain Filter | LocalAI (optional) |
| Domain Trainer | LocalAI, PostgreSQL |
| Domain Metrics | LocalAI, PostgreSQL |
| Extraction Flow | LocalAI, Extract Service |
| Training Flow | LocalAI, Extract, Training |
| A/B Testing | PostgreSQL, Redis |
| Rollback | PostgreSQL, Redis, LocalAI |
| Pattern Learning | LocalAI (optional) |
| Performance | LocalAI, Extract |

## Running Full Test Suite

### From Host
```bash
cd /home/aModels
./testing/run_all_tests_fixed.sh
```

### From Docker
```bash
docker exec -it training-shell bash
cd /workspace
./testing/run_tests_from_docker.sh
```

## Known Issues & Fixes

### Issue: Connection Refused
**Cause**: Services not accessible from host  
**Fix**: Use Docker service names when running from container

### Issue: Module Import Errors
**Cause**: Missing dependencies (PyTorch Geometric, etc.)  
**Fix**: Install dependencies or tests will skip gracefully

### Issue: Tests Fail But Services Running
**Cause**: Wrong URL (localhost vs service name)  
**Fix**: Use correct URL based on execution environment

## Test Execution Strategy

1. **Start Services**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
   ```

2. **Wait for Health**
   ```bash
   sleep 30
   ```

3. **Validate Tests**
   ```bash
   python3 testing/validate_all_tests.py
   ```

4. **Run Test Suites**
   ```bash
   # Week 1
   python3 testing/test_domain_detection.py
   python3 testing/test_domain_filter.py
   python3 testing/test_domain_trainer.py
   python3 testing/test_domain_metrics.py
   
   # Week 2
   python3 testing/test_extraction_flow.py
   python3 testing/test_training_flow.py
   python3 testing/test_ab_testing_flow.py
   python3 testing/test_rollback_flow.py
   
   # Week 3
   python3 testing/test_pattern_learning.py
   python3 testing/test_extraction_intelligence.py
   python3 testing/test_automation.py
   
   # Week 4
   python3 testing/test_performance.py
   python3 testing/test_load.py
   python3 testing/test_concurrent_requests.py
   python3 testing/performance_benchmark.py
   ```

## Success Criteria

### Test Files ✅
- All 21 test files are valid
- All imports work (when dependencies available)
- All test structures are correct

### Test Execution ⚠️
- Tests run without syntax errors
- Tests skip gracefully when services unavailable
- Tests provide clear feedback

### Services Required 🔧
- LocalAI must be running for most tests
- Extract service needed for extraction tests
- PostgreSQL/Redis needed for advanced features

## Next Steps

1. **Ensure all services are running**
2. **Run test validation** (already done ✅)
3. **Run individual test suites** to verify functionality
4. **Fix any service connectivity issues**
5. **Run full test suite** when services are ready

---

**Status**: All test files created and validated ✅  
**Ready**: Yes, when services are running  
**Total Tests**: 97+ comprehensive tests

