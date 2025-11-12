# aModels Testing Suite

Comprehensive testing infrastructure for the aModels project, organized by test type for maximum clarity and maintainability.

## Directory Structure

```
tests/
├── integration/          # Integration tests
│   ├── services/        # Service-to-service integration
│   └── workflows/       # Multi-step workflow tests
├── e2e/                 # End-to-end system tests
├── performance/         # Performance, load, and stress tests
├── benchmarks/          # Model evaluation benchmarks
├── domain/              # Domain-specific functionality tests
├── fixtures/            # Test data, helpers, and fixtures
├── manual/              # Manual test scripts and procedures
├── automation/          # Test automation and validation scripts
└── scripts/             # Test execution scripts
```

## Quick Start

### Run All Tests
```bash
cd tests/scripts
./run_all_tests.sh
```

### Run Specific Test Categories
```bash
# Integration tests only
./run_all_tests.sh --integration

# Performance tests only
cd tests/performance
python3 test_performance.py

# Benchmarks
cd tests/benchmarks
go test -v ./...

# Domain tests
cd tests/domain
python3 test_domain_detection.py
```

### Run with Service Health Check
```bash
cd tests/scripts
./run_all_tests.sh --check
```

### Docker Environment
```bash
cd tests/scripts
./run_tests_docker.sh
```

## Test Categories

### 1. Integration Tests (`integration/`)

Tests that verify interactions between multiple services or components.

#### Service Integration (`integration/services/`)
Tests service-to-service communication and dependencies.

**Files**:
- `test_localai_integration.py` - LocalAI service integration
- `test_gnn_agent_integration.py` - GNN agent integration
- `test_service_integrations.go` - Cross-service Go tests
- `test_deepseek_ocr.go` - DeepSeek OCR integration
- `test_embedding_models.go` - Embedding model consistency

**Run**:
```bash
# All service integration tests
cd tests/integration/services
python3 test_localai_integration.py
go test -v .

# Specific service
python3 test_localai_integration.py
```

#### Workflow Integration (`integration/workflows/`)
Tests complete workflows across multiple components.

**Files**:
- `test_extraction_flow.py` - Data extraction pipeline
- `test_training_flow.py` - Model training workflow
- `test_ab_testing_flow.py` - A/B testing workflow
- `test_rollback_flow.py` - Rollback procedures
- `test_extraction_intelligence.py` - Intelligent extraction

**Run**:
```bash
cd tests/integration/workflows
python3 test_extraction_flow.py
```

### 2. End-to-End Tests (`e2e/`)

Complete system tests that verify entire use cases from start to finish.

**Files**:
- `test_sgmi_end_to_end.sh` - SGMI system end-to-end test

**Run**:
```bash
cd tests/e2e
./test_sgmi_end_to_end.sh
```

### 3. Performance Tests (`performance/`)

Load testing, stress testing, and performance benchmarking.

**Files**:
- `test_performance.py` - General performance tests
- `test_load.py` - Load testing
- `test_concurrent_requests.py` - Concurrent request handling
- `test_concurrent_domains.py` - Multi-domain concurrency
- `test_large_graphs.py` - Large graph processing
- `performance_benchmark.py` - Performance benchmarking suite

**Run**:
```bash
cd tests/performance
python3 test_load.py --users 100 --duration 60
python3 test_concurrent_requests.py
```

### 4. Benchmarks (`benchmarks/`)

Model evaluation benchmarks for measuring AI/ML performance.

**Available Benchmarks**:
- `arc/` - Abstraction and Reasoning Corpus
- `boolq/` - Boolean Questions
- `hellaswag/` - Commonsense reasoning
- `piqa/` - Physical Interaction QA
- `socialiq/` - Social Interaction QA
- `triviaqa/` - Trivia questions
- `gsm-symbolic/` - Math reasoning
- `deepseekocr/` - OCR evaluation

**Run**:
```bash
cd tests/benchmarks
go test -v ./boolq
go test -v ./hellaswag

# Run specific benchmark
cd boolq
go test -v -run TestBoolQ
```

**Documentation**: See `benchmarks/README.md`

### 5. Domain Tests (`domain/`)

Tests for domain-specific functionality and features.

**Files**:
- `test_domain_detection.py` - Domain detection algorithms
- `test_domain_filter.py` - Domain filtering logic
- `test_domain_metrics.py` - Domain metrics calculation
- `test_domain_trainer.py` - Domain-specific training
- `test_pattern_learning.py` - Pattern learning algorithms

**Run**:
```bash
cd tests/domain
python3 test_domain_detection.py
python3 test_domain_filter.py
```

### 6. Test Fixtures (`fixtures/`)

Shared test data, helpers, and fixtures used across test suites.

**Files**:
- `test_helpers.py` - Common test helper functions
- `test_data_fixtures.py` - Test data fixtures and generators
- `test_data/` - Static test data files

**Usage**:
```python
from tests.fixtures.test_helpers import create_test_client, mock_response
from tests.fixtures.test_data_fixtures import sample_graph_data
```

### 7. Manual Tests (`manual/`)

Manual testing procedures and scripts that require human interaction.

**Contents**:
- `signavio/` - Signavio integration manual tests

**Use**: Follow specific README in each subdirectory.

### 8. Automation (`automation/`)

Test automation, validation, and quality assurance scripts.

**Files**:
- `test_automation.py` - Automated test generation and execution
- `validate_all_tests.py` - Test validation and verification

**Run**:
```bash
cd tests/automation
python3 validate_all_tests.py
```

### 9. Test Scripts (`scripts/`)

Executable scripts for running tests in various configurations.

**Available Scripts**:

#### Main Test Runners
- `run_all_tests.sh` - Run all tests with comprehensive reporting
- `run_integration_tests.sh` - Run integration tests only
- `run_smoke_tests.sh` - Quick smoke tests
- `run_tests_docker.sh` - Run tests in Docker environment

#### Setup & Configuration
- `setup_test_database.sh` - Initialize test database
- `check_services.sh` - Check service health before tests
- `validate_test_data.sh` - Validate test data integrity

#### Specialized Scripts
- `test_localai_integration.py` - LocalAI integration tester
- `test_deepagents_localai.py` - DeepAgents integration tester
- `test_localai_from_container.sh` - Container-based testing
- `test_all_services.sh` - All services validation

**Usage**:
```bash
cd tests/scripts

# Run all tests
./run_all_tests.sh

# Run with service check first
./run_all_tests.sh --check

# Use localhost URLs instead of Docker
./run_all_tests.sh --host

# Exit on first failure
./run_all_tests.sh --strict

# Docker mode
./run_tests_docker.sh
```

## Environment Configuration

### Docker Environment (Default)
```bash
export LOCALAI_URL="http://localai:8080"
export EXTRACT_SERVICE_URL="http://extract-service:8082"
export TRAINING_SERVICE_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export REDIS_URL="redis://redis:6379/0"
export NEO4J_URI="bolt://neo4j:7687"
```

### Host Environment
```bash
export LOCALAI_URL="http://localhost:8081"
export EXTRACT_SERVICE_URL="http://localhost:19080"
export TRAINING_SERVICE_URL="http://localhost:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
export REDIS_URL="redis://localhost:6379/0"
export NEO4J_URI="bolt://localhost:7687"
```

Scripts in `tests/scripts/` auto-detect and configure appropriately.

## Test Dependencies

### Python Tests
```bash
cd tests
pip install -r requirements.txt
```

### Go Tests
```bash
cd tests
go mod download
```

## Writing New Tests

### Integration Test Template (Python)
```python
#!/usr/bin/env python3
"""Integration test for <feature>."""

import os
import requests
from tests.fixtures.test_helpers import check_service_health

def test_feature_integration():
    """Test <feature> integration."""
    service_url = os.getenv("SERVICE_URL", "http://localhost:8080")
    
    # Health check
    if not check_service_health(service_url):
        print("⚠️  Service not available, skipping test")
        return
    
    # Test logic here
    response = requests.post(f"{service_url}/endpoint", json={...})
    assert response.status_code == 200
    
    print("✅ Test passed")

if __name__ == "__main__":
    test_feature_integration()
```

### Integration Test Template (Go)
```go
package services_test

import (
    "testing"
    "net/http"
)

func TestServiceIntegration(t *testing.T) {
    serviceURL := os.Getenv("SERVICE_URL")
    if serviceURL == "" {
        serviceURL = "http://localhost:8080"
    }
    
    // Test logic
    resp, err := http.Get(serviceURL + "/health")
    if err != nil {
        t.Fatalf("Failed to connect: %v", err)
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        t.Errorf("Expected 200, got %d", resp.StatusCode)
    }
}
```

## Best Practices

### Test Organization
- ✅ **Unit tests**: Co-located with source code in service directories
- ✅ **Integration tests**: Place in `tests/integration/`
- ✅ **E2E tests**: Place in `tests/e2e/`
- ✅ **Performance tests**: Place in `tests/performance/`

### Test Naming
- **Python**: `test_<feature>.py`
- **Go**: `<feature>_test.go` (unit), `<feature>_integration_test.go` (integration)
- **Shell**: `test_<feature>.sh`

### Test Independence
- Each test should be runnable independently
- Use fixtures for shared setup
- Clean up resources after tests
- Don't depend on test execution order

### Service Dependencies
- Always check service health before testing
- Use environment variables for service URLs
- Provide clear skip messages for unavailable services
- Include retry logic for transient failures

## Troubleshooting

### Services Not Available
```bash
# Check service health
cd tests/scripts
./check_services.sh

# Start services
docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d
```

### Test Failures
```bash
# Run with verbose output
python3 -v test_name.py
go test -v ./...

# Check service logs
docker-compose -f infrastructure/docker/brev/docker-compose.yml logs <service>
```

### Environment Issues
```bash
# Verify environment variables
cd tests/scripts
./run_all_tests.sh --check

# Use host mode if Docker networking issues
./run_all_tests.sh --host
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Run Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Start Services
        run: docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d
      
      - name: Wait for Services
        run: sleep 30
      
      - name: Run Tests
        run: |
          cd tests/scripts
          ./run_all_tests.sh --check
      
      - name: Cleanup
        if: always()
        run: docker-compose -f infrastructure/docker/brev/docker-compose.yml down
```

## Makefile Integration

Use Makefile targets for common test operations:

```bash
# From project root
make test                # Run all tests
make test-integration    # Integration tests only
make test-performance    # Performance tests only
make test-benchmarks     # Benchmarks only
```

See `/Makefile.services` for all available targets.

## Migration from Old Structure

Files were migrated from:
- `/testing/` → `/tests/<category>/`
- `/scripts/testing/` → `/tests/scripts/`

See `TESTING_AUDIT.md` for full migration details.

## Additional Documentation

- **Benchmarks**: `tests/benchmarks/README.md`
- **Integration Guide**: `tests/integration/README.md`
- **Testing Strategy**: `/docs/TESTING_STRATEGY.md`
- **Testing Guide**: `/docs/TESTING_GUIDE.md`

## Support

For questions or issues:
1. Check this README
2. Review `/TESTING_AUDIT.md`
3. Check service-specific test documentation
4. Review test scripts for examples

## Contributing

When adding new tests:
1. Choose the appropriate category
2. Follow naming conventions
3. Include documentation
4. Add to CI/CD if applicable
5. Update this README if adding new patterns
