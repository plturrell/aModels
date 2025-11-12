# Integration Tests

Integration tests verify that multiple components work together correctly. These tests check service-to-service communication, data flow, and system interactions.

## Organization

### Services (`services/`)
Tests individual service integrations and cross-service communication.

**Key Tests**:
- **LocalAI Integration** (`test_localai_integration.py`): Verifies LocalAI service endpoints, model availability, and integration with other services
- **GNN Agent Integration** (`test_gnn_agent_integration.py`): Tests GNN (Graph Neural Network) agent functionality and graph operations
- **Service Integrations** (`test_service_integrations.go`): Go-based cross-service integration tests
- **DeepSeek OCR** (`test_deepseek_ocr.go`): OCR service integration and functionality
- **Embedding Models** (`test_embedding_models.go`): Embedding model consistency and availability across services

### Workflows (`workflows/`)
Tests complete multi-step workflows that span multiple services.

**Key Workflows**:
- **Extraction Flow** (`test_extraction_flow.py`): Data extraction pipeline from ingestion to output
- **Training Flow** (`test_training_flow.py`): Model training workflow including data prep, training, and validation
- **A/B Testing Flow** (`test_ab_testing_flow.py`): A/B testing infrastructure and experiment management
- **Rollback Flow** (`test_rollback_flow.py`): System rollback and recovery procedures
- **Extraction Intelligence** (`test_extraction_intelligence.py`): Intelligent extraction with ML enhancements

## Running Integration Tests

### All Integration Tests
```bash
# From project root
cd tests/integration

# Run all service integration tests
cd services
python3 test_localai_integration.py
python3 test_gnn_agent_integration.py
go test -v .

# Run all workflow tests
cd ../workflows
python3 test_extraction_flow.py
python3 test_training_flow.py
python3 test_ab_testing_flow.py
```

### Specific Tests
```bash
# LocalAI integration
cd tests/integration/services
python3 test_localai_integration.py

# Training workflow
cd tests/integration/workflows
python3 test_training_flow.py
```

### Using Test Runner
```bash
cd tests/scripts
./run_integration_tests.sh
```

## Prerequisites

### Required Services
Integration tests require running services. Start them with:

```bash
docker-compose -f infrastructure/docker/brev/docker-compose.yml up -d
```

### Service Health Check
Always check services are ready:

```bash
cd tests/scripts
./check_services.sh
```

## Environment Variables

### Docker Environment (Default)
```bash
export LOCALAI_URL="http://localai:8080"
export DEEPAGENTS_URL="http://deepagents:9004"
export GRAPH_URL="http://graph-service:8080"
export EXTRACT_URL="http://extract-service:8082"
export TRAINING_URL="http://training-service:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@postgres:5432/amodels"
export NEO4J_URI="bolt://neo4j:7687"
export REDIS_URL="redis://redis:6379/0"
```

### Host Environment
```bash
export LOCALAI_URL="http://localhost:8081"
export DEEPAGENTS_URL="http://localhost:9004"
export GRAPH_URL="http://localhost:8080"
export EXTRACT_URL="http://localhost:19080"
export TRAINING_URL="http://localhost:8080"
export POSTGRES_DSN="postgresql://postgres:postgres@localhost:5432/amodels"
export NEO4J_URI="bolt://localhost:7687"
export REDIS_URL="redis://localhost:6379/0"
```

## Test Patterns

### Service Integration Test Template

```python
#!/usr/bin/env python3
"""Integration test for <Service> integration."""

import os
import sys
import requests

# Configuration
SERVICE_URL = os.getenv("SERVICE_URL", "http://localhost:8080")
DEPENDENT_SERVICE_URL = os.getenv("DEPENDENT_URL", "http://localhost:8081")

def check_health(service_url, service_name):
    """Check if service is healthy."""
    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ {service_name} is healthy")
            return True
        else:
            print(f"⚠️  {service_name} returned {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {service_name} not available: {e}")
        return False

def test_service_integration():
    """Test service integration."""
    print("Testing service integration...")
    
    # Check dependencies
    if not check_health(SERVICE_URL, "Service"):
        print("⚠️  Skipping test - Service not available")
        return
    
    if not check_health(DEPENDENT_SERVICE_URL, "Dependent Service"):
        print("⚠️  Skipping test - Dependent service not available")
        return
    
    # Test integration points
    # 1. Test service can call dependent service
    # 2. Test data flows correctly
    # 3. Test error handling
    # 4. Test edge cases
    
    print("✅ All integration tests passed")

if __name__ == "__main__":
    test_service_integration()
```

### Workflow Test Template

```python
#!/usr/bin/env python3
"""Integration test for <Workflow> workflow."""

import os
import sys
import requests
import time

# Services involved in workflow
SERVICES = {
    "service1": os.getenv("SERVICE1_URL", "http://localhost:8080"),
    "service2": os.getenv("SERVICE2_URL", "http://localhost:8081"),
}

def test_workflow():
    """Test complete workflow."""
    print("Testing workflow...")
    
    # Step 1: Initialize workflow
    print("Step 1: Initializing...")
    # ... initialization code
    
    # Step 2: Process data through services
    print("Step 2: Processing...")
    # ... processing code
    
    # Step 3: Validate results
    print("Step 3: Validating...")
    # ... validation code
    
    # Step 4: Cleanup
    print("Step 4: Cleanup...")
    # ... cleanup code
    
    print("✅ Workflow test passed")

if __name__ == "__main__":
    test_workflow()
```

## Best Practices

### Test Independence
- ✅ Each test should be runnable independently
- ✅ Clean up resources after test completion
- ✅ Don't rely on test execution order
- ✅ Use unique identifiers for test data

### Service Dependencies
- ✅ Always check service health before testing
- ✅ Provide clear skip messages for unavailable services
- ✅ Use appropriate timeouts
- ✅ Include retry logic for transient failures

### Data Management
- ✅ Use test fixtures for consistent data
- ✅ Clean up test data after tests
- ✅ Avoid hard-coded test data
- ✅ Use unique IDs to avoid conflicts

### Error Handling
- ✅ Test both success and failure paths
- ✅ Verify error messages are clear
- ✅ Test timeout scenarios
- ✅ Test partial failures

## Debugging

### Enable Verbose Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Service Logs
```bash
# View service logs
docker-compose -f infrastructure/docker/brev/docker-compose.yml logs <service>

# Follow logs in real-time
docker-compose -f infrastructure/docker/brev/docker-compose.yml logs -f <service>
```

### Test Individual Components
```bash
# Test service health endpoints
curl http://localhost:8080/health

# Test specific endpoints
curl -X POST http://localhost:8080/endpoint \
  -H "Content-Type: application/json" \
  -d '{"key": "value"}'
```

## Common Issues

### Service Not Available
**Symptom**: Connection refused or timeout errors

**Solutions**:
1. Check service is running: `docker ps`
2. Check service health: `curl http://localhost:8080/health`
3. Check service logs: `docker logs <container>`
4. Verify port mappings
5. Check network connectivity

### Test Data Conflicts
**Symptom**: Tests fail when run together but pass individually

**Solutions**:
1. Use unique test data identifiers
2. Clean up data between tests
3. Check for data isolation
4. Use test transactions (rollback after test)

### Intermittent Failures
**Symptom**: Tests sometimes pass, sometimes fail

**Solutions**:
1. Add retry logic for transient failures
2. Increase timeouts
3. Check for race conditions
4. Verify service startup time
5. Check resource availability

## Coverage Goals

Integration tests should cover:
- ✅ Happy path scenarios
- ✅ Error conditions
- ✅ Timeout scenarios
- ✅ Partial failures
- ✅ Data validation
- ✅ Authentication/authorization
- ✅ Rate limiting
- ✅ Concurrent operations

## Contributing

When adding new integration tests:

1. **Choose the right category**:
   - Service integration → `services/`
   - Multi-step workflow → `workflows/`

2. **Follow naming conventions**:
   - Python: `test_<feature>_integration.py`
   - Go: `<feature>_integration_test.go`

3. **Include documentation**:
   - Test purpose and scope
   - Required services
   - Expected behavior
   - Known limitations

4. **Add service checks**:
   - Verify services are available
   - Provide clear skip messages
   - Include health checks

5. **Update this README**:
   - Add test to appropriate section
   - Document any special setup
   - Note any dependencies
