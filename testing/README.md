# LocalAI Integration Tests

This directory contains comprehensive tests for all LocalAI interaction points to ensure all services are properly configured and working.

## Test Suites

### 1. Python Integration Test Suite (`test_localai_integration_suite.py`)

Comprehensive Python test suite that tests:
- LocalAI core endpoints (health, models, domains, chat, embeddings)
- DeepAgents → LocalAI integration
- Graph service → LocalAI integration
- Search-inference → LocalAI embeddings
- Extract service connection
- Gateway service connection
- Transformers service for embeddings
- Verification of no external API calls

**Usage:**
```bash
python3 testing/test_localai_integration_suite.py
```

**Environment Variables:**
- `LOCALAI_URL` - LocalAI service URL (default: http://localhost:8081)
- `DEEPAGENTS_URL` - DeepAgents service URL (default: http://localhost:9004)
- `GRAPH_URL` - Graph service URL (default: http://localhost:8080)
- `SEARCH_URL` - Search-inference URL (default: http://localhost:8090)
- `EXTRACT_URL` - Extract service URL (default: http://localhost:8082)
- `GATEWAY_URL` - Gateway service URL (default: http://localhost:8000)
- `TRANSFORMERS_URL` - Transformers service URL (default: http://localhost:9090)

### 2. Go Integration Tests

#### Embedding Models Tests (`test_embedding_models.go`)

Tests embedding models across services:
- Transformers service health
- LocalAI embeddings endpoint
- Search-inference embeddings
- Embedding model consistency (dimension verification)

**Usage:**
```bash
cd testing
go test -v test_embedding_models.go
```

#### Service Integration Tests (`test_service_integrations.go`)

Tests service-to-LocalAI integrations:
- DeepAgents → LocalAI
- Graph service → LocalAI
- Extract service → LocalAI
- Gateway service → LocalAI
- LocalAI connectivity
- LocalAI domains configuration
- Verification of no external API calls

**Usage:**
```bash
cd testing
go test -v test_service_integrations.go
```

#### DeepSeek OCR Tests (`test_deepseek_ocr.go`)

Tests DeepSeek OCR integration:
- DeepSeek OCR domain configuration
- OCR endpoint functionality
- OCR script availability

**Usage:**
```bash
cd testing
go test -v test_deepseek_ocr.go
```

### 3. DeepAgents Specific Test (`scripts/test_deepagents_localai.py`)

Focused test for DeepAgents → LocalAI integration:
- DeepAgents health
- LocalAI direct connection
- DeepAgents invoke endpoint (should call LocalAI)

**Usage:**
```bash
python3 scripts/test_deepagents_localai.py
```

## Running All Tests

Use the comprehensive test runner:

```bash
./testing/run_all_tests.sh
```

This script:
1. Checks if all services are running
2. Runs Python integration tests
3. Runs Go integration tests
4. Runs DeepAgents specific tests
5. Provides a summary of all results

## Test Coverage

### LocalAI Core
- ✅ Health endpoint
- ✅ Models endpoint
- ✅ Domains endpoint
- ✅ Chat completions
- ✅ Embeddings endpoint

### Service Integrations
- ✅ DeepAgents → LocalAI
- ✅ Graph service → LocalAI
- ✅ Search-inference → LocalAI (embeddings)
- ✅ Extract service → LocalAI
- ✅ Gateway → LocalAI

### Embedding Models
- ✅ Transformers service (all-MiniLM-L6-v2)
- ✅ LocalAI embeddings (0x3579-VectorProcessingAgent)
- ✅ Search-inference embeddings
- ✅ Embedding dimension consistency (384 for all-MiniLM-L6-v2)

### DeepSeek OCR
- ✅ OCR domain configuration
- ✅ OCR endpoint functionality
- ✅ OCR script availability

### Security/Configuration
- ✅ No external API calls configured
- ✅ All services use LocalAI only

## Expected Results

All tests should pass when:
1. All services are running (via docker-compose)
2. LocalAI is properly configured with domains.json
3. All services are configured to use LocalAI (no external APIs)
4. Embedding models are available (all-MiniLM-L6-v2)
5. DeepSeek OCR is configured (if needed)

## Troubleshooting

### Tests Skipping Services
If tests are skipping services, ensure:
- Services are running: `docker compose -f infrastructure/docker/brev/docker-compose.yml ps`
- Services are accessible on expected ports
- Health endpoints are available

### Embedding Tests Failing
- Verify `all-MiniLM-L6-v2` is loaded in transformers-service
- Check LocalAI domains.json has `0x3579-VectorProcessingAgent` configured
- Verify search-inference has `LOCALAI_BASE_URL` set

### External API Warnings
- Check environment variables for external API URLs
- Ensure docker-compose.yml has empty values for external API keys
- Verify no services are configured to use external APIs

## Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run LocalAI Integration Tests
  run: |
    docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
    sleep 30  # Wait for services to start
    ./testing/run_all_tests.sh
```

