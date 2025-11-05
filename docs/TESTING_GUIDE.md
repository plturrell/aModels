# Testing Guide - LocalAI Integration

This guide explains how to test all LocalAI interaction points to ensure everything is working properly.

## Quick Start

Run all tests:
```bash
./testing/run_all_tests.sh
```

## Test Files Created

### 1. Python Integration Test Suite
**File**: `testing/test_localai_integration_suite.py`

Comprehensive test suite covering:
- LocalAI core endpoints (health, models, domains, chat, embeddings)
- All service integrations (DeepAgents, Graph, Search, Extract, Gateway)
- Embedding models verification
- Security checks (no external APIs)

**Run:**
```bash
python3 testing/test_localai_integration_suite.py
```

### 2. Go Integration Tests

#### Embedding Models Tests
**File**: `testing/test_embedding_models.go`
- Transformers service tests
- LocalAI embeddings
- Search-inference embeddings
- Dimension consistency checks

#### Service Integration Tests
**File**: `testing/test_service_integrations.go`
- DeepAgents → LocalAI
- Graph service → LocalAI
- Extract service → LocalAI
- Gateway → LocalAI
- Configuration verification

#### DeepSeek OCR Tests
**File**: `testing/test_deepseek_ocr.go`
- OCR domain configuration
- OCR endpoint functionality

**Run:**
```bash
cd testing
go test -v .
```

### 3. DeepAgents Specific Test
**File**: `scripts/test_deepagents_localai.py`
- Focused test for DeepAgents → LocalAI integration

**Run:**
```bash
python3 scripts/test_deepagents_localai.py
```

## Test Coverage

### ✅ LocalAI Core (5 tests)
- Health endpoint
- Models endpoint
- Domains endpoint
- Chat completions
- Embeddings endpoint

### ✅ Service Integrations (5 tests)
- DeepAgents → LocalAI
- Graph service → LocalAI
- Search-inference → LocalAI
- Extract service → LocalAI
- Gateway → LocalAI

### ✅ Embedding Models (4 tests)
- Transformers service
- LocalAI embeddings
- Search-inference embeddings
- Dimension consistency

### ✅ DeepSeek OCR (2 tests)
- OCR domain configuration
- OCR script availability

### ✅ Configuration/Security (2 tests)
- No external API calls
- Environment variable verification

**Total: 18+ comprehensive tests**

## Running Tests

### Option 1: Run All Tests (Recommended)
```bash
./testing/run_all_tests.sh
```

### Option 2: Run Python Tests Only
```bash
python3 testing/test_localai_integration_suite.py
```

### Option 3: Run Go Tests Only
```bash
cd testing
go test -v .
```

### Option 4: Run Specific Service Test
```bash
# DeepAgents
python3 scripts/test_deepagents_localai.py

# Embeddings
cd testing && go test -v test_embedding_models.go

# Service integrations
cd testing && go test -v test_service_integrations.go
```

## Environment Variables

Tests respect these environment variables (defaults shown):

```bash
LOCALAI_URL=http://localhost:8081
DEEPAGENTS_URL=http://localhost:9004
GRAPH_URL=http://localhost:8080
SEARCH_URL=http://localhost:8090
EXTRACT_URL=http://localhost:8082
GATEWAY_URL=http://localhost:8000
TRANSFORMERS_URL=http://localhost:9090
```

For Docker, use service names:
```bash
LOCALAI_URL=http://localai:8081
DEEPAGENTS_URL=http://deepagents-service:9004
# etc.
```

## Prerequisites

Before running tests:

1. **Start all services:**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
   ```

2. **Wait for services to be ready:**
   ```bash
   sleep 30  # Adjust based on your system
   ```

3. **Verify LocalAI is accessible:**
   ```bash
   curl http://localhost:8081/health
   ```

## Expected Results

### All Tests Pass When:
- ✅ All services are running
- ✅ LocalAI is configured with domains.json
- ✅ All services use LocalAI only (no external APIs)
- ✅ Embedding models are available
- ✅ Required domains are configured

### Tests Skip When:
- ⏭️ Service is not running (with warning)
- ⏭️ Optional feature not configured (e.g., DeepSeek OCR)

### Tests Fail When:
- ❌ Service is misconfigured
- ❌ External API references found
- ❌ LocalAI not accessible
- ❌ Required domain missing

## Troubleshooting

### Services Not Running
```bash
# Check status
docker compose -f infrastructure/docker/brev/docker-compose.yml ps

# Start services
docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
```

### LocalAI Not Responding
```bash
# Check LocalAI logs
docker logs localai

# Verify configuration
cat services/localai/config/domains.json | jq '.domains | keys'
```

### Embedding Tests Failing
- Verify `all-MiniLM-L6-v2` is loaded
- Check transformers-service is running
- Verify `TRANSFORMERS_MODEL` environment variable

### External API Warnings
- Check environment variables: `env | grep -i api`
- Review docker-compose.yml for external API keys
- Ensure all external API URLs are empty

## CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: LocalAI Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: |
          docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
          sleep 30
      
      - name: Run tests
        run: |
          pip install httpx
          ./testing/run_all_tests.sh
      
      - name: Checkout test results
        if: failure()
        run: |
          docker compose -f infrastructure/docker/brev/docker-compose.yml logs
```

## Test Reports

Tests output:
- ✅ Passed tests
- ❌ Failed tests
- ⏭️ Skipped tests
- ⚠️ Warnings

Summary includes:
- Total tests run
- Pass/fail/skip counts
- Total duration
- List of failures (if any)

## Next Steps

After running tests:
1. Review any failures
2. Check service logs for errors
3. Verify configuration files
4. Re-run tests after fixes

## Support

For issues:
1. Check `testing/README.md` for detailed documentation
2. Review `testing/INTEGRATION_TEST_PLAN.md` for test plan
3. Check service logs: `docker compose logs <service-name>`

