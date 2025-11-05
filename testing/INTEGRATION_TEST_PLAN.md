# Integration Test Plan for LocalAI-Only Configuration

## Overview

This test plan ensures all services are properly configured to use only LocalAI and that all interaction points are working correctly.

## Test Categories

### 1. LocalAI Core Tests
- [x] Health endpoint (`/health`)
- [x] Models endpoint (`/v1/models`)
- [x] Domains endpoint (`/v1/domains`)
- [x] Chat completions (`/v1/chat/completions`)
- [x] Embeddings (`/v1/embeddings`)

### 2. Service Integration Tests

#### DeepAgents → LocalAI
- [x] DeepAgents health check
- [x] DeepAgents can connect to LocalAI
- [x] DeepAgents uses LocalAI for LLM calls
- [x] No external API keys configured

#### Graph Service → LocalAI
- [x] Graph service health check
- [x] Graph service uses LOCALAI_URL environment variable
- [x] Graph service can connect to LocalAI

#### Search-inference → LocalAI
- [x] Search-inference health check
- [x] Search-inference uses LOCALAI_BASE_URL
- [x] Embeddings work via LocalAI
- [x] Search service uses local embeddings only

#### Extract Service → LocalAI
- [x] Extract service health check
- [x] Extract service uses LocalAI for extraction
- [x] langextract-api disabled or configured for LocalAI
- [x] DeepSeek OCR available (if configured)

#### Gateway → LocalAI
- [x] Gateway health check
- [x] Gateway uses LOCALAI_URL
- [x] Gateway can proxy to LocalAI

### 3. Embedding Model Tests
- [x] Transformers service running (all-MiniLM-L6-v2)
- [x] LocalAI embeddings endpoint working
- [x] Search-inference embeddings working
- [x] Embedding dimension consistency (384 for all-MiniLM-L6-v2)
- [x] No external embedding APIs

### 4. Configuration Tests
- [x] No external API URLs in environment
- [x] No external API keys configured
- [x] All services use Docker service names (not localhost)
- [x] Elasticsearch uses local inference only
- [x] langextract-api disabled or uses LocalAI

### 5. DeepSeek OCR Tests
- [x] OCR domain configured in domains.json
- [x] OCR script available
- [x] OCR endpoint functional (if configured)

## Test Execution

### Prerequisites
1. All services running via docker-compose
2. LocalAI configured with domains.json
3. Models available in /models directory

### Quick Test
```bash
./testing/run_all_tests.sh
```

### Manual Test Steps

1. **Start Services**
   ```bash
   docker compose -f infrastructure/docker/brev/docker-compose.yml up -d
   ```

2. **Wait for Services**
   ```bash
   sleep 30  # Wait for services to start
   ```

3. **Run Tests**
   ```bash
   # Python tests
   python3 testing/test_localai_integration_suite.py
   
   # Go tests (if Go is available)
   cd testing && go test -v .
   
   # DeepAgents specific
   python3 scripts/test_deepagents_localai.py
   ```

## Expected Test Results

All tests should:
- ✅ Pass when services are running and configured correctly
- ⏭️ Skip (with warning) when services are not running
- ❌ Fail only when services are misconfigured or using external APIs

## Failure Scenarios

### Service Not Running
- Test skips with warning
- Check docker-compose status

### External API Detected
- Test fails with details
- Check environment variables
- Review docker-compose.yml

### Embedding Dimension Mismatch
- Test warns but doesn't fail
- Verify embedding model configuration

### Domain Not Found
- Test warns if required domain missing
- Check domains.json configuration

## Success Criteria

✅ All tests pass when:
1. Services are running
2. LocalAI is accessible
3. All services configured for LocalAI only
4. No external API references
5. Embedding models available
6. Required domains configured

