# GNN Integration Tests

This document describes the comprehensive integration test suite for GNN-Agent flow, bidirectional queries, and domain routing.

## Overview

The integration tests verify end-to-end functionality of:
1. **GNN REST API endpoints** - All GNN service endpoints (embeddings, classify, predict-links, structural-insights, domain models)
2. **DeepAgents GNN Tools** - Python-based tools for agents to query GNN services
3. **Bidirectional Query Router** - Intelligent routing between Knowledge Graph (factual) and GNN (structural) services
4. **Domain-aware Routing** - Domain detection and model routing for specialized GNN models
5. **StateGraph Workflows** - Go-based workflow integration for GNN processing
6. **Go Agent Pipelines** - GNN query methods in PerplexityPipeline and DMSPipeline

## Test Files

### Python Integration Tests

**File:** `testing/test_gnn_agent_integration.py`

Comprehensive Python test suite covering:
- GNN API endpoint tests (embeddings, classify, predict-links, structural-insights, domain models)
- DeepAgents service integration
- Query router functionality (structural vs factual detection)
- Domain registry and routing
- Cache management
- End-to-end agent-GNN workflows

**Usage:**
```bash
# Run all Python integration tests
python3 testing/test_gnn_agent_integration.py

# With custom service URLs
TRAINING_SERVICE_URL=http://localhost:8080 \
DEEPAGENTS_URL=http://localhost:9004 \
GRAPH_SERVICE_URL=http://localhost:8080 \
python3 testing/test_gnn_agent_integration.py
```

**Environment Variables:**
- `TRAINING_SERVICE_URL` - Training service URL (default: http://localhost:8080)
- `DEEPAGENTS_URL` - DeepAgents service URL (default: http://localhost:9004)
- `GRAPH_SERVICE_URL` - Graph service URL (default: http://localhost:8080)
- `EXTRACT_SERVICE_URL` - Extract service URL (default: http://localhost:19080)

### Go Integration Tests

**File:** `services/graph/pkg/workflows/gnn_processor_integration_test.go`

Tests for StateGraph GNN processors:
- `QueryGNNNode` - Direct GNN query processing
- `HybridQueryNode` - Hybrid KG + GNN queries
- `isStructuralQuery` - Query type detection
- `NewGNNProcessorWorkflow` - Complete GNN workflow
- Unified workflow GNN integration

**Usage:**
```bash
# Run Go integration tests
cd services/graph
go test -v -tags=integration ./pkg/workflows/... -run TestGNNProcessorIntegration

# Run all GNN-related integration tests
go test -v -tags=integration ./pkg/workflows/... -run GNN
```

**File:** `services/orchestration/agents/gnn_pipeline_integration_test.go`

Tests for Go agent pipeline GNN methods:
- `PerplexityPipeline` GNN query methods
- `DMSPipeline` GNN query methods
- End-to-end pipeline-GNN integration

**Usage:**
```bash
# Run agent pipeline GNN tests
cd services/orchestration/agents
go test -v -tags=integration -run TestPerplexityPipelineGNNIntegration
go test -v -tags=integration -run TestDMSPipelineGNNIntegration
```

**Environment Variables:**
- `TRAINING_SERVICE_URL` - Training service URL (default: http://training-service:8080)
- `GRAPH_SERVICE_URL` - Graph service URL (default: http://graph-service:8081)
- `EXTRACT_SERVICE_URL` - Extract service URL (default: http://extract-service:19080)
- `AGENTFLOW_SERVICE_URL` - AgentFlow service URL (default: http://agentflow-service:9001)
- `LOCALAI_URL` - LocalAI service URL (default: http://localai:8080)

## Test Coverage

### GNN API Endpoints

All REST API endpoints are tested:
- ✅ `POST /gnn/embeddings` - Graph and node-level embeddings
- ✅ `POST /gnn/classify` - Node classification
- ✅ `POST /gnn/predict-links` - Link prediction
- ✅ `POST /gnn/structural-insights` - Anomaly detection and pattern analysis
- ✅ `GET /gnn/domains/{domain_id}/model` - Domain model retrieval
- ✅ `POST /gnn/domains/{domain_id}/query` - Domain-specific queries
- ✅ `GET /gnn/registry/domains` - Domain registry listing
- ✅ `GET /gnn/cache/stats` - Cache statistics
- ✅ `POST /gnn/cache/invalidate` - Cache invalidation

### DeepAgents Integration

- ✅ GNN tools availability in agent factory
- ✅ Tool invocation and response handling
- ✅ Service health checks

### Query Router

- ✅ Structural query detection (similar, pattern, classify, embedding, predict, anomaly)
- ✅ Factual query detection (find, get, list, show, match, return)
- ✅ Hybrid query tool availability
- ✅ Query type classification accuracy

### Domain Routing

- ✅ Domain detection from graph properties
- ✅ Domain registry API endpoints
- ✅ Domain-specific model routing
- ✅ Auto domain detection

### StateGraph Workflows

- ✅ `QueryGNNNode` with embeddings
- ✅ `QueryGNNNode` with structural insights
- ✅ `HybridQueryNode` for combined queries
- ✅ `NewGNNProcessorWorkflow` complete workflow
- ✅ Unified workflow GNN integration

### Go Agent Pipelines

- ✅ `PerplexityPipeline.QueryGNNEmbeddings`
- ✅ `PerplexityPipeline.QueryGNNStructuralInsights`
- ✅ `PerplexityPipeline.QueryGNNPredictLinks`
- ✅ `PerplexityPipeline.QueryGNNClassifyNodes`
- ✅ `PerplexityPipeline.QueryGNNHybrid`
- ✅ `DMSPipeline.QueryGNNEmbeddings`
- ✅ `DMSPipeline.QueryGNNHybrid`

## Test Execution

### Prerequisites

1. **Services Running:**
   - Training service (GNN API)
   - DeepAgents service (optional, for agent tests)
   - Graph service (optional, for hybrid queries)
   - Extract service (optional, for graph data)

2. **Dependencies:**
   - Python 3.8+ with `httpx` installed
   - Go 1.19+ for Go tests
   - PyTorch and PyTorch Geometric (for GNN functionality)

### Running All Tests

```bash
# Python tests
python3 testing/test_gnn_agent_integration.py

# Go tests
cd services/graph && go test -v -tags=integration ./pkg/workflows/... -run GNN
cd services/orchestration/agents && go test -v -tags=integration -run GNN
```

### Running Specific Test Suites

```bash
# Test only GNN API endpoints
python3 -c "
from testing.test_gnn_agent_integration import GNNIntegrationTestSuite
suite = GNNIntegrationTestSuite()
suite.run_test('GNN Embeddings', 'Test embeddings', suite.test_gnn_embeddings_endpoint)
suite.run_test('GNN Classify', 'Test classify', suite.test_gnn_classify_endpoint)
"

# Test only query router
python3 -c "
from testing.test_gnn_agent_integration import GNNIntegrationTestSuite
suite = GNNIntegrationTestSuite()
suite.run_test('Query Router', 'Test router', suite.test_query_router_structural_detection)
"

# Test only StateGraph processors
cd services/graph
go test -v -tags=integration ./pkg/workflows/... -run TestGNNProcessorIntegration
```

## Test Results Interpretation

### Expected Behaviors

1. **Service Not Available:**
   - Tests will log warnings but not fail if services are not running
   - Tests check service health before running

2. **Model Not Trained:**
   - GNN endpoints may return errors if models aren't trained yet
   - Tests accept this as valid behavior and log warnings

3. **Cache Empty:**
   - Cache endpoints may return empty stats if cache is not initialized
   - This is acceptable behavior

### Success Criteria

- ✅ All API endpoints return 200 status or acceptable errors
- ✅ Response structures match expected formats
- ✅ Query router correctly identifies query types
- ✅ Domain detection works with domain keywords
- ✅ StateGraph nodes process queries successfully
- ✅ Agent pipelines can invoke GNN methods

### Failure Scenarios

Tests will fail if:
- ❌ API endpoints return unexpected status codes (not 200, 404, or 500)
- ❌ Response structures are missing required fields
- ❌ Query router incorrectly classifies queries
- ❌ StateGraph nodes fail to process requests
- ❌ Agent pipelines cannot invoke GNN methods

## Continuous Integration

### CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Run GNN Integration Tests
  run: |
    # Start services (if using Docker Compose)
    docker-compose up -d training-service graph-service
    
    # Run Python tests
    python3 testing/test_gnn_agent_integration.py
    
    # Run Go tests
    cd services/graph && go test -v -tags=integration ./pkg/workflows/... -run GNN
    cd services/orchestration/agents && go test -v -tags=integration -run GNN
```

### Test Isolation

Tests are designed to be:
- **Independent:** Each test can run standalone
- **Idempotent:** Tests can be run multiple times safely
- **Non-destructive:** Tests don't modify production data
- **Service-aware:** Tests gracefully handle missing services

## Troubleshooting

### Common Issues

1. **Connection Refused:**
   - Ensure services are running
   - Check service URLs in environment variables
   - Verify network connectivity

2. **Timeout Errors:**
   - Increase timeout values in test configuration
   - Check service performance
   - Verify GNN models are loaded

3. **Import Errors:**
   - Ensure Python dependencies are installed
   - Check Go module dependencies
   - Verify import paths

4. **Model Not Found:**
   - Train GNN models before running tests
   - Check model registry
   - Verify domain configuration

## Future Enhancements

Potential improvements:
- [ ] Mock services for faster test execution
- [ ] Performance benchmarking tests
- [ ] Load testing for GNN endpoints
- [ ] Test coverage metrics
- [ ] Automated test report generation
- [ ] Integration with test result dashboards

## Related Documentation

- [GNN API Implementation](./GNN_API_IMPLEMENTATION.md)
- [GNN Agent Tools Implementation](./GNN_AGENT_TOOLS_IMPLEMENTATION.md)
- [GNN Query Router Implementation](./GNN_QUERY_ROUTER_IMPLEMENTATION.md)
- [GNN StateGraph Workflow Implementation](./GNN_STATEGRAPH_WORKFLOW_IMPLEMENTATION.md)
- [GNN Domain Routing Implementation](./GNN_DOMAIN_ROUTING_IMPLEMENTATION.md)
- [GNN Cache Implementation](./GNN_CACHE_IMPLEMENTATION.md)

