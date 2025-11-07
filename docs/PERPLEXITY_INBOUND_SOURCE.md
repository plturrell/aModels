# Perplexity API Inbound Source - Implementation Summary

## Overview

A complete inbound source has been implemented to process documents from Perplexity API through the full pipeline: OCR → Catalog → Training → Local AI → Search.

## Files Created

### 1. Perplexity Connector
**File**: `services/orchestration/agents/connectors/perplexity_connector.go`

Implements the `SourceConnector` interface to:
- Connect to Perplexity API
- Discover schema (documents table structure)
- Extract documents based on search queries
- Fetch images from citations (optional)

### 2. Perplexity Pipeline
**File**: `services/orchestration/agents/perplexity_pipeline.go`

Orchestrates the complete document processing flow:
- Fetches documents from Perplexity API
- Processes images through DeepSeek OCR
- Registers documents in catalog service
- Exports documents for training
- Stores documents in local AI service
- Indexes documents in search service

### 3. HTTP API Handler
**File**: `services/orchestration/api/perplexity_handler.go`

Provides HTTP endpoints:
- `POST /api/perplexity/process` - Process documents through full pipeline
- `POST /api/perplexity/process-with-ingestion` - Process using ingestion agent pattern

### 4. Agent Factory Update
**File**: `services/orchestration/agents/agent_factory.go`

Added support for "perplexity" source type in `CreateDataIngestionAgent`.

### 5. Documentation
**File**: `services/orchestration/agents/PERPLEXITY_INTEGRATION.md`

Complete integration guide with:
- Architecture overview
- Component descriptions
- Usage examples
- Configuration details
- Troubleshooting guide

## Processing Flow

```
1. Perplexity API
   ↓ (fetch documents via search query)
2. PerplexityConnector
   ↓ (extract document data)
3. PerplexityPipeline
   ├─→ DeepSeek OCR (if images present)
   ├─→ Catalog Service (metadata storage)
   ├─→ Training Service (training data)
   ├─→ Local AI Service (embeddings & storage)
   └─→ Search Service (indexing)
```

## Key Features

1. **Flexible Query Support**: Accepts natural language queries to search Perplexity
2. **Image Processing**: Automatically processes images through DeepSeek OCR
3. **Resilient Pipeline**: Continues processing even if individual steps fail
4. **Standard Integration**: Works with existing data ingestion agent framework
5. **HTTP API**: Easy to trigger via REST endpoints

## Configuration

Required environment variables:
- `PERPLEXITY_API_KEY` - Perplexity API key
- `DEEPSEEK_OCR_ENDPOINT` - DeepSeek OCR service URL
- `DEEPSEEK_OCR_API_KEY` - DeepSeek OCR API key (optional)
- `CATALOG_URL` - Catalog service URL
- `TRAINING_URL` - Training service URL
- `LOCALAI_URL` - Local AI service URL
- `SEARCH_URL` - Search service URL

## Usage Example

### Via HTTP API

```bash
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest research on transformer architectures",
    "limit": 5,
    "include_images": true
  }'
```

### Via Go Code

```go
config := agents.PerplexityPipelineConfig{
    PerplexityAPIKey:    os.Getenv("PERPLEXITY_API_KEY"),
    DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
    CatalogURL:          "http://catalog:8080",
    TrainingURL:         "http://training:8080",
    LocalAIURL:          "http://localai:8080",
    SearchURL:           "http://search:8080",
    Logger:              logger,
}

pipeline, _ := agents.NewPerplexityPipeline(config)
pipeline.ProcessDocuments(ctx, map[string]interface{}{
    "query": "AI research",
    "limit": 10,
})
```

### Via Data Ingestion Agent

```go
factory := agents.NewAgentFactory(graphClient, ruleStore, alertManager, logger)
agent, _ := factory.CreateDataIngestionAgent("perplexity", map[string]interface{}{
    "api_key": "your-key",
    "query": "research papers",
})
agent.Ingest(ctx, config)
```

## Integration Points

1. **DeepSeek OCR**: Uses `pkg/vision/deepseek_client.go` for OCR processing
2. **Catalog**: Integrates with catalog service for metadata storage
3. **Training**: Exports documents to training service for model training
4. **Local AI**: Stores documents with embeddings in local AI service
5. **Search**: Indexes documents for semantic search

## Error Handling

- OCR failures don't stop processing (original content is used)
- Individual service failures are logged but don't halt the pipeline
- All errors are logged with context for debugging

## Next Steps

1. Add HTTP route registration in orchestration service
2. Add unit tests for connector and pipeline
3. Add integration tests with mock services
4. Add retry logic for failed API calls
5. Add rate limiting for Perplexity API
6. Add caching for processed documents

## Testing

To test the implementation:

1. Set environment variables
2. Start all required services (catalog, training, localai, search)
3. Trigger processing via HTTP API or Go code
4. Verify documents appear in all target services
5. Check logs for processing status

## Notes

- The connector implements the standard `SourceConnector` interface
- The pipeline is designed to be resilient to individual service failures
- All service URLs are configurable via environment variables
- The implementation follows existing patterns in the codebase

