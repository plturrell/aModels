# Perplexity API Inbound Source Integration

## Overview

This integration provides a complete inbound source from Perplexity API that processes documents through:
1. **DeepSeek OCR** - Extracts text from images
2. **Catalog** - Stores document metadata and structure
3. **Training** - Prepares documents for model training
4. **Local AI** - Stores documents in local AI service
5. **Search** - Indexes documents for searchability

## Architecture

```
Perplexity API
    ↓
PerplexityConnector (fetches documents)
    ↓
PerplexityPipeline
    ├─→ DeepSeek OCR (if images present)
    ├─→ Catalog Service (metadata storage)
    ├─→ Training Service (training data export)
    ├─→ Local AI Service (document storage)
    └─→ Search Service (indexing)
```

## Components

### 1. PerplexityConnector

**File**: `services/orchestration/agents/connectors/perplexity_connector.go`

Connects to Perplexity API and fetches documents based on search queries.

**Features**:
- Connects to Perplexity API using API key authentication
- Searches for documents using natural language queries
- Fetches images from document citations (optional)
- Returns structured document data

**Usage**:
```go
config := map[string]interface{}{
    "api_key": "your-perplexity-api-key",
    "base_url": "https://api.perplexity.ai", // optional
}

connector := connectors.NewPerplexityConnector(config, logger)
err := connector.Connect(ctx, config)

// Discover schema
schema, err := connector.DiscoverSchema(ctx)

// Extract documents
query := map[string]interface{}{
    "query": "latest research on machine learning",
    "model": "sonar", // optional, default: "sonar"
    "limit": 10,      // optional, default: 10
    "include_images": true, // optional, default: false
}
documents, err := connector.ExtractData(ctx, query)
```

### 2. PerplexityPipeline

**File**: `services/orchestration/agents/perplexity_pipeline.go`

Orchestrates the full document processing pipeline.

**Features**:
- Processes documents through OCR (if images present)
- Registers documents in catalog
- Exports documents for training
- Stores documents in local AI
- Indexes documents in search service

**Usage**:
```go
config := agents.PerplexityPipelineConfig{
    PerplexityAPIKey:    "your-perplexity-api-key",
    PerplexityBaseURL:   "https://api.perplexity.ai",
    DeepSeekOCREndpoint: "http://deepseek-ocr:8080/ocr",
    DeepSeekOCRAPIKey:   "your-deepseek-api-key",
    CatalogURL:         "http://catalog:8080",
    TrainingURL:         "http://training:8080",
    LocalAIURL:          "http://localai:8080",
    SearchURL:           "http://search:8080",
    ExtractURL:          "http://extract:8081",
    Logger:              logger,
}

pipeline, err := agents.NewPerplexityPipeline(config)
if err != nil {
    log.Fatal(err)
}

query := map[string]interface{}{
    "query": "latest research on AI",
    "limit": 5,
    "include_images": true,
}

err = pipeline.ProcessDocuments(ctx, query)
```

### 3. HTTP API Handler

**File**: `services/orchestration/api/perplexity_handler.go`

Provides HTTP endpoints for triggering document processing.

**Endpoints**:

#### POST `/api/perplexity/process`

Process documents from Perplexity API through the full pipeline.

**Request**:
```json
{
  "query": "latest research on machine learning",
  "model": "sonar",
  "limit": 10,
  "include_images": true,
  "config": {
    "additional": "config"
  }
}
```

**Response**:
```json
{
  "status": "completed",
  "query": "latest research on machine learning",
  "message": "Documents processed successfully through OCR, catalog, training, local AI, and search"
}
```

#### POST `/api/perplexity/process-with-ingestion`

Process documents using the data ingestion agent pattern (for consistency with other sources).

**Request**:
```json
{
  "query": "latest research on AI",
  "config": {
    "limit": 5
  }
}
```

## Configuration

### Environment Variables

```bash
# Perplexity API
export PERPLEXITY_API_KEY="your-perplexity-api-key"
export PERPLEXITY_BASE_URL="https://api.perplexity.ai"  # optional

# DeepSeek OCR
export DEEPSEEK_OCR_ENDPOINT="http://deepseek-ocr:8080/ocr"
export DEEPSEEK_OCR_API_KEY="your-deepseek-api-key"

# Service URLs
export CATALOG_URL="http://catalog:8080"
export TRAINING_URL="http://training:8080"
export LOCALAI_URL="http://localai:8080"
export SEARCH_URL="http://search:8080"
export EXTRACT_URL="http://extract:8081"
```

## Integration with Data Ingestion Agent

The Perplexity connector can be used with the standard data ingestion agent:

```go
factory := agents.NewAgentFactory(graphClient, ruleStore, alertManager, logger)

agent, err := factory.CreateDataIngestionAgent("perplexity", map[string]interface{}{
    "api_key": "your-api-key",
    "query": "latest research",
    "limit": 10,
})

err = agent.Ingest(ctx, config)
```

## Processing Flow

1. **Fetch Documents**: PerplexityConnector queries Perplexity API for documents
2. **OCR Processing**: If images are present, DeepSeek OCR extracts text
3. **Catalog Registration**: Document metadata is stored in catalog service
4. **Training Export**: Documents are prepared for training pipeline
5. **Local AI Storage**: Documents are stored in local AI service with embeddings
6. **Search Indexing**: Documents are indexed in search service for retrieval

## Error Handling

The pipeline is designed to be resilient:
- If OCR fails, the original content is still processed
- If catalog registration fails, processing continues
- If training export fails, other steps continue
- Each step logs errors but doesn't stop the pipeline

## Example Usage

### Using the HTTP API

```bash
curl -X POST http://localhost:8080/api/perplexity/process \
  -H "Content-Type: application/json" \
  -d '{
    "query": "latest research on transformer architectures",
    "limit": 5,
    "include_images": true
  }'
```

### Using the Go API

```go
package main

import (
    "context"
    "log"
    "os"

    "github.com/plturrell/aModels/services/orchestration/agents"
)

func main() {
    logger := log.New(os.Stdout, "[perplexity] ", log.LstdFlags)

    config := agents.PerplexityPipelineConfig{
        PerplexityAPIKey:    os.Getenv("PERPLEXITY_API_KEY"),
        DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
        CatalogURL:          "http://catalog:8080",
        TrainingURL:         "http://training:8080",
        LocalAIURL:          "http://localai:8080",
        SearchURL:           "http://search:8080",
        Logger:              logger,
    }

    pipeline, err := agents.NewPerplexityPipeline(config)
    if err != nil {
        log.Fatal(err)
    }

    query := map[string]interface{}{
        "query":         "AI research papers",
        "limit":        10,
        "include_images": true,
    }

    ctx := context.Background()
    if err := pipeline.ProcessDocuments(ctx, query); err != nil {
        log.Fatal(err)
    }

    log.Println("Documents processed successfully")
}
```

## Testing

To test the integration:

1. Set up environment variables
2. Ensure all services (catalog, training, localai, search) are running
3. Run the HTTP handler or use the Go API
4. Check logs for processing status
5. Verify documents appear in:
   - Catalog service
   - Training data exports
   - Local AI storage
   - Search index

## Troubleshooting

### Connection Issues

- Verify `PERPLEXITY_API_KEY` is set correctly
- Check network connectivity to Perplexity API
- Verify API key has proper permissions

### OCR Issues

- Ensure `DEEPSEEK_OCR_ENDPOINT` is correct
- Check DeepSeek OCR service is running
- Verify image data is valid base64

### Service Integration Issues

- Check all service URLs are correct
- Verify services are running and accessible
- Check service logs for detailed error messages

## Future Enhancements

- Batch processing for multiple queries
- Retry logic for failed requests
- Rate limiting for API calls
- Caching of processed documents
- Support for streaming responses
- Custom OCR prompts per document type

