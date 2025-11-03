# Inference Enhancements Documentation

## Overview

This document describes the inference enhancements implemented in the agenticAiETH Training component, including min-p sampling and RAG-based systematic generalization. These enhancements improve model generation quality and generalization without requiring model retraining.

## Table of Contents

1. [Min-p Sampling](#min-p-sampling)
2. [RAG Integration](#rag-integration)
3. [Unified Pipeline](#unified-pipeline)
4. [Configuration](#configuration)
5. [Usage Examples](#usage-examples)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting](#troubleshooting)

## Min-p Sampling

### Theory

Min-p sampling is an inference-only technique that filters tokens based on a minimum probability threshold relative to the maximum probability. It helps reduce low-probability tokens while maintaining diversity in generation.

**Algorithm:**
1. Calculate maximum probability in the distribution
2. Set threshold = min_p_threshold × max_probability
3. Filter tokens with probability < threshold
4. Sample from remaining tokens

### Implementation

The min-p sampling is implemented in `internal/sampling/minp.go`:

```go
// Create min-p sampler
sampler := sampling.NewMinPSampler(sampling.DefaultMinPConfig())

// Configure threshold
sampler.SetMinPThreshold(0.05)

// Sample from logits
token, logProb, err := sampler.Sample(logits)
```

### Configuration

Min-p sampling can be configured per task type:

```yaml
min_p:
  enabled: true
  default_threshold: 0.05
  task_thresholds:
    mcq: 0.03
    boolq: 0.05
    hellaswag: 0.04
    arc: 0.06
```

### Fallback Modes

When no tokens meet the threshold, the sampler can fall back to:
- **Greedy**: Select highest probability token
- **Nucleus**: Use nucleus sampling with configurable p

## RAG Integration

### Architecture

The RAG (Retrieval-Augmented Generation) system consists of:

1. **Search Client**: Communicates with Search component
2. **Embedding Service**: Handles embeddings with caching
3. **RAG Orchestrator**: Coordinates retrieval and augmentation
4. **HANA Storage**: Persistent storage for contexts and cache

### Components

#### Search Client (`internal/rag/search_client.go`)

```go
// Create search client
client := rag.NewSearchClient("http://localhost:9000", "api-key")

// Perform RAG search
req := &rag.RAGRequest{
    Query: "What is the capital of France?",
    TaskType: "trivia",
    TopK: 5,
    SimilarityThreshold: 0.7,
}

resp, err := client.RAGSearch(ctx, req)
```

#### Embedding Service (`internal/rag/embeddings.go`)

```go
// Create embedding service with caching
embeddingService := rag.NewEmbeddingService(searchClient, time.Hour)

// Get embeddings with caching
embeddings, err := embeddingService.GetEmbeddings(ctx, texts, "model-name")
```

#### RAG Orchestrator (`internal/rag/orchestrator.go`)

```go
// Create RAG orchestrator
orchestrator := rag.NewRAGOrchestrator(searchClient, embeddingService)

// Process RAG request
req := &rag.RAGRequest{
    Query: "Question",
    TaskType: "mcq",
    TopK: 5,
    IncludeExamples: true,
}

resp, err := orchestrator.ProcessRAG(ctx, req)
```

### Multi-hop RAG

For complex queries, the system supports multi-hop retrieval:

```go
// Multi-hop RAG with up to 3 hops
resp, err := orchestrator.MultiHopRAG(ctx, req, 3)
```

## Unified Pipeline

### Architecture

The unified pipeline combines RAG and min-p sampling:

```
Query → RAG Retrieval → Context Augmentation → Min-p Sampling → Response
```

### Implementation

```go
// Create unified pipeline
pipeline := pipeline.NewUnifiedInferencePipeline(
    ragOrchestrator,
    embeddingService,
    enhancedEngine,
    config,
)

// Run inference
req := &pipeline.UnifiedInferenceRequest{
    Query: "What is the capital of France?",
    TaskType: "trivia",
    UseMinP: true,
    MinP: 0.05,
    UseRAG: true,
    RAGTopK: 5,
}

resp, err := pipeline.Infer(ctx, req)
```

### Response Format

```go
type UnifiedInferenceResponse struct {
    Answer         string   `json:"answer"`
    Confidence     float64  `json:"confidence"`
    SamplingMethod string   `json:"sampling_method"` // "baseline", "min-p", "rag", "rag+min-p"
    RAGEnabled     bool     `json:"rag_enabled"`
    MinPEnabled    bool     `json:"minp_enabled"`
    RAGContext     string   `json:"rag_context,omitempty"`
    QueryTime      float64  `json:"query_time_ms"`
    TotalHits      int      `json:"total_hits,omitempty"`
}
```

## Configuration

### Configuration File

The system uses YAML configuration files (`configs/inference_config.yaml`):

```yaml
# Min-p Sampling Configuration
min_p:
  enabled: true
  default_threshold: 0.05
  fallback_mode: "greedy"
  task_thresholds:
    mcq: 0.03
    boolq: 0.05

# RAG Configuration
rag:
  enabled: true
  search_endpoint: "http://localhost:9000"
  default_top_k: 5
  similarity_threshold: 0.7
  max_context_length: 2000

# Component Endpoints
endpoints:
  localai: "http://localhost:8080"
  search: "http://localhost:9000"
  hana: "hana://user:password@localhost:39015/database"
```

### Environment Variables

Configuration can be overridden with environment variables:

```bash
export LOCALAI_ENDPOINT="http://localhost:8080"
export SEARCH_ENDPOINT="http://localhost:9000"
export RAG_API_KEY="your-api-key"
export DEBUG_MODE="true"
```

### Programmatic Configuration

```go
// Load configuration
config, err := config.LoadConfig("configs/inference_config.yaml")

// Override with environment variables
config, err := config.LoadConfigFromEnv("configs/inference_config.yaml")

// Get task-specific settings
profile := config.GetTaskProfile("mcq")
ragConfig := config.GetRAGTaskConfig("boolq")
```

## Usage Examples

### Command Line Interface

The aibench CLI supports inference enhancements:

```bash
# Run with min-p sampling
./aibench run --task=mcq --data=./data/mcq.json --use-minp --minp-threshold=0.05

# Run with RAG
./aibench run --task=trivia --data=./data/trivia.json --enable-rag --rag-top-k=5

# Run with both min-p and RAG
./aibench run --task=arc --data=./data/arc.json --use-minp --enable-rag --config=./configs/custom.yaml

# Run with custom configuration
./aibench run --task=boolq --data=./data/boolq.json --config=./configs/production.yaml
```

### Programmatic Usage

#### Basic Min-p Sampling

```go
package main

import (
    "context"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/sampling"
)

func main() {
    // Create sampler
    sampler := sampling.NewMinPSampler(sampling.DefaultMinPConfig())
    
    // Configure threshold
    sampler.SetMinPThreshold(0.05)
    
    // Sample from logits
    logits := []float64{0.1, 0.3, 0.05, 0.2, 0.35}
    token, logProb, err := sampler.Sample(logits)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Selected token: %d, log probability: %f\n", token, logProb)
}
```

#### RAG Integration

```go
package main

import (
    "context"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/rag"
)

func main() {
    // Create components
    searchClient := rag.NewSearchClient("http://localhost:9000", "api-key")
    embeddingService := rag.NewEmbeddingService(searchClient, time.Hour)
    orchestrator := rag.NewRAGOrchestrator(searchClient, embeddingService)
    
    // Process RAG request
    req := &rag.RAGRequest{
        Query: "What is the capital of France?",
        TaskType: "trivia",
        TopK: 5,
        SimilarityThreshold: 0.7,
        IncludeExamples: true,
    }
    
    resp, err := orchestrator.ProcessRAG(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Augmented query: %s\n", resp.AugmentedQuery)
    fmt.Printf("Context: %s\n", resp.Context)
}
```

#### Unified Pipeline

```go
package main

import (
    "context"
    "os"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/catalog/flightcatalog"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/pipeline"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/rag"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Training/internal/localai"
)

func main() {
    // Create components
    searchClient := rag.NewSearchClient("http://localhost:9000", "api-key")
    embeddingService := rag.NewEmbeddingService(searchClient, time.Hour)
    ragOrchestrator := rag.NewRAGOrchestrator(searchClient, embeddingService)
    ctx := context.Background()
    client := localai.NewClient(os.Getenv("LOCALAI_BASE_URL"), os.Getenv("LOCALAI_API_KEY"))
    var opts []localai.EngineOption
    if addr := os.Getenv("AGENTSDK_FLIGHT_ADDR"); addr != "" {
        if catalog, err := flightcatalog.Fetch(ctx, addr); err == nil {
            opts = append(opts, localai.WithAgentCatalog(&catalog))
        }
    }
    enhancedEngine := localai.NewEnhancedInferenceEngine(client, opts...)
    
    // Create pipeline
    pipeline := pipeline.NewUnifiedInferencePipeline(
        ragOrchestrator,
        embeddingService,
        enhancedEngine,
        nil, // Use default config
    )
    
    // Run inference
    req := &pipeline.UnifiedInferenceRequest{
        Query: "What is the capital of France?",
        TaskType: "trivia",
        UseMinP: true,
        MinP: 0.05,
        UseRAG: true,
        RAGTopK: 5,
    }
    
    resp, err := pipeline.Infer(context.Background(), req)
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Answer: %s\n", resp.Answer)
    fmt.Printf("Sampling method: %s\n", resp.SamplingMethod)
    fmt.Printf("Confidence: %f\n", resp.Confidence)
}
```

## Performance Tuning

### Min-p Sampling Tuning

1. **Threshold Selection**: Start with 0.05 and adjust based on task type
   - MCQ tasks: 0.03-0.05
   - Open-ended tasks: 0.05-0.1
   - Creative tasks: 0.1-0.2

2. **Fallback Mode**: Choose based on use case
   - Greedy: For deterministic results
   - Nucleus: For balanced diversity

### RAG Tuning

1. **Top-k Selection**: Balance between relevance and noise
   - Small datasets: 3-5
   - Large datasets: 5-10
   - Complex queries: 10-20

2. **Similarity Threshold**: Filter irrelevant results
   - High precision: 0.8-0.9
   - Balanced: 0.7-0.8
   - High recall: 0.6-0.7

3. **Context Length**: Balance between information and efficiency
   - Short responses: 1000-2000 tokens
   - Long responses: 2000-4000 tokens

### Caching Optimization

1. **Embedding Cache**: Set appropriate TTL
   - Development: 1 hour
   - Production: 24 hours
   - High-frequency: 1 week

2. **Inference Cache**: Store results for repeated queries
   - Enable for production workloads
   - Set appropriate expiry times

### Monitoring

Monitor key metrics:

- **Accuracy**: Task-specific accuracy improvements
- **Latency**: End-to-end inference time
- **Throughput**: Requests per second
- **Cache Hit Rate**: Embedding and inference cache efficiency
- **RAG Hit Rate**: Percentage of queries with relevant context

## Troubleshooting

### Common Issues

#### Min-p Sampling Issues

**Problem**: No tokens meet threshold
**Solution**: Lower threshold or use fallback mode

**Problem**: Too deterministic output
**Solution**: Increase threshold or use nucleus fallback

#### RAG Issues

**Problem**: No relevant context retrieved
**Solution**: 
- Lower similarity threshold
- Increase top-k
- Check search index quality

**Problem**: Context too long
**Solution**: 
- Reduce max context length
- Improve context filtering
- Use better summarization

#### Performance Issues

**Problem**: High latency
**Solution**:
- Enable caching
- Optimize similarity thresholds
- Use async RAG when possible

**Problem**: Memory usage
**Solution**:
- Reduce cache TTL
- Limit context length
- Use batch processing

### Debug Mode

Enable debug mode for detailed logging:

```yaml
development:
  debug_mode: true
  verbose_logging: true
```

Or via environment variable:

```bash
export DEBUG_MODE="true"
```

### Health Checks

Check component health:

```go
// Check pipeline health
err := pipeline.HealthCheck(ctx)

// Get pipeline stats
stats := pipeline.GetStats()

// Check embedding cache stats
cacheStats := embeddingService.GetCacheStats()
```

### Logging

Configure logging for different components:

```yaml
logging:
  level: "info"
  log_rag_queries: true
  log_minp_sampling: true
  log_performance_metrics: true
  log_cache_hits: true
```

## API Reference

### Min-p Sampling API

```go
// Create sampler
func NewMinPSampler(config *MinPConfig) *MinPSampler

// Sample from logits
func (s *MinPSampler) Sample(logits []float64) (int, float64, error)

// Configure threshold
func (s *MinPSampler) SetMinPThreshold(threshold float64)

// Get current config
func (s *MinPSampler) GetConfig() *MinPConfig
```

### RAG API

```go
// Create search client
func NewSearchClient(baseURL string, apiKey string) *SearchClient

// Perform RAG search
func (sc *SearchClient) RAGSearch(ctx context.Context, req *RAGRequest) (*RAGResponse, error)

// Get embeddings
func (sc *SearchClient) GetEmbeddings(ctx context.Context, req *EmbeddingRequest) (*EmbeddingResponse, error)
```

### Unified Pipeline API

```go
// Create pipeline
func NewUnifiedInferencePipeline(ragOrchestrator *rag.RAGOrchestrator, embeddingService *rag.EmbeddingService, enhancedEngine *localai.EnhancedInferenceEngine, config *PipelineConfig) *UnifiedInferencePipeline

// Run inference
func (uip *UnifiedInferencePipeline) Infer(ctx context.Context, req *UnifiedInferenceRequest) (*UnifiedInferenceResponse, error)

// Run benchmark
func (uip *UnifiedInferencePipeline) RunBenchmark(ctx context.Context, req *UnifiedInferenceRequest, correctAnswer string) (*localai.BenchmarkResult, error)
```

## Contributing

When contributing to the inference enhancements:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Consider performance implications
5. Test with different task types

## License

This implementation is part of the agenticAiETH project and follows the same licensing terms.
