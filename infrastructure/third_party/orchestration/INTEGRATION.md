# AgenticAI ETH - Orchestration Integration Guide

## Overview

The `agenticAiETH_layer4_Orchestration` module provides LangChain-style orchestration capabilities for the local LLM workflow, coordinating:
- **Models** (`agenticAiETH_layer4_Models/`) - Model assets and weights
- **LocalAI** (`agenticAiETH_layer4_LocalAI/`) - Inference serving
- **Search** (`agenticAiETH_layer4_Search/`) - Embedding and retrieval
- **Training** (`agenticAiETH_layer4_Training/`) - Benchmarking and calibration

## Module Structure

```
agenticAiETH_layer4_Orchestration/
├── chains/          # Chain abstractions (LLMChain, MapReduce, Sequential, etc.)
├── llms/            # LLM adapters
│   └── localai/     # LocalAI adapter (NEW)
├── prompts/         # Prompt templates
├── memory/          # Conversation memory
├── callbacks/       # Execution callbacks
├── schema/          # Core types (Document, Message, etc.)
└── ...
```

## Quick Start

### 1. Using LocalAI with Orchestration

The LocalAI adapter allows orchestration chains to call your local inference server:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

func main() {
    // Create LocalAI LLM instance
    llm, err := localai.New(
        localai.WithBaseURL("http://localhost:8080"),
        localai.WithModel("0x3579-VectorProcessingAgent"), // or "auto"
        localai.WithTemperature(0.7),
        localai.WithMaxTokens(500),
    )
    if err != nil {
        log.Fatal(err)
    }

    // Create a simple LLM chain
    template := "Summarize the following text:\n\n{{.text}}"
    prompt := prompts.NewPromptTemplate(template, []string{"text"})
    chain := chains.NewLLMChain(llm, prompt)

    // Run the chain
    ctx := context.Background()
    result, err := chains.Run(ctx, chain, map[string]any{
        "text": "AgenticAI ETH is a blockchain-based AI system...",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result)
}
```

### 2. Document QA with Training Benchmarks

Integrate with the training workspace to use calibrated models:

```go
import (
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
    "github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

// Load documents from training workspace
docs := []schema.Document{
    {PageContent: "...", Metadata: map[string]any{"source": "benchmark_data"}},
}

// Create QA chain
llm, _ := localai.New(localai.WithModel("auto"))
qaChain := chains.LoadStuffQA(llm)

// Ask question over documents
result, _ := chains.Run(ctx, qaChain, map[string]any{
    "input_documents": docs,
    "question":        "What is the accuracy on BoolQ?",
})
```

### 3. Map-Reduce for Large Documents

Process long documents using map-reduce strategy:

```go
// Load map-reduce chain
mrChain := chains.LoadMapReduceQA(llm)

// Process large document
result, _ := chains.Run(ctx, mrChain, map[string]any{
    "input_documents": largeDocuments,
    "question":        "What are the key findings?",
})
```

## Integration Patterns

### LocalAI → Orchestration

The orchestration layer calls LocalAI's HTTP API:

```
┌─────────────────────────────┐
│  Orchestration (chains)     │
│  - LLMChain                 │
│  - Sequential chains        │
└──────────┬──────────────────┘
           │
           │ HTTP POST /v1/chat/completions
           ▼
┌─────────────────────────────┐
│  LocalAI Server             │
│  - Domain routing           │
│  - VaultGemma inference     │
└─────────────────────────────┘
```

### Training → Orchestration

Use trained models via LocalAI:

```
┌─────────────────────────────┐
│  Training Workspace         │
│  - Benchmark results        │
│  - Calibrated weights       │
└──────────┬──────────────────┘
           │
           │ Export models
           ▼
┌─────────────────────────────┐
│  Models Repository          │
│  - VaultGemma weights       │
│  - SentencePiece tokenizers │
└──────────┬──────────────────┘
           │
           │ Load at startup
           ▼
┌─────────────────────────────┐
│  LocalAI Server             │
└──────────┬──────────────────┘
           │
           │ Inference calls
           ▼
┌─────────────────────────────┐
│  Orchestration              │
└─────────────────────────────┘
```

### Search → Orchestration

Integrate retrieval with chains:

```go
// Implement a custom retriever
type SearchRetriever struct {
    searchClient *search.SearchModel
}

func (r *SearchRetriever) GetRelevantDocuments(ctx context.Context, query string) ([]schema.Document, error) {
    // Call agenticAiETH_layer4_Search
    embeddings, err := r.searchClient.Embed(ctx, query)
    if err != nil {
        return nil, err
    }
    
    // Retrieve documents using embeddings
    // ... search logic ...
    
    return documents, nil
}

// Use with retrieval QA chain
retriever := &SearchRetriever{searchClient: searchModel}
qaChain := chains.NewRetrievalQA(llm, retriever)
```

## Configuration

### Environment Variables

```bash
# LocalAI server
export LOCALAI_BASE_URL="http://localhost:8080"
export LOCALAI_MODEL="auto"  # or specific domain

# Model paths (for LocalAI startup)
export MODEL_PATH="./agenticAiETH_layer4_Models/vaultgemma-transformers-1b-v1"

# Search inference
export SEARCH_ENDPOINT="http://localhost:9200"
```

### Module Dependencies

Ensure `go.work` includes all layer4 modules:

```go
use (
    ./agenticAiETH_layer4_LocalAI
    ./agenticAiETH_layer4_Orchestration
    ./agenticAiETH_layer4_Search/search-inference
    ./agenticAiETH_layer4_Training
    ./agenticAiETH_layer4_Training/models/sentencepiece
)
```

## Available Chains

### Document Chains
- `LoadStuffQA` - Stuff all documents into context
- `LoadRefineQA` - Iteratively refine answer
- `LoadMapReduceQA` - Map over docs, reduce results
- `LoadMapRerankQA` - Score and rerank answers

### Summarization
- `LoadStuffSummarization` - Summarize in one pass
- `LoadMapReduceSummarization` - Map-reduce summarization

### Conversational
- `LoadCondenseQuestionGenerator` - Rephrase follow-up questions
- `NewConversationalRetrievalQA` - QA with chat history

### Custom
- `NewLLMChain` - Basic prompt + LLM
- `NewSequentialChain` - Chain multiple steps

## Testing

Run orchestration tests with LocalAI:

```bash
cd agenticAiETH_layer4_Orchestration

# Start LocalAI first
cd ../agenticAiETH_layer4_LocalAI
./bin/vaultgemma-server &

# Run tests
cd ../agenticAiETH_layer4_Orchestration
go test ./llms/localai/...
```

## Next Steps

1. **Implement Search Retriever**: Create `agenticAiETH_layer4_Orchestration/retrievers/search/` adapter
2. **Add Training Evaluators**: Wrap benchmark chains as evaluators callable from Training workspace
3. **Build Agent Workflows**: Combine chains with tools for complex agent behaviors
4. **Optimize Dependencies**: Prune unused vector stores and cloud SDKs from `go.mod`

## Reference

- [LangChain Concepts](https://python.langchain.com/docs/concepts/)
- [LocalAI Server](../agenticAiETH_layer4_LocalAI/README.md)
- [Training Benchmarks](../agenticAiETH_layer4_Training/README.md)
- [Search Inference](../agenticAiETH_layer4_Search/search-inference/)

---

**Status**: ✅ Module renamed and integrated  
**Version**: 1.0.0  
**Last Updated**: October 19, 2025
