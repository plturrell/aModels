# Embeddings Package

This package provides a standardized interface for interacting with various text embedding models.

## Purpose

Embeddings are numerical representations (vectors) of text that capture its semantic meaning. This package abstracts away the specific details of each embedding provider's API, allowing you to generate embeddings for your documents in a consistent way.

These embeddings are the foundation of any Retrieval-Augmented Generation (RAG) system, as they are used to find the most relevant document chunks for a given user query.

## Core Interface

The primary interface in this package is `Embedder`. It defines the essential methods that an embedding model must implement:

```go
// The Embedder interface defines the contract for all embedding models.
type Embedder interface {
    // EmbedDocuments generates embeddings for a batch of documents.
    EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)

    // EmbedQuery generates an embedding for a single query string.
    EmbedQuery(ctx context.Context, text string) ([]float32, error)
}
```

-   `EmbedDocuments`: Used to generate embeddings for a large number of documents, which will typically be stored in a vector database.
-   `EmbedQuery`: Used to generate an embedding for a single user query, which will then be used to search for similar documents in the vector database. Some providers have different models or methods for queries vs. documents, and this interface accounts for that.

## Supported Models

This package can contain sub-packages for various embedding providers, such as:

-   `embeddings/localai`: A client for a local AI server that provides an embeddings endpoint.
-   `embeddings/openai`: A client for the official OpenAI embedding models.
-   `embeddings/huggingface`: A client for using open-source embedding models from the Hugging Face Hub.

## How to Use Embeddings

Embeddings are typically generated after documents have been loaded and split into chunks. The resulting vectors are then stored in a vector database for efficient retrieval.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/embeddings"
)

// 1. Initialize an embedding model client
embedder, err := embeddings.NewLocalAI("http://localhost:8080")
if err != nil {
    log.Fatal(err)
}

// 2. A list of text chunks from a text splitter
textChunks := []string{
    "The sky is blue.",
    "The grass is green.",
}

// 3. Generate embeddings for the documents
vectors, err := embedder.EmbedDocuments(context.Background(), textChunks)
if err != nil {
    log.Fatal(err)
}

// vectors is now a slice of numerical representations of the text chunks.
// These can be stored in a vector database.
```
