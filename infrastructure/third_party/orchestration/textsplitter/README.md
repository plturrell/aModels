# Text Splitters Package

This package provides utilities for splitting large pieces of text into smaller, more manageable chunks.

## Purpose

Large Language Models have a limited context window, meaning they can only process a certain amount of text at a time. To work with large documents, you must first split them into smaller pieces. This package provides several algorithms for doing so in an intelligent way.

Text splitters are a crucial component in any application that involves retrieving information from large documents (e.g., Retrieval-Augmented Generation, or RAG).

## Core Interface

The primary interface in this package is `TextSplitter`. It defines the methods for splitting text and documents.

```go
// The TextSplitter interface defines the contract for all text splitters.
type TextSplitter interface {
    // SplitText splits a single string into a slice of strings.
    SplitText(text string) ([]string, error)

    // SplitDocuments splits a slice of documents into smaller documents.
    SplitDocuments(documents []schema.Document) ([]schema.Document, error)
}
```

## Common Text Splitters

This package likely provides several different splitting strategies:

-   **`CharacterTextSplitter`**: The simplest splitter. It splits text based on a specific character (e.g., a newline) and measures chunk size by the number of characters.

-   **`RecursiveCharacterTextSplitter`**: A more advanced splitter that tries to split text based on a prioritized list of separators (e.g., `"\n\n"`, `"\n"`, `" "`). It attempts to keep related pieces of text together as much as possible, making it the recommended splitter for general use.

-   **`TokenTextSplitter`**: A splitter that divides text based on the number of tokens (as defined by an LLM's tokenizer) rather than the number of characters. This is the most precise way to ensure that chunks do not exceed an LLM's context limit.

## How to Use Text Splitters

Text splitters are typically used after loading documents with a `DocumentLoader`. The resulting smaller documents can then be processed, for example, by generating embeddings for them and storing them in a vector database.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/documentloaders"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/textsplitter"
)

// 1. Load a document
loader := documentloaders.NewTextLoader("./my_large_document.txt")
docs, _ := loader.Load(context.Background())

// 2. Create a text splitter
// This splitter will try to create chunks of 1000 characters with a 200-character overlap.
splitter := textsplitter.NewRecursiveCharacterTextSplitter(1000, 200)

// 3. Split the documents
chunkedDocs, err := splitter.SplitDocuments(docs)
if err != nil {
    log.Fatal(err)
}

// chunkedDocs is now a slice of smaller document chunks.
```
