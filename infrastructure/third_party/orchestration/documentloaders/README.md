# Document Loaders Package

This package provides utilities for loading data from various sources into a standardized `Document` format.

## Purpose

To build LLM applications that can answer questions about specific data, you first need to load that data into your application. Document loaders are the components responsible for this first step.

They abstract away the details of reading from different file formats and sources, providing a simple, unified interface for data ingestion.

## Core Interface

The primary interface in this package is `Loader`. It defines a single method, `Load`.

```go
// The Loader interface defines the contract for all document loaders.
type Loader interface {
    // Load reads from the source and returns a slice of Document objects.
    Load(ctx context.Context) ([]schema.Document, error)
}
```

Each `Document` object that is returned contains the text content and associated metadata (e.g., the source file path).

## Supported Loaders

This package can contain a variety of loaders for different sources, such as:

-   **`TextLoader`**: For loading data from plain text files.
-   **`PDFLoader`**: For extracting text from PDF files.
-   **`CSVLoader`**: For loading data from CSV files, often treating each row as a separate document.
-   **`WebLoader`**: For fetching and parsing content from a URL.

## How to Use Document Loaders

Document loaders are typically used at the beginning of a data processing pipeline. Once the documents are loaded, they are often passed to a `TextSplitter` to be broken down into smaller chunks.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/documentloaders"
)

// 1. Create a loader for a specific file
loader := documentloaders.NewTextLoader("./my_document.txt")

// 2. Load the documents
docs, err := loader.Load(context.Background())
if err != nil {
    log.Fatal(err)
}

// docs is now a slice of schema.Document objects
```
