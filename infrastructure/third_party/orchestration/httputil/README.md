# HTTP Utilities Package

This package provides helper functions and clients for making HTTP requests. It is used by other packages within the framework that need to communicate with external APIs.

## Purpose

Many components in the orchestration framework, such as `LLM` clients and `Tools`, need to make HTTP requests to external services. This package centralizes the logic for making these requests, providing a consistent and reusable way to handle common tasks like setting headers, handling request bodies, and parsing responses.

By using a shared HTTP client, the framework can also benefit from features like connection pooling and consistent timeout management.

## Core Components

### `Client`

This package likely provides a `Client` struct that wraps Go's built-in `http.Client`. This custom client might add features such as:

-   **Automatic Retries**: Automatically retry requests that fail with transient errors.
-   **Authentication**: A simple way to add authentication headers (e.g., `Authorization: Bearer <token>`) to all outgoing requests.
-   **JSON Handling**: Convenience methods for sending and receiving JSON data.

## How It's Used

Components that need to make HTTP requests, such as the client for the OpenAI API (`llms/openai`), would use the client from this package instead of creating their own. This promotes code reuse and consistency.

### Example

```go
// Inside an LLM client implementation

import "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/httputil"

// Create a new HTTP client with an API key
httpClient := httputil.NewClient(httputil.WithAuthToken(apiKey))

// Use the client to make a request
reqBody := // ... build the request body
respBody, err := httpClient.Post(ctx, "https://api.openai.com/v1/completions", reqBody)

// ... process the response
```
