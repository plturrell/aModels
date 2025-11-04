# LLMs Package

This package provides a standardized interface for interacting with various Large Language Models (LLMs).

## Purpose

The goal of this package is to abstract away the specific details of each LLM provider's API. By using the interfaces defined here, the rest of the application can interact with any supported LLM in a consistent way, making it easy to swap out models without changing the core application logic.

## Core Interface

The primary interface in this package is `LLM`. It defines the essential methods that an LLM must implement:

```go
// The LLM interface defines the contract for all language models.
type LLM interface {
    // Call generates a response from the model for a given prompt.
    Call(ctx context.Context, prompt string, options ...CallOption) (string, error)

    // Generate generates responses for a batch of prompts.
    Generate(ctx context.Context, prompts []string, options ...CallOption) ([]*Generation, error)
}
```

-   `Call`: A convenience method for getting a single response for a single prompt.
-   `Generate`: A more powerful method for handling a batch of prompts, which also returns additional information like token usage and finish reasons.

## Supported Models

This package contains sub-packages for each supported LLM provider. For example:

-   `llms/localai`: A client for a locally running AI server that is compatible with the OpenAI API format.
-   `llms/openai`: A client for the official OpenAI API.
-   `llms/huggingface`: A client for models hosted on the Hugging Face Hub.

To use a specific LLM, you would import its package and initialize a new client.

### Example

```go
import "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"

// Initialize a new client for a local AI server
llm, err := localai.New("http://localhost:8080")
if err != nil {
    log.Fatal(err)
}

// Use the LLM
response, err := llm.Call(context.Background(), "What is the capital of France?")
```
