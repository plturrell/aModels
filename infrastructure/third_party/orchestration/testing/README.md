# Testing Package

This package provides utilities and mocks to facilitate the testing of applications built with the orchestration framework.

## Purpose

Testing applications that rely on external services like Large Language Models can be slow, expensive, and non-deterministic. This package provides tools to mock the behavior of the framework's components, allowing you to write fast, reliable, and isolated unit tests.

## Core Components

### Mock LLMs

This package likely contains a mock implementation of the `llms.LLM` interface. This mock LLM can be programmed to return a specific, predetermined response when it is called with a certain prompt. This allows you to test your chains and agents without making a real API call.

### Mock Tools

Similarly, the package may provide mock implementations of the `tools.Tool` interface. This is useful for testing agents, as you can verify that the agent is calling the correct tool with the correct input, without actually executing the tool's logic.

## How to Use

In your tests, you would use the mock components from this package instead of the real ones. You can then make assertions about how these mocks were called.

### Example: Testing a simple chain

```go
import (
    "testing"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/testing"
)

func TestMyChain(t *testing.T) {
    // 1. Create a mock LLM that expects a certain prompt and will return a specific response.
    mockLLM := testing.NewMockLLM(
        "What is the capital of France?", // Expected prompt
        "The capital of France is Paris.", // Response to return
    )

    // 2. Create the chain with the mock LLM
    template := prompts.NewPromptTemplate("What is the capital of {{.country}}?", []string{"country"})
    chain := chains.NewLLMChain(mockLLM, template)

    // 3. Run the chain
    result, err := chains.Call(context.Background(), chain, map[string]any{"country": "France"})
    if err != nil {
        t.Fatalf("Chain call failed: %v", err)
    }

    // 4. Assert that the result is what we expect
    if result["text"] != "The capital of France is Paris." {
        t.Errorf("Unexpected result: got %v", result["text"])
    }

    // 5. You can also add assertions to the mock to verify it was called correctly.
}
```
