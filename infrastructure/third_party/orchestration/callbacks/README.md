# Callbacks Package

This package provides a callback system that allows you to hook into the lifecycle of various events within the orchestration framework.

## Purpose

The callback system provides a powerful mechanism for monitoring, logging, and streaming. You can implement custom callback handlers to be notified when a chain starts or ends, when an LLM is called, or when an agent takes an action.

This is essential for:

-   **Logging**: Recording the inputs and outputs of each step in a chain.
-   **Monitoring**: Tracking the performance and cost of LLM calls.
-   **Streaming**: Streaming the output of an LLM to a user interface in real-time.
-   **Debugging**: Gaining visibility into the internal workings of a chain or agent.

## Core Interface

The package defines a `CallbackManager` that holds a list of `CallbackHandler` interfaces. The `CallbackHandler` interface defines a set of methods that are called at different points in the execution lifecycle.

```go
// The CallbackHandler interface defines the methods that can be implemented to handle events.
type CallbackHandler interface {
    HandleLLMStart(ctx context.Context, prompts []string) error
    HandleLLMEnd(ctx context.Context, result schema.LLMResult) error
    HandleChainStart(ctx context.Context, inputs map[string]any) error
    HandleChainEnd(ctx context.Context, outputs map[string]any) error
    HandleToolStart(ctx context.Context, input string) error
    HandleToolEnd(ctx context.Context, output string) error
    // ... and other event handlers
}
```

A handler can choose to implement only the methods it cares about.

## How to Use Callbacks

When you create a new `Chain` or `Executor`, you can provide it with a `CallbackManager` configured with your custom handlers. The component will then automatically call the appropriate handler methods during its execution.

### Example: A simple logging handler

```go
import (
    "fmt"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/callbacks"
)

// 1. Create a custom handler
type MyLoggingHandler struct{}

func (h *MyLoggingHandler) HandleChainStart(ctx context.Context, inputs map[string]any) error {
    fmt.Printf("Chain started with inputs: %v\n", inputs)
    return nil
}

func (h *MyLoggingHandler) HandleChainEnd(ctx context.Context, outputs map[string]any) error {
    fmt.Printf("Chain ended with outputs: %v\n", outputs)
    return nil
}

// 2. Create a callback manager with the handler
manager := callbacks.NewManager()
manager.AddHandler(&MyLoggingHandler{})

// 3. Pass the manager to a chain or executor when it is created
// The chain will now print its inputs and outputs when it runs.
chain := chains.NewLLMChain(llm, template, chains.WithCallbackManager(manager))
```
