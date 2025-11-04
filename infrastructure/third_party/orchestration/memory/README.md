# Memory Package

This package provides components that give chains and agents a "memory" of their past interactions.

## Purpose

By default, `Chains` and `Agents` are stateless. Each time you call them, they process the request without any knowledge of previous interactions. The `memory` package provides a way to persist and recall conversation history, allowing you to build applications that can hold a coherent conversation over multiple turns.

## Core Interface

The primary interface in this package is `Memory`. It defines the methods for loading and saving the conversational context.

```go
// The Memory interface defines the contract for all memory types.
type Memory interface {
    // GetMemoryKeys returns the keys that this memory object will add to the chain's input.
    GetMemoryKeys() []string

    // LoadMemoryVariables loads the conversation history and returns it as a map.
    LoadMemoryVariables(inputs map[string]any) (map[string]any, error)

    // SaveContext saves the context of the current run to memory.
    SaveContext(inputs map[string]any, outputs map[string]any) error

    // Clear clears the memory.
    Clear() error
}
```

## Common Memory Types

This package likely provides several different memory implementations:

-   **`BufferMemory`**: The simplest type of memory. It stores the entire conversation history as a single string.
-   **`BufferWindowMemory`**: Similar to `BufferMemory`, but it only keeps a sliding window of the most recent interactions to prevent the history from growing too large.
-   **`ConversationSummaryMemory`**: A more advanced type of memory that uses an LLM to create a summary of the conversation as it happens. This keeps the context concise while retaining the key information.

## How to Use Memory

Memory is typically added to a `Chain` or `Agent` when it is constructed. The chain will then automatically use the memory to load the conversation history into its prompt and to save the context of the latest run.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/memory"
)

// 1. Create a memory instance
mem := memory.NewBufferMemory()

// 2. Create a chain with memory
// The chain constructor would be designed to accept a memory object.
chain := chains.NewConversationChain(llm, mem)

// 3. Run the chain multiple times
// The memory will automatically be populated with the history.
chain.Call(context.Background(), map[string]any{"input": "Hi, I'm Bob."})
chain.Call(context.Background(), map[string]any{"input": "What's my name?"})
// The LLM will know that your name is Bob.
```
