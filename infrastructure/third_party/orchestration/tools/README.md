# Tools Package

This package provides the interface and implementations for "tools" that an agent can use to interact with the outside world.

## Purpose

Large Language Models are powerful reasoning engines, but their knowledge is limited to what they were trained on. Tools give LLMs access to external systems and information, allowing them to answer questions about recent events, perform calculations, or interact with APIs.

In the context of an `Agent`, the LLM acts as a router, deciding which tool to call to best answer a user's query.

## Core Interface

The primary interface in this package is `Tool`. It defines the contract that all tools must implement:

```go
// The Tool interface defines the contract for all tools.
type Tool interface {
    // Name returns the unique name of the tool.
    Name() string

    // Description returns a description of what the tool does, which the LLM uses to decide when to use it.
    Description() string

    // Call executes the tool with the given input string.
    Call(ctx context.Context, input string) (string, error)
}
```

-   `Name`: A simple, unique name for the tool (e.g., `"calculator"`).
-   `Description`: A clear, natural language description of what the tool does and what its input should be. This is a critical part of the prompt that the agent's LLM uses to make its decisions.
-   `Call`: The function that executes the tool's logic.

## Common Tools

This package can contain a variety of common tools, such as:

-   **`Calculator`**: A tool that can evaluate mathematical expressions.
-   **`WebSearch`**: A tool that can search the web for information.
-   **`Database`**: A tool for executing SQL queries against a database.
-   **`API`**: A generic tool for making requests to a third-party API.

## How to Use Tools

Tools are primarily used by `Agents`. When you create an agent, you provide it with a list of tools that it is allowed to use. The agent's prompt is then automatically constructed to include the names and descriptions of these tools, giving the LLM the context it needs to decide which one to call.

### Example

```go
import "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/tools"

// Define a list of tools for an agent
toolList := []tools.Tool{
    tools.NewCalculator(),
    tools.NewWebSearch(os.Getenv("SEARCH_API_KEY")),
}

// This list would then be passed to the agent's constructor.
```
