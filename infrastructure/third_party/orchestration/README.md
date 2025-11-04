# Go Agentic Orchestration Framework

---

## 1. Overview

This project is a powerful, Go-native framework for developing applications powered by Large Language Models (LLMs). It provides a comprehensive set of tools and abstractions that simplify the entire lifecycle of an LLM application, from prompt management and model interaction to chaining components and creating autonomous agents.

Inspired by successful Python frameworks, this library brings the power of agentic and chained LLM workflows to the Go ecosystem, emphasizing performance, type safety, and concurrency.

## 2. Core Concepts

The framework is built around a few key abstractions that can be composed to create sophisticated applications:

- **`llms`**: A standardized interface for interacting with various LLMs, from local models served by `LocalAI` to commercial APIs.

- **`prompts`**: Tools for creating, managing, and formatting dynamic prompts that can be customized with user input and examples.

- **`chains`**: A mechanism for linking multiple components together to execute a sequence of operations. A chain can combine an LLM with a prompt, a parser, and other chains to create a complex workflow.

- **`agents`**: A higher-level concept where an LLM is used as a reasoning engine to decide which actions to take. Agents are given access to a set of `tools` and use the LLM to determine the sequence in which to use them to accomplish a goal.

- **`memory`**: Components that give chains and agents a sense of state, allowing them to remember previous interactions and use that context in their reasoning.

- **`tools`**: Functions that agents can use to interact with the outside world, such as performing a web search, querying a database, or calling an API.

- **`documentloaders` & `textsplitter`**: Utilities for loading documents from various sources (e.g., files, websites) and splitting them into smaller chunks suitable for processing by an LLM.

## 3. Architecture

The library is designed to be highly modular. Each component is defined by a clear Go interface, allowing for easy customization and extension. A typical application involves:

1.  **Loading Data**: Using a `documentloader` to ingest data.
2.  **Formatting Prompts**: Using a `prompt` template to structure the input for the LLM.
3.  **Interacting with an LLM**: Using an `llm` instance to get a response.
4.  **Chaining Operations**: Using a `chain` to combine the prompt, LLM, and an `outputparser`.
5.  **Adding State**: Wrapping the chain with `memory` to maintain conversation history.
6.  **Creating an Agent**: Providing an agent with `tools` and an LLM to act as a reasoning engine.

## 4. Getting Started

To use the framework, you will typically import the packages you need and compose them in your Go application.

**Example: A simple Q&A chain**

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai" // Assuming a localai LLM client
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

func main() {
    // 1. Initialize the LLM
    llm, err := localai.New("http://localhost:8080")
    if err != nil {
        log.Fatal(err)
    }

    // 2. Create a prompt template
    template := prompts.NewPromptTemplate(
        "Answer the following question: {{.question}}",
        []string{"question"},
    )

    // 3. Create a simple chain
    chain := chains.NewLLMChain(llm, template)

    // 4. Run the chain
    ctx := context.Background()
    result, err := chains.Call(ctx, chain, map[string]any{
        "question": "What is the capital of France?",
    })
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result["text"])
}
```

## 5. Package Reference

For detailed information on each component, please see the `README.md` file located in the corresponding subdirectory.
