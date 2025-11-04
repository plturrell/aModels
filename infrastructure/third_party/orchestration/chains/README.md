# Chains Package

This package provides the core concept of "chains," which are used to link together various components to create a single, coherent application.

## Purpose

Chains are the heart of the orchestration framework. They allow you to combine multiple primitives, such as LLMs, prompts, and output parsers, into a structured sequence of operations. This enables the creation of complex workflows that go beyond a single LLM call.

## Core Interface

The primary interface in this package is `Chain`. It defines a single method, `Call`, which executes the chain's logic.

```go
// The Chain interface is the main interface for all chains.
type Chain interface {
    // Call executes the chain with the given input.
    Call(ctx context.Context, inputs map[string]any, options ...ChainCallOption) (map[string]any, error)

    // GetInputKeys returns the expected input keys for the chain.
    GetInputKeys() []string

    // GetOutputKeys returns the output keys that the chain will produce.
    GetOutputKeys() []string
}
```

## Common Chains

This package provides several common chain implementations:

### `LLMChain`

The most fundamental chain is the `LLMChain`. It combines a `PromptTemplate` with an `LLM`.

**Workflow:**

1.  It takes user input.
2.  It formats a prompt using the `PromptTemplate`.
3.  It sends the formatted prompt to the `LLM`.
4.  It returns the LLM's output.

### Sequential Chains

This package also provides mechanisms for running chains in sequence, where the output of one chain becomes the input for the next. This allows you to build sophisticated, multi-step pipelines.

## How to Use Chains

Chains are designed to be composable. You can create a simple `LLMChain` to handle a specific task, and then combine it with other chains to build a more complex application.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

// 1. Initialize the LLM and PromptTemplate
llm, _ := localai.New("http://localhost:8080")
template := prompts.NewPromptTemplate("Tell me a joke about {{.topic}}", []string{"topic"})

// 2. Create the LLMChain
chain := chains.NewLLMChain(llm, template)

// 3. Run the chain
result, err := chains.Call(context.Background(), chain, map[string]any{
    "topic": "software developers",
})
```
