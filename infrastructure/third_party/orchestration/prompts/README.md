# Prompts Package

This package provides tools for creating, managing, and formatting dynamic prompts for Large Language Models (LLMs).

## Purpose

A well-crafted prompt is critical for getting good results from an LLM. This package provides a structured way to build prompts that are flexible and reusable. Instead of hardcoding prompts, you can create templates that can be dynamically populated with user input, examples, and other information.

## Core Components

### `PromptTemplate`

The `PromptTemplate` is the primary component of this package. It is created from a template string that contains placeholders for variables.

**Features:**

-   **Dynamic Input**: Define variables in your template string (e.g., `{{.question}}`) that can be filled in at runtime.
-   **Input Validation**: The template automatically checks that all required variables are provided when formatting the prompt.

**Example:**

```go
import "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"

// Create a new prompt template
template := prompts.NewPromptTemplate(
    "Answer the following question: {{.question}}",
    []string{"question"},
)

// Format the prompt with a specific question
prompt, err := template.Format(map[string]any{
    "question": "What is the capital of France?",
})
// Result: "Answer the following question: What is the capital of France?"
```

### Few-Shot Prompting

This package also likely includes helpers for "few-shot" prompting, a technique where you provide the LLM with several examples of the desired input/output format to improve its performance on a new task. This is often done by creating a template for the examples and another for the final prompt.

## How It's Used

Prompt templates are a fundamental building block of `Chains`. A typical `LLMChain` will take a `PromptTemplate` and an `LLM` as its two main components. When the chain is run, it first uses the prompt template to format the user input and then sends the resulting prompt to the LLM.
