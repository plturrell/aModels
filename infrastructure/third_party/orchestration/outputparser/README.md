# Output Parsers Package

This package provides components for parsing the raw string output of a Large Language Model (LLM) into a more structured format.

## Purpose

LLMs produce text, but applications often need to work with structured data. For example, when an `Agent` decides to use a tool, the LLM might output a block of text containing the tool's name and its input. An output parser is responsible for extracting this information from the text and converting it into a structured object (like an `AgentAction`).

Output parsers bridge the gap between the unstructured world of language and the structured world of code.

## Core Interface

The primary interface in this package is `OutputParser`. It defines the methods for parsing the LLM's output.

```go
// The OutputParser interface defines the contract for all output parsers.
type OutputParser interface {
    // Parse parses the raw text from the LLM into a structured object.
    Parse(text string) (any, error)

    // GetFormatInstructions returns a string describing the format that the LLM should use for its output.
    GetFormatInstructions() string
}
```

-   `Parse`: Takes the raw text from the LLM and returns a structured Go object.
-   `GetFormatInstructions`: This is a critical method. It returns a string that is included in the prompt sent to the LLM, instructing it on how to format its response so that the parser can understand it.

## Common Output Parsers

This package can contain several types of parsers:

-   **`AgentOutputParser`**: A parser specifically designed to extract `AgentAction` or `AgentFinish` objects from an LLM's output. This is the most common parser used in agents.
-   **`JSONOutputParser`**: A parser that instructs the LLM to respond with a JSON object and then parses that JSON into a Go struct.
-   **`CommaSeparatedListOutputParser`**: A simple parser that can extract a list of items from a comma-separated string.

## How to Use Output Parsers

Output parsers are typically used within `Chains` and `Agents`. The `GetFormatInstructions` method is used to augment the prompt, and the `Parse` method is called on the LLM's final output.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/outputparser"
)

// 1. Create a parser
parser := outputparser.NewAgentOutputParser()

// 2. Get the format instructions to include in the prompt
formatInstructions := parser.GetFormatInstructions()
// This might be something like: "...Use a tool by responding with a JSON blob of the following format: {\"action\": \"tool_name\", \"action_input\": \"input\"}"

// 3. After getting the LLM's response...
llmOutput := "... {\"action\": \"calculator\", \"action_input\": \"2+2\"} ..."
parsedResult, err := parser.Parse(llmOutput)

// parsedResult would now be an AgentAction struct.
```
