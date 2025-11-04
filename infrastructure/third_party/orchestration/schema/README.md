# Schema Package

This package defines the core data structures and interfaces that are used throughout the orchestration framework.

## Purpose

The `schema` package provides a consistent set of data types that are passed between the different components of the system (LLMs, Chains, Agents, etc.). By defining these structures in a central location, we ensure that all components can communicate with each other effectively and predictably.

This package acts as the common language for the entire framework.

## Key Data Structures

This package likely defines several key data structures:

-   **`Generation`**: Represents the output of a single generation from an LLM. It typically includes the generated text and may also include additional information, such as the finish reason or token usage.

-   **`LLMResult`**: Represents the full output of an LLM call, which may include multiple `Generation` objects (if multiple responses were requested) and the overall token usage for the entire call.

-   **`AgentAction`**: Represents the action that an agent has decided to take. It includes the name of the tool to be called and the input for that tool.

-   **`AgentFinish`**: Represents the final response from an agent when it has completed its work. It contains the final output to be returned to the user.

-   **`Document`**: Represents a piece of text that has been loaded into the system. It contains the text content and associated metadata.

## How It's Used

These data structures are used everywhere in the framework. For example:

-   An `LLM`'s `Generate` method returns an `LLMResult`.
-   An `Agent`'s reasoning loop produces either an `AgentAction` (to call a tool) or an `AgentFinish` (to end the process).
-   A `DocumentLoader` returns a slice of `Document` objects.

By understanding the structures in this package, you can better understand the flow of data through the entire orchestration framework.
