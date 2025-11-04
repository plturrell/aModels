# Agents Package

This package contains the logic for creating and running autonomous agents. Agents use an LLM as a reasoning engine to determine which actions to take to accomplish a goal.

## Purpose

While `Chains` execute a predetermined sequence of steps, `Agents` are more dynamic. An agent is given a high-level objective and a set of available `Tools`. It then uses the LLM to repeatedly decide which tool to use next, processes the tool's output, and continues this loop until the objective is met.

This allows for the creation of more sophisticated applications that can solve problems that are not known in advance.

## Core Components

### Agent

The `Agent` is the core component. It is responsible for the main reasoning loop, which is often referred to as a "ReAct" (Reasoning and Acting) loop.

**Typical Workflow:**

1.  **Prompt**: The agent is given a prompt that includes the user's objective, the available tools, and its previous actions (the "scratchpad").
2.  **Reasoning**: The agent sends this prompt to the LLM.
3.  **Action**: The LLM's response is parsed to determine the next action to take, which is typically the name of a tool and the input for that tool.
4.  **Observation**: The chosen tool is executed, and its output (the "observation") is recorded.
5.  **Loop**: The process repeats from step 1, with the new observation added to the scratchpad, until the LLM determines that the final answer has been found.

### Executor

The `Executor` is responsible for running the agent. It manages the loop, calls the agent to get the next action, executes the tool, and passes the result back to the agent.

## How to Use Agents

To use an agent, you typically need to:

1.  Initialize an `LLM`.
2.  Define a set of `Tools` that the agent can use.
3.  Create an `Agent` with the LLM and the tools.
4.  Create an `Executor` to run the agent.

### Example

```go
import (
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/agents"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
    "github.com/agenticAiETH/agenticAiETH_layer4_Orchestration/tools"
)

// 1. Initialize the LLM
llm, _ := localai.New("http://localhost:8080")

// 2. Define the tools
toolList := []tools.Tool{
    tools.NewCalculator(), // A simple tool that can do math
    // ... other tools like a web search or database query tool
}

// 3. Create the agent and executor
// The constructor would take the LLM and the tools
agent := agents.NewConversationalAgent(llm, toolList)
executor := agents.NewExecutor(agent)

// 4. Run the executor
result, err := executor.Run(context.Background(), "What is 5 to the power of 4.5?")
```
