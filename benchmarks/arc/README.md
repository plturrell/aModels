# ARC Benchmark Implementation

This directory contains a sophisticated, hybrid implementation for solving the Abstraction and Reasoning Corpus (ARC) benchmark.

## Approach

Our solution goes beyond simple pattern matching or end-to-end deep learning. It employs a hybrid strategy that combines several advanced AI techniques:

1.  **Program Synthesis:** We define a Domain-Specific Language (DSL) tailored for grid-based transformations. The system attempts to synthesize a program in this DSL that correctly transforms the training inputs to the corresponding outputs. If a valid program is found, it is then applied to the test input.

2.  **Logical Neural Networks (LNNs):** We implement custom Logical Neural Network models directly in Go (likely using the Gorgonia library). These models, including variants for `gemma` and `phi`, are capable of learning logical rules and combining them with sub-symbolic (neural) processing. The LNNs are used to guide the search process and predict optimal parameters for other components.

3.  **Monte Carlo Tree Search (MCTS):** When a direct program synthesis solution isn't found, we use MCTS to search through the space of possible transformation sequences. The search is guided by a reward function that measures how well a sequence performs on the training pairs.

4.  **Modular Reasoning Strategies:** The system incorporates various reasoning modules, including causal, temporal, and hierarchical reasoners, allowing it to adapt its strategy to the specific characteristics of a given ARC task.

## Key Files

*   `arc.go`: The main benchmark file that orchestrates the different components.
*   `dsl.go`: Defines the Domain-Specific Language for grid transformations.
*   `synthesis.go`: Implements the program synthesis engine.
*   `lnn.go`, `lnn_gemma.go`, `lnn_phi.go`: The core implementations of the Logical Neural Network models.
*   `evaluation.go`: Contains the logic for evaluating the accuracy of predictions.
*   `causal.go`, `temporal.go`, `hierarchical.go`: Modules for different reasoning strategies.

## Running the ARC Benchmark

To run the ARC benchmark with the hybrid model, use the `aibench` tool with `model=hybrid`. You can further control the behavior of the solver using `--param` flags.

```bash
# Run with program synthesis and MCTS
go run ./cmd/aibench run \
    --task=arc \
    --data=<path_to_arc_data> \
    --model=hybrid \
    --param=arc_synthesis=1 \
    --param=mcts_rollouts=100

# Run with a specific LNN model configuration
go run ./cmd/aibench run \
    --task=arc \
    --data=<path_to_arc_data> \
    --model=hybrid \
    --params-in=./models/lnn_gemma_params.json
```
