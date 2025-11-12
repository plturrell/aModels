# Benchmark Implementations

This directory contains the specific implementations for each benchmark task supported by the `aibench` tool.

Each subdirectory corresponds to a single benchmark and contains the necessary logic for data loading, model evaluation, and metric calculation.

## Available Benchmarks

As of this writing, the following benchmarks are implemented:

*   **`arc`**: A sophisticated, hybrid solution for the Abstraction and Reasoning Corpus (ARC).
*   **`boolq`**: The Boolean Questions dataset.
*   **`hellaswag`**: The HellaSwag dataset for commonsense reasoning.
*   **`piqa`**: The Physical Interaction: Question Answering (PIQA) dataset.
*   **`socialiq`**: The Social Interaction QA (SocialIQa) dataset.
*   **`triviaqa`**: The TriviaQA dataset.
*   **`deepseek_ocr`**: Document OCR evaluation leveraging the DeepSeek-OCR vision model.

## Adding a New Benchmark

To add a new benchmark to the framework:

1.  Create a new subdirectory in this `benchmarks` directory (e.g., `mynewbenchmark`).
2.  Inside the new directory, create a Go file (e.g., `mynewbenchmark.go`) that implements the `registry.Runner` interface.
3.  In that same file, register your new runner using `registry.Register(runner{})` in an `init()` function.
4.  Import your new benchmark package for its side effects in `cmd/aibench/main.go`:

    ```go
    import (
        // ... other imports
        _ "ai_benchmarks/benchmarks/mynewbenchmark"
    )
    ```

This modular design makes it straightforward to extend the framework with new tasks and models.
