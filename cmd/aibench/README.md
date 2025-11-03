# aibench: AI Benchmark Runner

`aibench` is a command-line tool for running AI benchmarks, training models, and evaluating their performance. It is the primary entry point for the AI Benchmarking and Training Framework.

## Commands

### `list`

Lists all available benchmark tasks registered in the framework.

**Usage:**

```bash
go run ./cmd/aibench list
```

### `run`

Runs a specific benchmark task. This command allows you to specify the dataset, model, and various other parameters for the run.

**Usage:**

```bash
go run ./cmd/aibench run [flags]
```

**Flags:**

*   `--task=<id>`: (Required) The ID of the task to run (e.g., `arc`, `boolq`).
*   `--data=<path>`: (Required) The path to the dataset file or directory.
*   `--model=<name>`: The model to use (e.g., `baseline`, `hybrid`). Defaults to `baseline`.
*   `--out=<file>`: An optional output file for the results in JSON format.
*   `--limit=<N>`: An optional limit on the number of examples to use from the dataset.
*   `--seed=<S>`: A random seed for reproducibility.
*   `--fit=<path>`: The path to a training dataset to fit a model.
*   `--model-in=<path>`: The path to load a pre-trained model from a JSON file.
*   `--model-out=<path>`: The path to save a newly trained model to a JSON file.
*   `--param=<key=value>`: Overrides a model parameter. Can be repeated multiple times.
*   `--params-in=<path>`: The path to a JSON file containing model parameters.
*   `--params-out=<path>`: The path to save the final parameters used in the run.

### `tune`

This command is likely used for hyperparameter tuning. (The implementation is in `tune.go`.)

### `nash`

This command is likely used for game theory-based model analysis. (The implementation is in `nash.go`.)

## Example

```bash
go run ./cmd/aibench run \
    --task=arc \
    --data=./data/arc/training \
    --model=hybrid \
    --model-out=./models/arc_model.json \
    --param=arc_synthesis=1 \
    --param=mcts_rollouts=100
```
