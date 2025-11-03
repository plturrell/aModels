# Model Calibration Tool

This tool is used to calibrate a model's generation parameters (`temperature` and `top_p`) to find the optimal settings for a specific benchmark task. It systematically tests different parameter combinations and identifies the set that yields the highest accuracy.

## Purpose

Language models have several parameters that control the randomness and creativity of their output. The `temperature` and `top_p` (nucleus sampling) parameters are two of the most important. Finding the best values for these parameters can significantly improve a model's performance on a given task. This tool automates that search process.

## How It Works

The calibrator connects to a running LocalAI server and iterates through a predefined range of `temperature` and `top_p` values. For each combination, it evaluates the model's performance on a sample of the provided test data. Finally, it reports the best-performing parameter set and saves the detailed results to a JSON file.

## Usage

```bash
go run ./cmd/calibrate/main.go [flags]
```

### Flags

*   `--url`: The URL of the LocalAI server (default: `http://localhost:8080`).
*   `--key`: The API key for the LocalAI server.
*   `--model`: (Required) The name of the model to calibrate (e.g., `phi-2`).
*   `--task`: (Required) The benchmark task to calibrate on (e.g., `boolq`, `hellaswag`).
*   `--data`: (Required) The path to the test data in JSONL format.
*   `--output`: The path to save the calibration results (default: `calibration_result.json`).
*   `--samples`: The maximum number of samples from the test data to use for calibration (default: `100`).

### Example

```bash
go run ./cmd/calibrate/main.go \
    --model=phi-2 \
    --task=boolq \
    --data=./data/boolq/validation.jsonl \
    --output=./calibration/phi-2_boolq_calibration.json
```

## Output

The tool will print a summary report to the console and save a detailed JSON file with the following information:

*   The optimal `temperature` and `top_p` values.
*   The accuracy achieved with the optimal parameters.
*   The performance scores for every parameter combination tested.
*   Recommendations for model configuration.
