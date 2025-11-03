# Benchmark Data Factory

This tool is a data processing pipeline for converting raw data from various sources into a standardized JSON format that can be used by the `aibench` tool.

## Purpose

Benchmark datasets come in many different formats (CSV, JSONL, etc.). This tool provides a consistent way to process these disparate formats into a single, unified structure that the rest of the framework can easily consume. It follows a standard ETL (Extract, Transform, Load) process.

## How It Works

1.  **Extract:** A **Connector** reads raw data from an input source (e.g., a CSV file).
2.  **Transform:** A **Mapper** takes the raw data from the connector and transforms it into a standardized `BenchmarkTask` struct. There is a specific mapper for each benchmark type (e.g., `BoolQMapper`, `ARCMapper`).
3.  **Load:** A **Formatter** writes the list of `BenchmarkTask` structs to a structured output file (e.g., a JSON file).

This modular design allows for easy extension to support new data sources (by adding new Connectors) or new benchmark formats (by adding new Mappers).

## Usage

```bash
go run ./cmd/factory/main.go [flags]
```

### Flags

*   `--in=<path>`: (Required) The path to the input data file.
*   `--out=<path>`: (Required) The path for the generated output JSON file.
*   `--connector=<type>`: The type of connector to use based on the input file format (e.g., `csv`).
*   `--mapper=<type>`: The type of mapper to use to transform the data (e.g., `boolq`, `arc`).

### Example

To convert a CSV file containing BoolQ data into the standard JSON format:

```bash
go run ./cmd/factory/main.go \
    --in=./raw_data/boolq.csv \
    --out=./data/boolq/validation.json \
    --connector=csv \
    --mapper=boolq
```
