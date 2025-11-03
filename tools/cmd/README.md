# Command-Line Tools

This directory contains the main entry points for the various command-line tools and services included in this project.

Each subdirectory is a self-contained application that can be run using `go run`.

## Tools and Services

*   **`aibench`**: The primary tool for running AI benchmarks. This is the main entry point for the framework. See the `aibench/README.md` for detailed usage instructions.

*   **`arcagi_service`**: A web service that provides a RESTful API for accessing benchmark datasets stored in a SAP HANA database. See the `arcagi_service/README.md` for API endpoints and configuration details.

*   **`vaultgemma_test`**: A simple client for sending test prompts to a `vaultgemma` model served by a LocalAI instance. This is useful for quick smoke tests and verification.

*   **`calibrate`**: A tool for model calibration.

*   **`factory`**: A tool related to model or data factories.
