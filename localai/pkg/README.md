# Core Packages

This directory contains the core, reusable packages that define the primary functionality of the VaultGemma LocalAI server. The logic is separated into distinct packages to ensure a clean and maintainable architecture.

## Sub-Packages

*   **`domain`**: This package is responsible for all logic related to the management of AI agent domains. It handles the loading of domain configurations from the `domains.json` file, the keyword-based routing of prompts, and the selection of the appropriate agent for a given request.

*   **`server`**: This package sets up and manages the HTTP server. It defines the API routes, implements the HTTP handlers (e.g., for `/v1/chat/completions`, `/health`), and orchestrates the overall request/response lifecycle, including middleware for logging, rate limiting, and CORS.
