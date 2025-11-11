# Automated Tests

This directory contains the automated tests for the VaultGemma LocalAI server. The tests are written using Go's built-in testing framework and are designed to ensure the correctness and stability of the application's core components.

## Purpose

The primary purpose of the test suite is to:

- **Verify Core Logic**: Ensure that critical components like the domain router and API handlers are working as expected.
- **Prevent Regressions**: Catch any bugs or unintended side effects that may be introduced during development.
- **Validate Configuration**: Test the loading and parsing of the `domains.json` configuration file.

## How to Run Tests

To run the entire test suite, execute the following command from the root of the project:

```bash
# Run all tests in verbose mode
go test -v ./tests/...
```

To run a specific test, you can use the `-run` flag with a regular expression that matches the test name:

```bash
# Run only the domain detection tests
go test -v ./tests/ -run TestDomainDetection
```

To check test coverage:

```bash
go test -v -cover ./tests/...
```

## Key Test Cases

While the exact test cases may vary, the suite should cover at least the following scenarios:

*   **`TestDomainLoading`**: Verifies that the `domains.json` file is correctly parsed and that all domains are loaded into the registry.
*   **`TestDomainDetection`**: Tests the keyword-based routing logic by sending various prompts and asserting that the correct domain is selected.
*   **`TestChatCompletionsAPI`**: Makes mock requests to the `/v1/chat/completions` endpoint and validates the response format and status codes.
*   **`TestHealthEndpoint`**: Checks that the `/health` endpoint returns a `200 OK` status.
