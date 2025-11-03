# Command-Line Applications

This directory contains the main entry points for the executable applications in this project.

## `vaultgemma-server`

The `vaultgemma-server` subdirectory holds the `main.go` file, which is the primary entry point for starting the VaultGemma LocalAI server.

### Purpose

This application is responsible for:

1.  Initializing the server configuration.
2.  Loading the agent domain configurations from the specified JSON file.
3.  Setting up the HTTP server and API routes.
4.  Starting the server and listening for incoming requests.

### How to Run

To build and run the server from the root of the project:

```bash
# Build the binary
go build -o bin/vaultgemma-server ./cmd/vaultgemma-server

# Run the server
./bin/vaultgemma-server --port=8080 --config=config/domains.json
```
