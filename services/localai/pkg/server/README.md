# HTTP Server Package

This package is responsible for all aspects of the HTTP server, including its setup, API routing, and the implementation of the HTTP handlers.

## Core Responsibilities

1.  **Server Initialization**: This package provides a `NewServer` function that initializes a new server instance. This includes setting up the router, registering middleware, and injecting dependencies like the domain router.

2.  **API Routing**: It defines all the API endpoints for the application. This includes the OpenAI-compatible endpoints like `/v1/chat/completions` and `/v1/models`, as well as the administrative endpoints like `/health` and `/metrics`.

3.  **HTTP Handlers**: The package contains the implementation for each HTTP handler. The handlers are responsible for parsing incoming requests, calling the appropriate backend logic (e.g., the domain router), and writing the HTTP response.

4.  **Middleware**: The server is wrapped in several layers of middleware to provide production-grade features, including:
    *   **Logging**: To record details of each incoming request.
    *   **CORS**: To allow cross-origin requests from web applications.
    *   **Rate Limiting**: To protect the server from being overwhelmed by too many requests.
    *   **Request Validation**: To ensure that incoming requests are well-formed.

## Key Files

*   `vaultgemma_server.go`: This file likely contains the main `Server` struct and the `NewServer` function, as well as the registration of all API routes and middleware.
*   `handlers.go` (Assumed): This file would contain the specific implementations for the HTTP handlers, such as `handleChatCompletions`, `handleListModels`, and `handleHealthCheck`.
