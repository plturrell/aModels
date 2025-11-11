# Web Interface

This directory contains the static assets for the VaultGemma LocalAI web interface. The UI is a simple, single-page application built with HTML, CSS, and JavaScript that provides a user-friendly way to interact with the server.

## Features

The web UI provides the following features:

- **Domain Browser**: Displays a real-time list of all agent domains loaded by the server, along with their metadata.
- **Chat Interface**: A familiar chat window for sending prompts and viewing the model's responses in a conversational format.
- **Domain Selection**: Users can either let the server auto-detect the best domain for a prompt or manually select a specific domain from a dropdown list for targeted testing.
- **Live Statistics**: The UI displays real-time statistics about the server, such as the number of loaded domains, total messages processed, and average response time.
- **Server Status**: A status indicator shows whether the UI is successfully connected to the backend server.

## How It Works

The server is configured to serve the static files from this directory when a user navigates to the root URL (`/`). The `index.html` file is the main entry point, and the accompanying JavaScript file (`app.js`, assumed) makes API calls to the server's endpoints (e.g., `/v1/domains`, `/v1/chat/completions`) to populate the UI and handle user interactions.

## How to Use

1.  Run the VaultGemma LocalAI server.
2.  Open your web browser and navigate to `http://localhost:8080` (or the configured port).
3.  The web interface will load automatically.
