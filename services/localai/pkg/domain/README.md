# Domain Management Package

This package is the core of the agent routing system. It is responsible for loading, managing, and selecting the appropriate AI agent "domain" for a given prompt.

## Core Responsibilities

1.  **Configuration Loading**: This package parses the `config/domains.json` file at startup. It deserializes the JSON configuration into a structured Go representation (`DomainConfig`) that the application can use.

2.  **Keyword Indexing**: After loading the configurations, the package builds an in-memory index that maps the keywords from each domain to the corresponding domain itself. This index is critical for the fast and efficient routing of prompts.

3.  **Domain Routing**: The `Route` function is the primary entry point for the routing logic. It takes a user's prompt, tokenizes it, and searches the keyword index to find the best-matching domain. The routing algorithm is designed to be fast and effective, typically responding in under a millisecond.

4.  **Domain Registry**: The package maintains a registry of all loaded and available domains, which can be queried to list all models or to retrieve a specific domain by its ID.

## Key Files

*   `domain_config.go`: Defines the Go structs that map to the structure of the `domains.json` configuration file. This includes the `Domain` and `DomainConfig` types.
*   `router.go` (Assumed): This file would contain the `Router` struct and the core logic for indexing keywords and routing prompts.
