// Package checkpoint exposes abstractions for persisting and restoring agent
// state during long-running LangGraph executions. Each backend (Postgres,
// SQLite, in-memory, etc.) will implement the interfaces defined here.
package checkpoint
