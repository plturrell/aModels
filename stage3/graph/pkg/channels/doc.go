// Package channels defines streaming primitives, message envelopes, and
// request/response pipes used by
// LangGraph flows. Streams mirror the async iterator semantics from the Python
// runtime, supporting backpressure, graceful shutdown, error propagation,
// fan-out broadcasting, and bidirectional request handling for multi-consumer
// pipelines.
package channels
