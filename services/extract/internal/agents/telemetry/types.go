package telemetry

import (
	"encoding/json"
	"time"
)

// AgentMetricEventKind enumerates known telemetry event types from goose.
type AgentMetricEventKind string

const (
	EventUserPrompt        AgentMetricEventKind = "user_prompt"
	EventToolCallStarted   AgentMetricEventKind = "tool_call_started"
	EventToolCallCompleted AgentMetricEventKind = "tool_call_completed"
	EventModelChange       AgentMetricEventKind = "model_change"
)

// AgentMetricEvent represents a telemetry event from the goose agent metrics API.
type AgentMetricEvent struct {
	Timestamp time.Time            `json:"timestamp"`
	SessionID string               `json:"session_id"`
	Type      AgentMetricEventKind `json:"type"`
	Payload   map[string]any       `json:"payload,omitempty"`
	Raw       json.RawMessage      `json:"-"`
}

// EventsResponse is the outer wrapper returned by the telemetry API.
type EventsResponse struct {
	SessionID string             `json:"session_id"`
	Events    []AgentMetricEvent `json:"events"`
}
