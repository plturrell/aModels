package models

import "time"

// OperationStatus enumerates possible lifecycle states of a logged operation.
type OperationStatus int

const (
	OperationStatusUnspecified OperationStatus = iota
	OperationStatusRunning
	OperationStatusSuccess
	OperationStatusError
)

// LangOperation mirrors the protobuf payload while using Go-native structures.
type LangOperation struct {
	ID           string          `json:"id"`
	LibraryType  string          `json:"library_type"`
	Operation    string          `json:"operation"`
	Input        map[string]any  `json:"input"`
	Output       map[string]any  `json:"output"`
	Status       OperationStatus `json:"status"`
	Error        string          `json:"error,omitempty"`
	LatencyMs    int64           `json:"latency_ms"`
	CreatedAt    time.Time       `json:"created_at"`
	CompletedAt  *time.Time      `json:"completed_at,omitempty"`
	SessionID    string          `json:"session_id"`
	UserIDHash   string          `json:"user_id_hash"`
	PrivacyLevel string          `json:"privacy_level"`
}

// CleanupResult represents the outcome of a cleanup operation.
type CleanupResult struct {
	Deleted int64
}
