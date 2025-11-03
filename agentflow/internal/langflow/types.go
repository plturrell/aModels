package langflow

import (
	"encoding/json"
	"fmt"
	"time"
)

// RunFlowRequest models the minimal payload required to execute a Langflow flow.
type RunFlowRequest struct {
	InputValue  string           `json:"input_value,omitempty"`
	Inputs      map[string]any   `json:"inputs,omitempty"`
	ChatHistory []map[string]any `json:"chat_history,omitempty"`
	SessionID   string           `json:"session_id,omitempty"`
	Tweaks      map[string]any   `json:"tweaks,omitempty"`
	Stream      bool             `json:"stream,omitempty"`
}

// RunFlowResult wraps the full JSON response returned by the Langflow API.
type RunFlowResult struct {
	Raw map[string]any
}

// MarshalJSON allows RunFlowResult to be marshalled back to JSON if required.
func (r RunFlowResult) MarshalJSON() ([]byte, error) {
	if r.Raw == nil {
		return []byte("null"), nil
	}
	return json.Marshal(r.Raw)
}

// FlowRecord provides a normalised representation of flow metadata returned by Langflow.
type FlowRecord struct {
	ID          string
	Name        string
	Description string
	ProjectID   string
	UpdatedAt   time.Time
	Raw         map[string]any
}

// FlowImportRequest wraps a raw Langflow flow definition for import/synchronisation.
type FlowImportRequest struct {
	Flow       json.RawMessage `json:"flow"`
	Force      bool            `json:"force"`
	ProjectID  string          `json:"project_id,omitempty"`
	FolderPath string          `json:"folder_path,omitempty"`
	RemoteID   string
}

// APIError captures non-success responses from the Langflow service.
type APIError struct {
	StatusCode int
	Message    string
	Body       []byte
}

func (e *APIError) Error() string {
	if e == nil {
		return "<nil>"
	}
	if e.Message != "" {
		return fmt.Sprintf("langflow api error: status=%d message=%s", e.StatusCode, e.Message)
	}
	if len(e.Body) > 0 {
		return fmt.Sprintf("langflow api error: status=%d body=%s", e.StatusCode, string(e.Body))
	}
	return fmt.Sprintf("langflow api error: status=%d", e.StatusCode)
}
