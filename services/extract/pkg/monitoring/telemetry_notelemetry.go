// +build notelemetry

package monitoring

import (
	"context"
	"crypto/sha256"
	"fmt"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// TelemetryConfig configures a telemetry client.
type TelemetryConfig struct {
	Address          string
	LibraryType      string
	DefaultOperation string
	PrivacyLevel     string
	UserIDHash       string
	DialTimeout      time.Duration
	CallTimeout      time.Duration
}

// TelemetryClient provides telemetry logging capabilities (disabled build).
type TelemetryClient struct{}

// TelemetryRecord represents a telemetry log entry.
type TelemetryRecord struct {
	LibraryType  string
	Operation    string
	Input        map[string]any
	Output       map[string]any
	Error        error
	ErrorMessage string
	StartedAt    time.Time
	CompletedAt  time.Time
	Latency      time.Duration
	SessionID    string
	PrivacyLevel string
	UserIDHash   string
}

// NewTelemetryClient creates a new telemetry client (disabled).
func NewTelemetryClient(ctx context.Context, cfg TelemetryConfig) (*TelemetryClient, error) {
	return nil, fmt.Errorf("telemetry disabled in this build")
}

// Close closes the telemetry client connection.
func (c *TelemetryClient) Close() error {
	return nil
}

// Log records a telemetry event (no-op when disabled).
func (c *TelemetryClient) Log(ctx context.Context, record TelemetryRecord) error {
	return nil
}

// PromptID generates a deterministic ID for a prompt instance (no-op version).
func PromptID(template string, vars map[string]any) string {
	h := sha256.New()
	h.Write([]byte(template))
	for k, v := range vars {
		h.Write([]byte(fmt.Sprintf("%s=%v", k, v)))
	}
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// LogPromptTokens is a no-op when telemetry is disabled.
func (c *TelemetryClient) LogPromptTokens(ctx context.Context, promptID, templateType string, tokens []llms.Token, variableCount int) error {
	return nil
}

