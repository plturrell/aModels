// +build !notelemetry

package monitoring

import (
	"context"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

// LogPromptTokensSafe logs prompt tokens without requiring telemetry client initialization.
// This allows token logging even when telemetry is disabled or unavailable.
func LogPromptTokensSafe(ctx context.Context, telemetryClient interface{}, promptID, templateType string, tokens []llms.Token, variableCount int) error {
	if telemetryClient == nil {
		return nil // Silently skip if no telemetry client
	}

	// Try to use the telemetry client if it's available
	if tc, ok := telemetryClient.(*TelemetryClient); ok && tc != nil {
		return tc.LogPromptTokens(ctx, promptID, templateType, tokens, variableCount)
	}

	// If telemetry is not available, just return nil (non-fatal)
	return nil
}

