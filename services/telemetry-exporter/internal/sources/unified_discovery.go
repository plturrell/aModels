package sources

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/internal/exporter"
)

// UnifiedDiscovery coordinates discovery from multiple sources.
type UnifiedDiscovery struct {
	extractClient      *ExtractServiceClient
	agentTelemetryClient *AgentTelemetryClient
	logger             *log.Logger
}

// NewUnifiedDiscovery creates a new unified discovery coordinator.
func NewUnifiedDiscovery(
	extractClient *ExtractServiceClient,
	agentTelemetryClient *AgentTelemetryClient,
	logger *log.Logger,
) *UnifiedDiscovery {
	return &UnifiedDiscovery{
		extractClient:        extractClient,
		agentTelemetryClient: agentTelemetryClient,
		logger:               logger,
	}
}

// SessionInfo represents information about a discovered session.
type SessionInfo struct {
	SessionID string
	Source    string
	Timestamp time.Time
}

// GetSessionTelemetry fetches telemetry for a session from the specified source.
func (d *UnifiedDiscovery) GetSessionTelemetry(ctx context.Context, sessionID, source string) (*exporter.ExtractServiceResponse, error) {
	switch source {
	case "extract":
		if d.extractClient == nil {
			return nil, fmt.Errorf("extract service client not configured")
		}
		return d.extractClient.GetSessionTelemetry(ctx, sessionID)

	case "agent_telemetry":
		if d.agentTelemetryClient == nil {
			return nil, fmt.Errorf("agent telemetry client not configured")
		}
		// Convert agent telemetry events to Extract service format
		eventsResp, err := d.agentTelemetryClient.GetEvents(ctx, sessionID)
		if err != nil {
			return nil, err
		}
		return convertAgentTelemetryToExtractFormat(eventsResp), nil

	default:
		return nil, fmt.Errorf("unknown source: %s", source)
	}
}

// DiscoverSessions attempts to discover sessions from available sources.
// Note: This is a placeholder - actual discovery depends on available APIs.
func (d *UnifiedDiscovery) DiscoverSessions(ctx context.Context, since time.Time) ([]SessionInfo, error) {
	// For now, discovery is manual - sessions must be provided via API
	// In the future, this could:
	// - Query a database of execution logs
	// - Listen to event streams
	// - Poll agent telemetry service for recent sessions
	d.logger.Printf("Session discovery not yet implemented - sessions must be provided manually")
	return []SessionInfo{}, nil
}

// convertAgentTelemetryToExtractFormat converts agent telemetry events to Extract service format.
func convertAgentTelemetryToExtractFormat(eventsResp *EventsResponse) *exporter.ExtractServiceResponse {
	// Calculate metrics from events
	metrics := exporter.ExtractMetricsSummary{}
	var latencySum float64
	var latencyCount int

	for _, event := range eventsResp.Events {
		switch event.Type {
		case "user_prompt":
			metrics.UserPromptCount++
			if prompt, ok := event.Payload["prompt"].(string); ok && prompt != "" {
				metrics.LastUserPrompt = prompt
			}
		case "tool_call_started":
			metrics.ToolCallStartedCount++
		case "tool_call_completed":
			metrics.ToolCallCount++
			if success, ok := event.Payload["success"].(bool); ok {
				if success {
					metrics.ToolSuccessCount++
				} else {
					metrics.ToolErrorCount++
				}
			}
			if latency, ok := event.Payload["latency_ms"].(float64); ok {
				latencySum += latency
				latencyCount++
			}
		case "model_change":
			metrics.ModelChangeCount++
		}
	}

	if metrics.ToolCallCount > 0 {
		metrics.ToolSuccessRate = float64(metrics.ToolSuccessCount) / float64(metrics.ToolCallCount)
	}
	if latencyCount > 0 {
		metrics.AverageToolLatencyMs = latencySum / float64(latencyCount)
	}

	// Convert events to Extract format
	extractEvents := make([]exporter.ExtractMetricEvent, len(eventsResp.Events))
	for i, event := range eventsResp.Events {
		extractEvents[i] = exporter.ExtractMetricEvent{
			Timestamp: event.Timestamp,
			SessionID: event.SessionID,
			Type:      string(event.Type),
			Payload:   event.Payload,
		}
	}

	return &exporter.ExtractServiceResponse{
		SessionID: eventsResp.SessionID,
		Metrics:   metrics,
		Events:    extractEvents,
	}
}

