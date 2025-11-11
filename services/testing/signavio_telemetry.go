package testing

import (
	"fmt"
	"os"
	"strings"
)

// getAgentName gets the agent name from environment or system.
func getAgentName() string {
	// Try environment variable first
	if name := os.Getenv("AGENT_NAME"); name != "" {
		return name
	}
	
	// Try SERVICE_NAME
	if name := os.Getenv("SERVICE_NAME"); name != "" {
		return name
	}
	
	// Try hostname
	if hostname, err := os.Hostname(); err == nil && hostname != "" {
		return hostname
	}
	
	// Default fallback
	return "testing-service"
}

// FormatExecutionForSignavio converts TestExecution to Signavio telemetry format.
func FormatExecutionForSignavio(execution *TestExecution) SignavioTelemetryRecord {
	// Calculate latency in milliseconds
	latencyMs := int64(execution.Metrics.TotalDuration.Milliseconds())
	
	// Build outcome summary
	outcomeSummary := buildOutcomeSummary(execution)
	
	// Build notes with key metrics
	notes := buildNotes(execution)
	
	record := SignavioTelemetryRecord{
		AgentRunID:      execution.ID,
		AgentName:       getAgentName(),
		TaskID:          execution.ScenarioID,
		TaskDescription: fmt.Sprintf("Test execution: %s", execution.ScenarioID),
		StartTime:       execution.StartTime.UnixMilli(),
		EndTime:         execution.EndTime.UnixMilli(),
		Status:          execution.Status,
		OutcomeSummary:  &outcomeSummary,
		LatencyMs:       &latencyMs,
		Notes:           &notes,
	}
	
	// Add enhanced fields for richer dashboard data
	record.ToolsUsed = formatToolsUsed(execution)
	record.LLMCalls = formatLLMCalls(execution)
	record.ProcessSteps = formatProcessSteps(execution)
	
	return record
}

// buildOutcomeSummary builds a summary of the execution outcome.
func buildOutcomeSummary(execution *TestExecution) string {
	parts := []string{}
	
	if execution.Status == "completed" {
		parts = append(parts, "Test completed successfully")
	} else if execution.Status == "failed" {
		parts = append(parts, "Test failed")
	}
	
	if execution.Metrics.ToolCallCount > 0 {
		parts = append(parts, fmt.Sprintf("%d tool calls", execution.Metrics.ToolCallCount))
	}
	
	if execution.Metrics.LLMCallCount > 0 {
		parts = append(parts, fmt.Sprintf("%d LLM calls", execution.Metrics.LLMCallCount))
	}
	
	if len(execution.QualityIssues) > 0 {
		parts = append(parts, fmt.Sprintf("%d quality issues", len(execution.QualityIssues)))
	}
	
	if len(parts) == 0 {
		return "Test execution completed"
	}
	
	return strings.Join(parts, "; ")
}

// buildNotes builds detailed notes for the execution.
func buildNotes(execution *TestExecution) string {
	parts := []string{}
	
	// Add data volumes
	if len(execution.Metrics.DataVolumes) > 0 {
		volParts := []string{}
		for table, count := range execution.Metrics.DataVolumes {
			volParts = append(volParts, fmt.Sprintf("%s: %d rows", table, count))
		}
		parts = append(parts, "Data volumes: "+strings.Join(volParts, ", "))
	}
	
	// Add tool success rate
	if execution.Metrics.ToolCallCount > 0 {
		parts = append(parts, fmt.Sprintf("Tool success rate: %.2f%%", execution.Metrics.ToolSuccessRate*100))
	}
	
	// Add LLM metrics
	if execution.Metrics.LLMCallCount > 0 {
		parts = append(parts, fmt.Sprintf("LLM calls: %d, tokens: %d", execution.Metrics.LLMCallCount, execution.Metrics.LLMTokensUsed))
	}
	
	// Add error count
	if len(execution.Metrics.Errors) > 0 {
		parts = append(parts, fmt.Sprintf("Errors: %d", len(execution.Metrics.Errors)))
	}
	
	if len(parts) == 0 {
		return "No additional notes"
	}
	
	return strings.Join(parts, " | ")
}

// formatToolsUsed formats tool usage statistics for Signavio.
func formatToolsUsed(execution *TestExecution) []SignavioToolUsage {
	if len(execution.Metrics.ToolsUsed) == 0 {
		return nil
	}
	
	// Aggregate tools by name
	toolStats := make(map[string]*SignavioToolUsage)
	
	for _, toolCall := range execution.Metrics.ToolsUsed {
		stats, exists := toolStats[toolCall.ToolName]
		if !exists {
			stats = &SignavioToolUsage{
				ToolName: toolCall.ToolName,
			}
			toolStats[toolCall.ToolName] = stats
		}
		
		stats.CallCount++
		stats.TotalLatencyMs += toolCall.Duration.Milliseconds()
		if toolCall.Success {
			stats.SuccessCount++
		}
	}
	
	// Convert map to slice
	result := make([]SignavioToolUsage, 0, len(toolStats))
	for _, stats := range toolStats {
		result = append(result, *stats)
	}
	
	return result
}

// formatLLMCalls formats LLM call statistics for Signavio.
func formatLLMCalls(execution *TestExecution) []SignavioLLMCall {
	if len(execution.Metrics.LLMCalls) == 0 {
		return nil
	}
	
	// Aggregate LLM calls by model and purpose
	llmStats := make(map[string]*SignavioLLMCall)
	
	for _, llmCall := range execution.Metrics.LLMCalls {
		key := fmt.Sprintf("%s:%s", llmCall.Model, llmCall.Purpose)
		stats, exists := llmStats[key]
		if !exists {
			stats = &SignavioLLMCall{
				Model:   llmCall.Model,
				Purpose: llmCall.Purpose,
			}
			llmStats[key] = stats
		}
		
		stats.CallCount++
		stats.TotalTokens += llmCall.TokensUsed
		stats.TotalLatencyMs += llmCall.Latency.Milliseconds()
	}
	
	// Convert map to slice
	result := make([]SignavioLLMCall, 0, len(llmStats))
	for _, stats := range llmStats {
		result = append(result, *stats)
	}
	
	return result
}

// formatProcessSteps formats process events as steps for Signavio.
func formatProcessSteps(execution *TestExecution) []SignavioProcessStep {
	if len(execution.Metrics.ProcessEvents) == 0 {
		return nil
	}
	
	// Filter for step completion events and format
	result := make([]SignavioProcessStep, 0)
	
	for _, event := range execution.Metrics.ProcessEvents {
		// Only include completed steps (skip started events, include completed/failed)
		if event.EventType == "step_completed" || event.EventType == "step_failed" {
			step := SignavioProcessStep{
				StepName:   event.StepName,
				StartTime:  event.Timestamp.UnixMilli(),
				EndTime:    event.Timestamp.UnixMilli(),
				Status:     event.Status,
				DurationMs: event.Duration.Milliseconds(),
			}
			
			// Try to find the corresponding start event for accurate start time
			for _, startEvent := range execution.Metrics.ProcessEvents {
				if startEvent.StepName == event.StepName && startEvent.EventType == "step_started" {
					step.StartTime = startEvent.Timestamp.UnixMilli()
					step.EndTime = event.Timestamp.UnixMilli()
					break
				}
			}
			
			result = append(result, step)
		}
	}
	
	return result
}

// FormatEventsForSignavio converts process events to Signavio event format.
// This is a helper function for more detailed event tracking if needed.
func FormatEventsForSignavio(execution *TestExecution) []SignavioProcessStep {
	return formatProcessSteps(execution)
}

