package exporter

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/testing"
)

// ExtractServiceResponse represents the response from Extract service agent metrics endpoint.
type ExtractServiceResponse struct {
	SessionID string                 `json:"session_id"`
	Metrics   ExtractMetricsSummary  `json:"metrics"`
	Events    []ExtractMetricEvent   `json:"events"`
}

// ExtractMetricsSummary represents summary metrics from Extract service.
type ExtractMetricsSummary struct {
	UserPromptCount      int     `json:"user_prompt_count"`
	ToolCallCount        int     `json:"tool_call_count"`
	ToolCallStartedCount int     `json:"tool_call_started_count"`
	ToolSuccessCount     int     `json:"tool_success_count"`
	ToolErrorCount       int     `json:"tool_error_count"`
	ToolSuccessRate      float64 `json:"tool_success_rate"`
	AverageToolLatencyMs float64 `json:"avg_tool_latency_ms"`
	ModelChangeCount     int     `json:"model_change_count"`
	LastUserPrompt       string  `json:"last_user_prompt,omitempty"`
	// New comprehensive metrics fields
	ServiceName          string  `json:"service_name,omitempty"`
	WorkflowName         string  `json:"workflow_name,omitempty"`
	AgentType            string  `json:"agent_type,omitempty"`
	PromptCount          int     `json:"prompt_count"`
	TotalPromptTokens    int     `json:"total_prompt_tokens"`
	TotalResponseTokens  int     `json:"total_response_tokens"`
	AveragePromptLatencyMs float64 `json:"avg_prompt_latency_ms"`
}

// ExtractMetricEvent represents a single telemetry event from Extract service.
type ExtractMetricEvent struct {
	Timestamp time.Time            `json:"timestamp"`
	SessionID string               `json:"session_id"`
	Type      string               `json:"type"`
	Payload   map[string]any       `json:"payload"`
}

// FormatFromExtractService converts Extract service response to Signavio format.
func FormatFromExtractService(telemetry *ExtractServiceResponse, agentName string) *testing.SignavioTelemetryRecord {
	// Find start and end times from events
	var startTime, endTime time.Time
	if len(telemetry.Events) > 0 {
		startTime = telemetry.Events[0].Timestamp
		endTime = telemetry.Events[len(telemetry.Events)-1].Timestamp
		for _, event := range telemetry.Events {
			if event.Timestamp.Before(startTime) {
				startTime = event.Timestamp
			}
			if event.Timestamp.After(endTime) {
				endTime = event.Timestamp
			}
		}
	} else {
		startTime = time.Now()
		endTime = time.Now()
	}

	latencyMs := int64(endTime.Sub(startTime).Milliseconds())

	// Determine status based on success rate
	status := "completed"
	if telemetry.Metrics.ToolSuccessRate < 0.5 {
		status = "failed"
	} else if telemetry.Metrics.ToolSuccessRate < 1.0 {
		status = "partial"
	}

	// Build outcome summary
	outcomeSummary := fmt.Sprintf("Agent execution completed; %d tool calls; %d model changes; success rate: %.2f%%",
		telemetry.Metrics.ToolCallCount,
		telemetry.Metrics.ModelChangeCount,
		telemetry.Metrics.ToolSuccessRate*100)

	// Build notes
	notes := fmt.Sprintf("Total events: %d | Average tool latency: %.2f ms | Tool calls: %d (success: %d, errors: %d) | Model changes: %d",
		len(telemetry.Events),
		telemetry.Metrics.AverageToolLatencyMs,
		telemetry.Metrics.ToolCallCount,
		telemetry.Metrics.ToolSuccessCount,
		telemetry.Metrics.ToolErrorCount,
		telemetry.Metrics.ModelChangeCount)

	// Use provided agent name or get from environment
	if agentName == "" {
		agentName = getAgentName()
	}

	// Extract agent metrics
	serviceName, workflowName, workflowVersion, agentType, agentState := extractAgentMetrics(telemetry.Events)

	// Extract enhanced fields
	toolsUsed := extractToolsUsed(telemetry.Events)
	llmCalls := extractLLMCalls(telemetry.Events)
	processSteps := extractProcessSteps(telemetry.Events)
	promptMetrics := extractPromptMetrics(telemetry.Events)

	// Enhance outcome summary with new metrics
	if serviceName != "" {
		outcomeSummary = fmt.Sprintf("%s | Service: %s", outcomeSummary, serviceName)
	}
	if workflowName != "" {
		outcomeSummary = fmt.Sprintf("%s | Workflow: %s", outcomeSummary, workflowName)
	}
	if agentType != "" {
		outcomeSummary = fmt.Sprintf("%s | Agent Type: %s", outcomeSummary, agentType)
	}
	if promptMetrics != nil {
		outcomeSummary = fmt.Sprintf("%s | Prompts: %d chars", outcomeSummary, promptMetrics.PromptLength)
	}

	// Enhance notes with detailed metrics
	if len(toolsUsed) > 0 {
		notes = fmt.Sprintf("%s | Tools: %d types", notes, len(toolsUsed))
	}
	if len(llmCalls) > 0 {
		notes = fmt.Sprintf("%s | Models: %d", notes, len(llmCalls))
	}
	if promptMetrics != nil && promptMetrics.PromptLatencyMs > 0 {
		notes = fmt.Sprintf("%s | Prompt latency: %d ms", notes, promptMetrics.PromptLatencyMs)
	}

	record := &testing.SignavioTelemetryRecord{
		AgentRunID:      telemetry.SessionID,
		AgentName:       agentName,
		TaskID:          telemetry.SessionID,
		TaskDescription: fmt.Sprintf("Agent execution session: %s", telemetry.SessionID),
		StartTime:       startTime.UnixMilli(),
		EndTime:         endTime.UnixMilli(),
		Status:          status,
		OutcomeSummary:  &outcomeSummary,
		LatencyMs:       &latencyMs,
		Notes:           &notes,
		ServiceName:     serviceName,
		WorkflowName:    workflowName,
		WorkflowVersion: workflowVersion,
		AgentType:       agentType,
		AgentState:      agentState,
		PromptMetrics:   promptMetrics,
		ToolsUsed:       toolsUsed,
		LLMCalls:        llmCalls,
		ProcessSteps:    processSteps,
	}

	return record
}

// extractToolsUsed extracts tool usage statistics from events.
func extractToolsUsed(events []ExtractMetricEvent) []testing.SignavioToolUsage {
	toolMap := make(map[string]*testing.SignavioToolUsage)

	for _, event := range events {
		if event.Type == "tool_call_started" || event.Type == "tool_call_completed" {
			toolName, _ := event.Payload["tool_name"].(string)
			if toolName == "" {
				continue
			}

			if toolMap[toolName] == nil {
				toolMap[toolName] = &testing.SignavioToolUsage{
					ToolName: toolName,
					Parameters: make(map[string]any),
				}
			}

			toolMap[toolName].CallCount++

			if event.Type == "tool_call_completed" {
				if success, ok := event.Payload["success"].(bool); ok && success {
					toolMap[toolName].SuccessCount++
				} else if !success {
					// Extract error details
					if errorMsg, ok := event.Payload["error"].(string); ok && errorMsg != "" {
						toolMap[toolName].ErrorDetails = errorMsg
					}
					if errorDetails, ok := event.Payload["error_details"].(string); ok && errorDetails != "" {
						toolMap[toolName].ErrorDetails = errorDetails
					}
				}
				if latency, ok := event.Payload["latency_ms"].(float64); ok {
					toolMap[toolName].TotalLatencyMs += int64(latency)
				}
			}

			// Extract tool parameters
			if params, ok := event.Payload["parameters"].(map[string]any); ok {
				for k, v := range params {
					toolMap[toolName].Parameters[k] = v
				}
			}

			// Extract tool category
			if category, ok := event.Payload["category"].(string); ok && category != "" {
				toolMap[toolName].Category = category
			}

			// Extract service name
			if serviceName, ok := event.Payload["service_name"].(string); ok && serviceName != "" {
				toolMap[toolName].ServiceName = serviceName
			}
		}
	}

	result := make([]testing.SignavioToolUsage, 0, len(toolMap))
	for _, tool := range toolMap {
		result = append(result, *tool)
	}

	return result
}

// extractLLMCalls extracts LLM call statistics from events.
func extractLLMCalls(events []ExtractMetricEvent) []testing.SignavioLLMCall {
	llmMap := make(map[string]*testing.SignavioLLMCall)

	for _, event := range events {
		if event.Type == "model_change" || event.Payload["model"] != nil {
			model, _ := event.Payload["model"].(string)
			if model == "" {
				continue
			}

			// Use model + purpose as key for better tracking
			purpose := "general"
			if p, ok := event.Payload["purpose"].(string); ok && p != "" {
				purpose = p
			}
			key := fmt.Sprintf("%s:%s", model, purpose)

			if llmMap[key] == nil {
				llmMap[key] = &testing.SignavioLLMCall{
					Model:   model,
					Purpose: purpose,
				}
			}

			llmMap[key].CallCount++

			// Extract token counts separately
			if inputTokens, ok := event.Payload["input_tokens"].(float64); ok {
				llmMap[key].InputTokens += int(inputTokens)
			}
			if outputTokens, ok := event.Payload["output_tokens"].(float64); ok {
				llmMap[key].OutputTokens += int(outputTokens)
			}
			// Fallback to total tokens if separate counts not available
			if tokens, ok := event.Payload["tokens"].(float64); ok {
				llmMap[key].TotalTokens += int(tokens)
				if llmMap[key].InputTokens == 0 && llmMap[key].OutputTokens == 0 {
					// Estimate: assume 70% input, 30% output if not specified
					llmMap[key].InputTokens = int(float64(tokens) * 0.7)
					llmMap[key].OutputTokens = int(float64(tokens) * 0.3)
				}
			}

			if latency, ok := event.Payload["latency_ms"].(float64); ok {
				llmMap[key].TotalLatencyMs += int64(latency)
			}

			// Extract model parameters
			if temp, ok := event.Payload["temperature"].(float64); ok {
				llmMap[key].Temperature = temp
			}
			if cost, ok := event.Payload["cost"].(float64); ok {
				llmMap[key].Cost += cost
			}
			if contextLen, ok := event.Payload["context_length"].(float64); ok {
				llmMap[key].ContextLength = int(contextLen)
			}

			// Extract service name
			if serviceName, ok := event.Payload["service_name"].(string); ok && serviceName != "" {
				llmMap[key].ServiceName = serviceName
			}
		}
	}

	result := make([]testing.SignavioLLMCall, 0, len(llmMap))
	for _, llm := range llmMap {
		result = append(result, *llm)
	}

	return result
}

// extractPromptMetrics extracts prompt statistics from events.
func extractPromptMetrics(events []ExtractMetricEvent) *testing.PromptMetrics {
	var promptMetrics *testing.PromptMetrics
	var promptTime time.Time
	var responseTime time.Time

	for _, event := range events {
		if event.Type == "user_prompt" {
			prompt, _ := event.Payload["prompt"].(string)
			if prompt != "" {
				promptTime = event.Timestamp
				promptMetrics = &testing.PromptMetrics{
					PromptLength: len(prompt),
					PromptType:   classifyPromptType(prompt),
					PromptCategory: classifyPromptCategory(prompt),
				}

				// Extract token counts if available
				if tokens, ok := event.Payload["tokens"].(float64); ok {
					promptMetrics.InputTokens = int(tokens)
				}
			}
		}

		// Look for response after prompt
		if promptMetrics != nil && !promptTime.IsZero() {
			if event.Timestamp.After(promptTime) {
				// Check if this is a response event
				if response, ok := event.Payload["response"].(string); ok && response != "" {
					promptMetrics.ResponseLength = len(response)
					responseTime = event.Timestamp
					promptMetrics.PromptLatencyMs = responseTime.Sub(promptTime).Milliseconds()

					if outputTokens, ok := event.Payload["output_tokens"].(float64); ok {
						promptMetrics.OutputTokens = int(outputTokens)
					}
				}
			}
		}
	}

	return promptMetrics
}

// classifyPromptType classifies the type of prompt.
func classifyPromptType(prompt string) string {
	prompt = strings.ToLower(prompt)
	if strings.Contains(prompt, "?") || strings.HasPrefix(strings.TrimSpace(prompt), "what") ||
		strings.HasPrefix(strings.TrimSpace(prompt), "how") || strings.HasPrefix(strings.TrimSpace(prompt), "why") {
		return "question"
	}
	if strings.Contains(prompt, "please") || strings.Contains(prompt, "do") || strings.Contains(prompt, "execute") {
		return "instruction"
	}
	if strings.Contains(prompt, "analyze") || strings.Contains(prompt, "extract") || strings.Contains(prompt, "find") {
		return "task"
	}
	return "general"
}

// classifyPromptCategory classifies the category of prompt.
func classifyPromptCategory(prompt string) string {
	prompt = strings.ToLower(prompt)
	if strings.Contains(prompt, "data") || strings.Contains(prompt, "table") || strings.Contains(prompt, "database") {
		return "data_processing"
	}
	if strings.Contains(prompt, "code") || strings.Contains(prompt, "function") || strings.Contains(prompt, "api") {
		return "code_generation"
	}
	if strings.Contains(prompt, "search") || strings.Contains(prompt, "query") || strings.Contains(prompt, "find") {
		return "search"
	}
	return "general"
}

// extractProcessSteps extracts process steps from events.
func extractProcessSteps(events []ExtractMetricEvent) []testing.SignavioProcessStep {
	steps := make([]testing.SignavioProcessStep, 0, len(events))
	stepMap := make(map[string]*testing.SignavioProcessStep) // Track steps by name for dependencies

	for i, event := range events {
		var endTime time.Time
		if i+1 < len(events) {
			endTime = events[i+1].Timestamp
		} else {
			endTime = event.Timestamp.Add(100 * time.Millisecond) // Default duration
		}

		stepName := event.Type
		if name, ok := event.Payload["step_name"].(string); ok && name != "" {
			stepName = name
		}

		status := "completed"
		if event.Type == "tool_call_started" {
			status = "running"
		} else if event.Type == "tool_call_completed" {
			if success, ok := event.Payload["success"].(bool); ok && !success {
				status = "failed"
			}
		}

		step := testing.SignavioProcessStep{
			StepName:    stepName,
			StartTime:   event.Timestamp.UnixMilli(),
			EndTime:     endTime.UnixMilli(),
			Status:      status,
			DurationMs:  endTime.Sub(event.Timestamp).Milliseconds(),
		}

		// Extract workflow information
		if workflowName, ok := event.Payload["workflow_name"].(string); ok && workflowName != "" {
			step.WorkflowName = workflowName
		}
		if workflowVersion, ok := event.Payload["workflow_version"].(string); ok && workflowVersion != "" {
			step.WorkflowVersion = workflowVersion
		}

		// Extract dependencies
		if deps, ok := event.Payload["dependencies"].([]interface{}); ok {
			step.Dependencies = make([]string, 0, len(deps))
			for _, dep := range deps {
				if depStr, ok := dep.(string); ok {
					step.Dependencies = append(step.Dependencies, depStr)
				}
			}
		}

		// Check for parallel execution
		if parallel, ok := event.Payload["parallel"].(bool); ok {
			step.ParallelExecution = parallel
		}

		// Track step for dependency analysis
		stepMap[stepName] = &step
		steps = append(steps, step)
	}

	return steps
}

// extractAgentMetrics extracts agent-related metrics from events.
func extractAgentMetrics(events []ExtractMetricEvent) (serviceName, workflowName, workflowVersion, agentType, agentState string) {
	for _, event := range events {
		// Extract service name
		if serviceName == "" {
			if sn, ok := event.Payload["service_name"].(string); ok && sn != "" {
				serviceName = sn
			}
		}

		// Extract workflow information
		if workflowName == "" {
			if wn, ok := event.Payload["workflow_name"].(string); ok && wn != "" {
				workflowName = wn
			}
		}
		if workflowVersion == "" {
			if wv, ok := event.Payload["workflow_version"].(string); ok && wv != "" {
				workflowVersion = wv
			}
		}

		// Extract agent type
		if agentType == "" {
			if at, ok := event.Payload["agent_type"].(string); ok && at != "" {
				agentType = at
			}
		}

		// Extract agent state (use the latest state)
		if as, ok := event.Payload["agent_state"].(string); ok && as != "" {
			agentState = as
		}
	}

	return serviceName, workflowName, workflowVersion, agentType, agentState
}

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
	return "agent"
}

