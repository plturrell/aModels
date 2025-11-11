package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/catalog/breakdetection"
)

// BreakDetectionProcessorOptions configures the break detection processor
type BreakDetectionProcessorOptions struct {
	BreakDetectionServiceURL string
	BaselineManagerURL      string
	Logger                  *log.Logger
}

// ProcessBreakDetectionNode returns a node that processes break detection quality-check steps
func ProcessBreakDetectionNode(opts BreakDetectionProcessorOptions) func(ctx context.Context, state map[string]any) (map[string]any, error) {
	return func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract step configuration from state
		stepConfig, ok := state["step_config"].(map[string]any)
		if !ok {
			return nil, fmt.Errorf("step_config not found in state")
		}

		// Extract system name
		systemNameStr, ok := stepConfig["system"].(string)
		if !ok {
			return nil, fmt.Errorf("system not found in step_config")
		}

		// Map system name to SystemName enum
		var systemName breakdetection.SystemName
		switch strings.ToLower(systemNameStr) {
		case "sap_fioneer", "sap-fioneer":
			systemName = breakdetection.SystemSAPFioneer
		case "bcrs":
			systemName = breakdetection.SystemBCRS
		case "rco":
			systemName = breakdetection.SystemRCO
		case "axiomsl", "axiom-sl":
			systemName = breakdetection.SystemAxiomSL
		default:
			return nil, fmt.Errorf("unknown system: %s", systemNameStr)
		}

		// Extract checks configuration
		checksConfig, ok := stepConfig["checks"].([]interface{})
		if !ok {
			return nil, fmt.Errorf("checks not found in step_config")
		}

		// Determine detection type from checks
		var detectionType breakdetection.DetectionType
		for _, checkRaw := range checksConfig {
			check, ok := checkRaw.(map[string]interface{})
			if !ok {
				continue
			}
			checkType, ok := check["type"].(string)
			if !ok {
				continue
			}
			
			// Map check type to detection type
			switch checkType {
			case "finance_break":
				detectionType = breakdetection.DetectionTypeFinance
			case "capital_break":
				detectionType = breakdetection.DetectionTypeCapital
			case "liquidity_break":
				detectionType = breakdetection.DetectionTypeLiquidity
			case "regulatory_break":
				detectionType = breakdetection.DetectionTypeRegulatory
			}
			
			if detectionType != "" {
				break
			}
		}

		if detectionType == "" {
			return nil, fmt.Errorf("could not determine detection type from checks")
		}

		// Extract baseline ID
		baselineID := ""
		if compareWith, ok := stepConfig["compare_with"].(string); ok {
			baselineID = compareWith
		} else {
			// Try to get baseline from workflow context
			if workflowContext, ok := state["workflow_context"].(map[string]any); ok {
				if baseline, ok := workflowContext["baseline_id"].(string); ok {
					baselineID = baseline
				}
			}
		}

		if baselineID == "" {
			// Try to get current version baseline
			baselineID = fmt.Sprintf("baseline-%s-current", systemName)
		}

		// Extract threshold
		threshold := 0.0
		for _, checkRaw := range checksConfig {
			check, ok := checkRaw.(map[string]interface{})
			if !ok {
				continue
			}
			if thresh, ok := check["threshold"].(float64); ok {
				threshold = thresh
				break
			}
		}

		// Build detection configuration
		detectionConfig := map[string]interface{}{
			"threshold": threshold,
			"checks":    checksConfig,
		}

		// Extract workflow instance ID
		workflowInstanceID := ""
		if workflowID, ok := state["workflow_id"].(string); ok {
			workflowInstanceID = workflowID
		}

		// Create detection request
		detectionRequest := &breakdetection.DetectionRequest{
			SystemName:         systemName,
			BaselineID:         baselineID,
			DetectionType:      detectionType,
			Configuration:      detectionConfig,
			WorkflowInstanceID: workflowInstanceID,
		}

		// Call break detection service
		if opts.Logger != nil {
			opts.Logger.Printf("Processing break detection: system=%s, type=%s, baseline=%s",
				systemName, detectionType, baselineID)
		}

		var result map[string]interface{}
		var detectionResult *breakdetection.DetectionResult
		var err error

		// If break detection service URL is provided, make HTTP call
		if opts.BreakDetectionServiceURL != "" {
			detectionResult, err = callBreakDetectionService(ctx, opts.BreakDetectionServiceURL, detectionRequest, opts.Logger)
			if err != nil {
				// Log error but continue with workflow (don't fail the entire workflow)
				if opts.Logger != nil {
					opts.Logger.Printf("ERROR: Failed to call break detection service: %v", err)
				}
				// Return error result in state
				result = map[string]interface{}{
					"status":              "failed",
					"error":               err.Error(),
					"total_breaks":        0,
					"total_records_checked": 0,
					"breaks":              []interface{}{},
					"detection_type":       string(detectionType),
					"system":              string(systemName),
				}
			} else {
				// Convert DetectionResult to map for state
				result = map[string]interface{}{
					"status":              string(detectionResult.Status),
					"run_id":              detectionResult.RunID,
					"total_breaks":        detectionResult.TotalBreaksDetected,
					"total_records_checked": detectionResult.TotalRecordsChecked,
					"breaks":              convertBreaksToInterface(detectionResult.Breaks),
					"detection_type":       string(detectionType),
					"system":              string(systemName),
					"result_summary":      detectionResult.ResultSummary,
				}
				if detectionResult.ErrorMessage != "" {
					result["error_message"] = detectionResult.ErrorMessage
				}
			}
		} else {
			// No service URL provided, return placeholder result
			if opts.Logger != nil {
				opts.Logger.Printf("WARNING: Break detection service URL not provided, returning placeholder result")
			}
			result = map[string]interface{}{
				"status":              "completed",
				"total_breaks":        0,
				"total_records_checked": 0,
				"breaks":              []interface{}{},
				"detection_type":       string(detectionType),
				"system":              string(systemName),
			}
		}

		// Update state with break detection results
		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}

		newState["break_detection_result"] = result
		newState["break_detection_status"] = "completed"
		
		// Store request for potential retry
		requestJSON, _ := json.Marshal(detectionRequest)
		newState["break_detection_request"] = string(requestJSON)

		// Check if breaks were detected
		if totalBreaks, ok := result["total_breaks"].(int); ok && totalBreaks > 0 {
			newState["break_detection_has_breaks"] = true
			
			// Check on_error action
			onError := "continue" // Default
			if onErr, ok := stepConfig["on_error"].(string); ok {
				onError = onErr
			}
			
			switch onError {
			case "stop", "fail":
				return nil, fmt.Errorf("break detection found %d breaks (on_error=%s)", totalBreaks, onError)
			case "alert", "warn":
				if opts.Logger != nil {
					opts.Logger.Printf("WARNING: Break detection found %d breaks (continuing with alert)", totalBreaks)
				}
				newState["break_detection_alert"] = true
			case "continue":
				// Continue normally
			}
		} else {
			newState["break_detection_has_breaks"] = false
		}

		return newState, nil
	}
}

// callBreakDetectionService makes an HTTP POST request to the break detection service
func callBreakDetectionService(ctx context.Context, serviceURL string, req *breakdetection.DetectionRequest, logger *log.Logger) (*breakdetection.DetectionResult, error) {
	// Ensure service URL ends with /
	baseURL := strings.TrimSuffix(serviceURL, "/")
	endpoint := fmt.Sprintf("%s/catalog/break-detection/detect", baseURL)

	// Marshal request to JSON
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal detection request: %w", err)
	}

	// Create HTTP request with timeout
	requestCtx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create HTTP request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Create HTTP client with timeout
	client := &http.Client{
		Timeout: 30 * time.Second,
	}

	if logger != nil {
		logger.Printf("Calling break detection service: %s", endpoint)
	}

	// Make HTTP request with retry logic
	var resp *http.Response
	maxRetries := 3
	for attempt := 0; attempt < maxRetries; attempt++ {
		if attempt > 0 {
			// Wait before retry (exponential backoff: 1s, 2s, 4s)
			waitTime := time.Duration(1<<uint(attempt-1)) * time.Second
			if logger != nil {
				logger.Printf("Retrying break detection service call (attempt %d/%d) after %v", attempt+1, maxRetries, waitTime)
			}
			select {
			case <-time.After(waitTime):
			case <-requestCtx.Done():
				return nil, fmt.Errorf("request cancelled or timed out: %w", requestCtx.Err())
			}

			// Create new request for retry (with fresh context)
			requestCtx, cancel = context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			httpReq, err = http.NewRequestWithContext(requestCtx, http.MethodPost, endpoint, bytes.NewBuffer(reqBody))
			if err != nil {
				return nil, fmt.Errorf("failed to create HTTP request for retry: %w", err)
			}
			httpReq.Header.Set("Content-Type", "application/json")
		}

		resp, err = client.Do(httpReq)
		if err == nil {
			break
		}

		// Check if context was cancelled
		if requestCtx.Err() != nil {
			return nil, fmt.Errorf("request cancelled or timed out: %w", requestCtx.Err())
		}

		if logger != nil && attempt < maxRetries-1 {
			logger.Printf("HTTP request failed (attempt %d/%d): %v", attempt+1, maxRetries, err)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("failed to call break detection service after %d attempts: %w", maxRetries, err)
	}
	defer resp.Body.Close()

	// Read response body
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	// Check HTTP status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("break detection service returned error status %d: %s", resp.StatusCode, string(respBody))
	}

	// Parse response
	var apiResponse struct {
		Result  *breakdetection.DetectionResult `json:"result"`
		Message string                          `json:"message,omitempty"`
		Error   string                          `json:"error,omitempty"`
	}

	if err := json.Unmarshal(respBody, &apiResponse); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %w", err)
	}

	if apiResponse.Error != "" {
		return nil, fmt.Errorf("break detection service error: %s", apiResponse.Error)
	}

	if apiResponse.Result == nil {
		return nil, fmt.Errorf("break detection service returned nil result")
	}

	if logger != nil {
		logger.Printf("Break detection completed: run_id=%s, breaks=%d, records_checked=%d",
			apiResponse.Result.RunID, apiResponse.Result.TotalBreaksDetected, apiResponse.Result.TotalRecordsChecked)
	}

	return apiResponse.Result, nil
}

// convertBreaksToInterface converts []*Break to []interface{} for state storage
func convertBreaksToInterface(breaks []*breakdetection.Break) []interface{} {
	result := make([]interface{}, len(breaks))
	for i, b := range breaks {
		// Convert break to map for JSON serialization
		breakMap := map[string]interface{}{
			"break_id":         b.BreakID,
			"run_id":           b.RunID,
			"system_name":      string(b.SystemName),
			"detection_type":   string(b.DetectionType),
			"break_type":       string(b.BreakType),
			"severity":         string(b.Severity),
			"status":           string(b.Status),
			"current_value":     b.CurrentValue,
			"baseline_value":   b.BaselineValue,
			"difference":       b.Difference,
			"affected_entities": b.AffectedEntities,
			"detected_at":      b.DetectedAt,
		}
		if b.RootCauseAnalysis != "" {
			breakMap["root_cause_analysis"] = b.RootCauseAnalysis
		}
		if len(b.Recommendations) > 0 {
			breakMap["recommendations"] = b.Recommendations
		}
		if b.AIDescription != "" {
			breakMap["ai_description"] = b.AIDescription
		}
		if b.AICategory != "" {
			breakMap["ai_category"] = b.AICategory
		}
		if b.AIPriorityScore > 0 {
			breakMap["ai_priority_score"] = b.AIPriorityScore
		}
		if len(b.SimilarBreaks) > 0 {
			breakMap["similar_breaks"] = b.SimilarBreaks
		}
		result[i] = breakMap
	}
	return result
}

// RegisterBreakDetectionNode registers break detection as a workflow step type
// Note: This would be integrated with the workflow engine to handle quality-check steps
// The actual registration would depend on the workflow engine being used

