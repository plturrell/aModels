package workflows

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

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
		// Note: This would typically be an HTTP call to the break detection service
		// For now, we'll create a placeholder that returns the request
		// In production, this would make an actual HTTP call to the break detection service
		if opts.Logger != nil {
			opts.Logger.Printf("Processing break detection: system=%s, type=%s, baseline=%s",
				systemName, detectionType, baselineID)
		}

		// TODO: Make actual HTTP call to break detection service
		// For now, we'll simulate the result
		result := map[string]interface{}{
			"status":              "completed",
			"total_breaks":        0,
			"total_records_checked": 0,
			"breaks":              []interface{}{},
			"detection_type":       string(detectionType),
			"system":              string(systemName),
		}

		// If break detection service URL is provided, make HTTP call
		if opts.BreakDetectionServiceURL != "" {
			// In production, use HTTP client to call the service
			// For now, we'll return a placeholder
			if opts.Logger != nil {
				opts.Logger.Printf("Would call break detection service at: %s", opts.BreakDetectionServiceURL)
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

// RegisterBreakDetectionNode registers break detection as a workflow step type
// Note: This would be integrated with the workflow engine to handle quality-check steps
// The actual registration would depend on the workflow engine being used

