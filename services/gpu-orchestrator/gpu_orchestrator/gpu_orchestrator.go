package gpu_orchestrator

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_allocator"
	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_monitor"
	"github.com/plturrell/aModels/services/gpu-orchestrator/scheduler"
	"github.com/plturrell/aModels/services/gpu-orchestrator/workload_analyzer"
)

// GPUOrchestrator is the main orchestrator that coordinates GPU allocation
type GPUOrchestrator struct {
	allocator       *gpu_allocator.GPUAllocator
	scheduler       *scheduler.Scheduler
	workloadAnalyzer *workload_analyzer.WorkloadAnalyzer
	monitor         *gpu_monitor.GPUMonitor
	deepAgentsURL   string
	graphServiceURL string
	httpClient      *http.Client
	logger          *log.Logger
}

// NewGPUOrchestrator creates a new GPU orchestrator
func NewGPUOrchestrator(
	allocator *gpu_allocator.GPUAllocator,
	scheduler *scheduler.Scheduler,
	workloadAnalyzer *workload_analyzer.WorkloadAnalyzer,
	monitor *gpu_monitor.GPUMonitor,
	deepAgentsURL string,
	graphServiceURL string,
	logger *log.Logger,
) *GPUOrchestrator {
	return &GPUOrchestrator{
		allocator:       allocator,
		scheduler:       scheduler,
		workloadAnalyzer: workloadAnalyzer,
		monitor:         monitor,
		deepAgentsURL:   deepAgentsURL,
		graphServiceURL: graphServiceURL,
		httpClient:      &http.Client{Timeout: 60 * time.Second},
		logger:          logger,
	}
}

// AllocateGPUs allocates GPUs for a service using intelligent scheduling
func (o *GPUOrchestrator) AllocateGPUs(ctx context.Context, serviceName string, workloadType string, workloadData map[string]interface{}) (*gpu_allocator.Allocation, error) {
	// Extract workflow context from workload data (Phase 1)
	workflowID := ""
	workflowPriority := 5 // Default priority (1-10 scale)
	workflowDependencies := []string{}
	
	if workloadData != nil {
		if wfID, ok := workloadData["workflow_id"].(string); ok {
			workflowID = wfID
		}
		if priority, ok := workloadData["workflow_priority"].(int); ok {
			workflowPriority = priority
		} else if priority, ok := workloadData["workflow_priority"].(float64); ok {
			workflowPriority = int(priority)
		}
		if deps, ok := workloadData["workflow_dependencies"].([]interface{}); ok {
			for _, dep := range deps {
				if depStr, ok := dep.(string); ok {
					workflowDependencies = append(workflowDependencies, depStr)
				}
			}
		}
		
		// Ensure workflow context is in workload data for scheduler
		if workflowID != "" {
			workloadData["workflow_id"] = workflowID
		}
		workloadData["workflow_priority"] = workflowPriority
		if len(workflowDependencies) > 0 {
			workloadData["workflow_dependencies"] = workflowDependencies
		}
	}
	
	if o.logger != nil && workflowID != "" {
		o.logger.Printf("Allocating GPUs for workflow %s (priority: %d, dependencies: %v)", workflowID, workflowPriority, workflowDependencies)
	}
	
	// Use DeepAgents for intelligent allocation if available
	if o.deepAgentsURL != "" {
		allocation, err := o.allocateViaDeepAgents(ctx, serviceName, workloadType, workloadData)
		if err == nil {
			return allocation, nil
		}
		o.logger.Printf("DeepAgents allocation failed, falling back to standard allocation: %v", err)
	}

	// Fallback to priority-based scheduling (Phase 1)
	return o.scheduler.Schedule(serviceName, workloadType, workloadData)
}

// allocateViaDeepAgents uses DeepAgents to intelligently allocate GPUs
func (o *GPUOrchestrator) allocateViaDeepAgents(ctx context.Context, serviceName string, workloadType string, workloadData map[string]interface{}) (*gpu_allocator.Allocation, error) {
	// Use analyze_workload tool via DeepAgents with structured output
	workloadDataJSON, _ := json.Marshal(workloadData)
	prompt := fmt.Sprintf(
		"Use the analyze_workload tool to analyze GPU requirements for service '%s' with workload type '%s'.\n\n"+
			"Workload data: %s\n\n"+
			"Then use the query_gpu_status tool to check current GPU availability.\n\n"+
			"Based on the analysis, provide a structured recommendation with required_gpus, min_memory_mb, and priority.",
		serviceName, workloadType, string(workloadDataJSON))

	// Define JSON schema for structured output
	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"required_gpus": map[string]interface{}{
				"type":    "integer",
				"minimum": 0,
			},
			"min_memory_mb": map[string]interface{}{
				"type":    "integer",
				"minimum": 0,
			},
			"priority": map[string]interface{}{
				"type":    "integer",
				"minimum": 1,
				"maximum": 10,
			},
			"reasoning": map[string]interface{}{
				"type": "string",
			},
		},
		"required": []string{"required_gpus", "min_memory_mb", "priority"},
	}

	request := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"response_format": map[string]interface{}{
			"type":       "json_schema",
			"json_schema": jsonSchema,
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/invoke/structured", o.deepAgentsURL)
	resp, err := o.httpClient.Post(url, "application/json", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call DeepAgents: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DeepAgents returned status %d", resp.StatusCode)
	}

	var agentResponse struct {
		Messages         []map[string]interface{} `json:"messages"`
		StructuredOutput map[string]interface{}    `json:"structured_output"`
		Result           interface{}               `json:"result,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&agentResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Extract allocation strategy from structured output
	allocationReq := o.parseAllocationFromStructured(agentResponse.StructuredOutput, serviceName, workloadType, workloadData)
	if allocationReq != nil {
		// Use parsed allocation strategy
		return o.allocator.Allocate(allocationReq)
	}

	// Fall back to standard scheduling if parsing fails
	if o.logger != nil {
		o.logger.Printf("Failed to parse allocation strategy from structured output, using standard scheduling")
	}
	return o.scheduler.Schedule(serviceName, workloadType, workloadData)
}

// parseAllocationFromStructured extracts allocation from structured output.
func (o *GPUOrchestrator) parseAllocationFromStructured(
	structuredOutput map[string]interface{},
	serviceName string,
	workloadType string,
	workloadData map[string]interface{},
) *gpu_allocator.AllocationRequest {
	if structuredOutput == nil {
		return nil
	}

	requiredGPUs := 1
	if rg, ok := structuredOutput["required_gpus"].(float64); ok {
		requiredGPUs = int(rg)
	}

	minMemoryMB := 0
	if mm, ok := structuredOutput["min_memory_mb"].(float64); ok {
		minMemoryMB = int(mm)
	}

	priority := 5
	if p, ok := structuredOutput["priority"].(float64); ok {
		priority = int(p)
	}

	return &gpu_allocator.AllocationRequest{
		ServiceName:  serviceName,
		WorkloadType: workloadType,
		WorkloadData: workloadData,
		RequiredGPUs: requiredGPUs,
		MinMemoryMB:  minMemoryMB,
		Priority:     priority,
	}
}

// parseAllocationStrategy extracts allocation strategy from agent response.
// Looks for JSON in messages or result containing: required_gpus, min_memory_mb, priority.
func (o *GPUOrchestrator) parseAllocationStrategy(
	agentResponse struct {
		Messages []map[string]interface{} `json:"messages"`
		Result   interface{}               `json:"result,omitempty"`
	},
	serviceName string,
	workloadType string,
	workloadData map[string]interface{},
) *gpu_allocator.AllocationRequest {
	// Try to extract from Result field first
	if agentResponse.Result != nil {
		if req := o.extractAllocationFromData(agentResponse.Result, serviceName); req != nil {
			return req
		}
	}

	// Try to extract from Messages (look for assistant messages with JSON content)
	for _, msg := range agentResponse.Messages {
		if role, ok := msg["role"].(string); ok && role == "assistant" {
			if content, ok := msg["content"].(string); ok {
				// Try to parse JSON from content
				var data interface{}
				if err := json.Unmarshal([]byte(content), &data); err == nil {
					if req := o.extractAllocationFromData(data, serviceName); req != nil {
						return req
					}
				}
			}
			// Also check if message has direct allocation data
			if req := o.extractAllocationFromData(msg, serviceName); req != nil {
				return req
			}
		}
	}

	return nil
}

// extractAllocationFromData extracts allocation request from various data formats.
func (o *GPUOrchestrator) extractAllocationFromData(data interface{}, serviceName string) *gpu_allocator.AllocationRequest {
	// Convert to map for easier access
	var dataMap map[string]interface{}
	switch v := data.(type) {
	case map[string]interface{}:
		dataMap = v
	default:
		// Try to marshal/unmarshal to convert to map
		jsonBytes, err := json.Marshal(data)
		if err != nil {
			return nil
		}
		if err := json.Unmarshal(jsonBytes, &dataMap); err != nil {
			return nil
		}
	}

	// Extract allocation fields
	req := &gpu_allocator.AllocationRequest{
		ServiceName: serviceName,
		RequiredGPUs: 1, // Default
		Priority:     5,  // Default medium priority
	}

	// Extract required_gpus
	if rg, ok := dataMap["required_gpus"].(float64); ok {
		req.RequiredGPUs = int(rg)
	} else if rg, ok := dataMap["required_gpus"].(int); ok {
		req.RequiredGPUs = rg
	}

	// Extract min_memory_mb
	if mm, ok := dataMap["min_memory_mb"].(float64); ok {
		req.MinMemoryMB = int64(mm)
	} else if mm, ok := dataMap["min_memory_mb"].(int64); ok {
		req.MinMemoryMB = mm
	} else if mm, ok := dataMap["min_memory_mb"].(int); ok {
		req.MinMemoryMB = int64(mm)
	}

	// Extract priority
	if p, ok := dataMap["priority"].(float64); ok {
		req.Priority = int(p)
	} else if p, ok := dataMap["priority"].(int); ok {
		req.Priority = p
	}

	// Extract max_memory_mb (optional)
	if mm, ok := dataMap["max_memory_mb"].(float64); ok {
		req.MaxMemoryMB = int64(mm)
	} else if mm, ok := dataMap["max_memory_mb"].(int64); ok {
		req.MaxMemoryMB = mm
	} else if mm, ok := dataMap["max_memory_mb"].(int); ok {
		req.MaxMemoryMB = int64(mm)
	}

	// Extract max_utilization (optional)
	if mu, ok := dataMap["max_utilization"].(float64); ok {
		req.MaxUtilization = mu
	}

	// Extract preferred_gpus (optional)
	if pg, ok := dataMap["preferred_gpus"].([]interface{}); ok {
		preferredGPUs := make([]int, 0, len(pg))
		for _, gpuID := range pg {
			switch v := gpuID.(type) {
			case float64:
				preferredGPUs = append(preferredGPUs, int(v))
			case int:
				preferredGPUs = append(preferredGPUs, v)
			}
		}
		if len(preferredGPUs) > 0 {
			req.PreferredGPUs = preferredGPUs
		}
	}

	// Only return if we found at least required_gpus or min_memory_mb
	if req.RequiredGPUs > 0 || req.MinMemoryMB > 0 {
		if o.logger != nil {
			o.logger.Printf("Parsed allocation strategy: GPUs=%d, MemoryMB=%d, Priority=%d",
				req.RequiredGPUs, req.MinMemoryMB, req.Priority)
		}
		return req
	}

	return nil
}

// ReleaseGPUs releases GPUs for a service
func (o *GPUOrchestrator) ReleaseGPUs(allocationID string) error {
	return o.allocator.Release(allocationID)
}

// ReleaseGPUsByService releases all GPUs for a service
func (o *GPUOrchestrator) ReleaseGPUsByService(serviceName string) error {
	return o.allocator.ReleaseByService(serviceName)
}

// GetGPUStatus returns the current GPU status
func (o *GPUOrchestrator) GetGPUStatus() ([]*gpu_monitor.GPUInfo, error) {
	if o.monitor == nil {
		return nil, fmt.Errorf("GPU monitoring not available")
	}

	return o.monitor.ListGPUs(), nil
}

// GetAllocationStatus returns the status of an allocation
func (o *GPUOrchestrator) GetAllocationStatus(allocationID string) (*gpu_allocator.Allocation, error) {
	return o.allocator.GetAllocation(allocationID)
}

// ListAllocations returns all active allocations
func (o *GPUOrchestrator) ListAllocations() []*gpu_allocator.Allocation {
	return o.allocator.ListAllocations()
}

// GetQueueStatus returns the scheduler queue status
func (o *GPUOrchestrator) GetQueueStatus() []*scheduler.ScheduledRequest {
	return o.scheduler.GetQueueStatus()
}

// AnalyzeWorkload analyzes a workload and returns requirements
func (o *GPUOrchestrator) AnalyzeWorkload(workloadType string, workloadData map[string]interface{}) (*workload_analyzer.WorkloadRequirements, error) {
	return o.workloadAnalyzer.AnalyzeWorkload(workloadType, workloadData)
}

