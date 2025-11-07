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
	// Use DeepAgents for intelligent allocation if available
	if o.deepAgentsURL != "" {
		allocation, err := o.allocateViaDeepAgents(ctx, serviceName, workloadType, workloadData)
		if err == nil {
			return allocation, nil
		}
		o.logger.Printf("DeepAgents allocation failed, falling back to standard allocation: %v", err)
	}

	// Fallback to standard scheduling
	return o.scheduler.Schedule(serviceName, workloadType, workloadData)
}

// allocateViaDeepAgents uses DeepAgents to intelligently allocate GPUs
func (o *GPUOrchestrator) allocateViaDeepAgents(ctx context.Context, serviceName string, workloadType string, workloadData map[string]interface{}) (*gpu_allocator.Allocation, error) {
	// Query DeepAgents for allocation strategy
	request := map[string]interface{}{
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": fmt.Sprintf(
					"Analyze GPU allocation for service '%s' with workload type '%s'. "+
						"Determine optimal GPU allocation strategy considering: "+
						"1. Required number of GPUs, 2. Memory requirements, 3. Priority, "+
						"4. Current GPU utilization. Return JSON with 'required_gpus', 'min_memory_mb', 'priority' fields.",
					serviceName, workloadType),
			},
		},
		"config": map[string]interface{}{
			"workload_data": workloadData,
		},
	}

	jsonData, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/invoke", o.deepAgentsURL)
	resp, err := o.httpClient.Post(url, "application/json", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to call DeepAgents: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("DeepAgents returned status %d", resp.StatusCode)
	}

	var agentResponse struct {
		Messages []map[string]interface{} `json:"messages"`
		Result   interface{}               `json:"result,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&agentResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Parse agent response to extract allocation strategy
	allocationReq := o.parseAllocationStrategy(agentResponse, serviceName, workloadType, workloadData)
	if allocationReq != nil {
		// Use parsed allocation strategy
		return o.allocator.Allocate(allocationReq)
	}

	// Fall back to standard scheduling if parsing fails
	if o.logger != nil {
		o.logger.Printf("Failed to parse allocation strategy from agent response, using standard scheduling")
	}
	return o.scheduler.Schedule(serviceName, workloadType, workloadData)
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

