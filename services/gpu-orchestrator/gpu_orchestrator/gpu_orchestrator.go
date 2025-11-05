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

	// Extract allocation strategy from agent response
	// For now, fall back to standard analysis
	// TODO: Parse agent response to extract allocation strategy

	return o.scheduler.Schedule(serviceName, workloadType, workloadData)
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

