package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// GPUProcessorOptions configures the GPU orchestration workflow.
type GPUProcessorOptions struct {
	GPUOrchestratorURL string // URL to GPU orchestrator service (e.g., "http://gpu-orchestrator:8086")
}

// GPUAllocationRequest represents a request to allocate GPUs as part of a workflow.
type GPUAllocationRequest struct {
	ServiceName   string                 `json:"service_name"`
	WorkloadType  string                 `json:"workload_type"`
	WorkloadData  map[string]interface{} `json:"workload_data,omitempty"`
	WorkflowID    string                 `json:"workflow_id,omitempty"`
}

// GPUAllocation represents a GPU allocation result.
type GPUAllocation struct {
	AllocationID string   `json:"allocation_id"`
	ServiceName   string   `json:"service_name"`
	GPUIDs       []int    `json:"gpu_ids"`
	AllocatedAt  string   `json:"allocated_at"`
	ExpiresAt    *string  `json:"expires_at,omitempty"`
	Priority     int      `json:"priority"`
}

// gpuHTTPClient uses connection pooling for better performance (Phase 1)
var gpuHTTPClient = &http.Client{
	Transport: &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
	},
	Timeout: 30 * time.Second,
}

// ProcessGPUAllocationNode returns a node that processes GPU allocation requests.
func ProcessGPUAllocationNode(opts GPUProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Get GPU orchestrator URL from options or environment
		gpuOrchestratorURL := opts.GPUOrchestratorURL
		if gpuOrchestratorURL == "" {
			gpuOrchestratorURL = os.Getenv("GPU_ORCHESTRATOR_URL")
			if gpuOrchestratorURL == "" {
				gpuOrchestratorURL = "http://gpu-orchestrator:8086"
			}
		}

		// Extract GPU allocation request from state
		var gpuReq GPUAllocationRequest

		if req, ok := state["gpu_allocation_request"].(map[string]any); ok {
			gpuReq.ServiceName = getString(req["service_name"])
			gpuReq.WorkloadType = getString(req["workload_type"])
			gpuReq.WorkflowID = getString(req["workflow_id"])
			
			// Extract workflow priority and dependencies (Phase 1)
			if priority, ok := req["workflow_priority"].(float64); ok {
				if gpuReq.WorkloadData == nil {
					gpuReq.WorkloadData = make(map[string]interface{})
				}
				gpuReq.WorkloadData["workflow_priority"] = int(priority)
			}
			if deps, ok := req["workflow_dependencies"].([]interface{}); ok {
				if gpuReq.WorkloadData == nil {
					gpuReq.WorkloadData = make(map[string]interface{})
				}
				gpuReq.WorkloadData["workflow_dependencies"] = deps
			}

			if workloadData, ok := req["workload_data"].(map[string]any); ok {
				gpuReq.WorkloadData = workloadData
			}
		} else {
			// Try to infer from unified workflow request
			if unifiedReq, ok := state["unified_request"].(map[string]any); ok {
				// Determine workload type from unified request
				if kgReq, ok := unifiedReq["knowledge_graph_request"].(map[string]any); ok && kgReq != nil {
					gpuReq.WorkloadType = "graph_processing"
					gpuReq.WorkloadData = map[string]interface{}{
						"node_count": 10000, // Estimate
					}
				} else if orchReq, ok := unifiedReq["orchestration_request"].(map[string]any); ok && orchReq != nil {
					gpuReq.WorkloadType = "inference"
					chainName := getString(orchReq["chain_name"])
					gpuReq.WorkloadData = map[string]interface{}{
						"chain_name": chainName,
					}
					// Extract workflow ID and priority if available (Phase 1)
					if workflowID, ok := state["workflow_id"].(string); ok {
						gpuReq.WorkflowID = workflowID
						gpuReq.WorkloadData["workflow_id"] = workflowID
					}
					if priority, ok := state["workflow_priority"].(float64); ok {
						gpuReq.WorkloadData["workflow_priority"] = int(priority)
					}
				} else if afReq, ok := unifiedReq["agentflow_request"].(map[string]any); ok && afReq != nil {
					gpuReq.WorkloadType = "inference"
					flowID := getString(afReq["flow_id"])
					gpuReq.WorkloadData = map[string]interface{}{
						"flow_id": flowID,
					}
					// Extract workflow ID and priority if available (Phase 1)
					if workflowID, ok := state["workflow_id"].(string); ok {
						gpuReq.WorkflowID = workflowID
						gpuReq.WorkloadData["workflow_id"] = workflowID
					}
					if priority, ok := state["workflow_priority"].(float64); ok {
						gpuReq.WorkloadData["workflow_priority"] = int(priority)
					}
				}
			}
		}

		if gpuReq.ServiceName == "" {
			log.Println("No GPU allocation request found; skipping GPU allocation")
			return state, nil
		}

		if gpuReq.WorkloadType == "" {
			gpuReq.WorkloadType = "generic"
		}

		log.Printf("Processing GPU allocation for service: %s, workload type: %s", gpuReq.ServiceName, gpuReq.WorkloadType)

		// Ensure workflow ID is included in workload data (Phase 1)
		if gpuReq.WorkflowID != "" {
			if gpuReq.WorkloadData == nil {
				gpuReq.WorkloadData = make(map[string]interface{})
			}
			gpuReq.WorkloadData["workflow_id"] = gpuReq.WorkflowID
		}
		
		// Call GPU orchestrator API
		requestBody := map[string]interface{}{
			"service_name":  gpuReq.ServiceName,
			"workload_type": gpuReq.WorkloadType,
			"workload_data": gpuReq.WorkloadData,
		}

		jsonData, err := json.Marshal(requestBody)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal GPU allocation request: %w", err)
		}

		url := fmt.Sprintf("%s/gpu/allocate", gpuOrchestratorURL)
		resp, err := gpuHTTPClient.Post(url, "application/json", bytes.NewReader(jsonData))
		if err != nil {
			log.Printf("Warning: Failed to allocate GPU: %v", err)
			// Continue without GPU allocation
			return state, nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			log.Printf("Warning: GPU orchestrator returned status %d", resp.StatusCode)
			// Continue without GPU allocation
			return state, nil
		}

		var allocation GPUAllocation
		if err := json.NewDecoder(resp.Body).Decode(&allocation); err != nil {
			log.Printf("Warning: Failed to decode GPU allocation response: %v", err)
			// Continue without GPU allocation
			return state, nil
		}

		log.Printf("Successfully allocated GPUs: %v for service: %s", allocation.GPUIDs, allocation.ServiceName)

		// Merge GPU allocation into state
		newState := make(map[string]any, len(state)+2)
		for k, v := range state {
			newState[k] = v
		}

		newState["gpu_allocation"] = map[string]any{
			"allocation_id": allocation.AllocationID,
			"service_name":  allocation.ServiceName,
			"gpu_ids":       allocation.GPUIDs,
			"allocated_at":  allocation.AllocatedAt,
			"expires_at":    allocation.ExpiresAt,
			"priority":      allocation.Priority,
		}

		return newState, nil
	})
}

// ReleaseGPUAllocationNode returns a node that releases GPU allocations.
func ReleaseGPUAllocationNode(opts GPUProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Get GPU orchestrator URL
		gpuOrchestratorURL := opts.GPUOrchestratorURL
		if gpuOrchestratorURL == "" {
			gpuOrchestratorURL = os.Getenv("GPU_ORCHESTRATOR_URL")
			if gpuOrchestratorURL == "" {
				gpuOrchestratorURL = "http://gpu-orchestrator:8086"
			}
		}

		// Extract allocation ID from state
		var allocationID string
		if gpuAlloc, ok := state["gpu_allocation"].(map[string]any); ok {
			if id, ok := gpuAlloc["allocation_id"].(string); ok {
				allocationID = id
			}
		}

		if allocationID == "" {
			log.Println("No GPU allocation found to release")
			return state, nil
		}

		log.Printf("Releasing GPU allocation: %s", allocationID)

		// Call GPU orchestrator API to release
		requestBody := map[string]interface{}{
			"allocation_id": allocationID,
		}

		jsonData, err := json.Marshal(requestBody)
		if err != nil {
			log.Printf("Warning: Failed to marshal GPU release request: %v", err)
			return state, nil
		}

		url := fmt.Sprintf("%s/gpu/release", gpuOrchestratorURL)
		resp, err := gpuHTTPClient.Post(url, "application/json", bytes.NewReader(jsonData))
		if err != nil {
			log.Printf("Warning: Failed to release GPU: %v", err)
			return state, nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			log.Printf("Warning: GPU orchestrator returned status %d for release", resp.StatusCode)
			return state, nil
		}

		log.Printf("Successfully released GPU allocation: %s", allocationID)

		// Remove GPU allocation from state
		newState := make(map[string]any)
		for k, v := range state {
			if k != "gpu_allocation" {
				newState[k] = v
			}
		}

		return newState, nil
	})
}

