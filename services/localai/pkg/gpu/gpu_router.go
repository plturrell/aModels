package gpu

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// GPURouter manages GPU allocation and routing for LocalAI
type GPURouter struct {
	orchestratorURL string
	allocatedGPUs   []int
	allocationID    string
	mu              sync.RWMutex
	httpClient      *http.Client
	logger          *log.Logger
}

// NewGPURouter creates a new GPU router
func NewGPURouter(orchestratorURL string, logger *log.Logger) *GPURouter {
	if logger == nil {
		logger = log.New(os.Stdout, "[gpu-router] ", log.LstdFlags|log.Lmsgprefix)
	}
	
	return &GPURouter{
		orchestratorURL: orchestratorURL,
		httpClient:      &http.Client{Timeout: 10 * time.Second},
		logger:          logger,
	}
}

// AllocateGPUs allocates GPUs from the orchestrator for LocalAI service
func (r *GPURouter) AllocateGPUs(ctx context.Context, requiredGPUs int) error {
	_, _, err := r.AllocateGPUsWithWorkload(ctx, requiredGPUs, nil)
	return err
}

// AllocateGPUsWithWorkload allocates GPUs with detailed workload data
// Returns allocation ID, allocated GPU IDs, and error
func (r *GPURouter) AllocateGPUsWithWorkload(ctx context.Context, requiredGPUs int, workloadData map[string]interface{}) (string, []int, error) {
	if r.orchestratorURL == "" {
		r.logger.Println("GPU orchestrator URL not configured, skipping allocation")
		return "", nil, nil
	}

	// Build workload data with defaults
	if workloadData == nil {
		workloadData = make(map[string]interface{})
	}
	workloadData["required_gpus"] = requiredGPUs
	
	// Ensure service_name is set
	if _, ok := workloadData["service_name"]; !ok {
		workloadData["service_name"] = "localai"
	}

	requestData := map[string]interface{}{
		"service_name":  workloadData["service_name"],
		"workload_type": "inference",
		"workload_data": workloadData,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return "", nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/gpu/allocate", r.orchestratorURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return "", nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(req)
	if err != nil {
		r.logger.Printf("Warning: Failed to allocate GPUs: %v", err)
		return "", nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		r.logger.Printf("Warning: GPU orchestrator returned status %d", resp.StatusCode)
		return "", nil, fmt.Errorf("GPU orchestrator returned status %d", resp.StatusCode)
	}

	var allocation struct {
		ID      string `json:"id"`
		GPUIDs  []int  `json:"gpu_ids"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&allocation); err != nil {
		return "", nil, fmt.Errorf("failed to decode response: %w", err)
	}

	r.mu.Lock()
	r.allocationID = allocation.ID
	r.allocatedGPUs = allocation.GPUIDs
	r.mu.Unlock()

	// Set CUDA_VISIBLE_DEVICES environment variable
	if len(allocation.GPUIDs) > 0 {
		gpuList := make([]string, len(allocation.GPUIDs))
		for i, gpuID := range allocation.GPUIDs {
			gpuList[i] = fmt.Sprintf("%d", gpuID)
		}
		os.Setenv("CUDA_VISIBLE_DEVICES", strings.Join(gpuList, ","))
		r.logger.Printf("Allocated GPUs %v from orchestrator (allocation ID: %s)", allocation.GPUIDs, allocation.ID)
	}

	return allocation.ID, allocation.GPUIDs, nil
}

// ReleaseGPUs releases allocated GPUs
func (r *GPURouter) ReleaseGPUs(ctx context.Context) error {
	r.mu.RLock()
	allocationID := r.allocationID
	r.mu.RUnlock()

	if allocationID == "" || r.orchestratorURL == "" {
		return nil
	}

	requestData := map[string]interface{}{
		"allocation_id": allocationID,
	}

	jsonData, err := json.Marshal(requestData)
	if err != nil {
		return fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/gpu/release", r.orchestratorURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(req)
	if err != nil {
		r.logger.Printf("Warning: Failed to release GPUs: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		r.logger.Printf("Warning: GPU orchestrator returned status %d for release", resp.StatusCode)
	}

	r.mu.Lock()
	r.allocationID = ""
	r.allocatedGPUs = nil
	r.mu.Unlock()

	r.logger.Printf("Released GPU allocation %s", allocationID)
	return nil
}

// GetGPUs returns the currently allocated GPU IDs
func (r *GPURouter) GetGPUs() []int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	
	gpus := make([]int, len(r.allocatedGPUs))
	copy(gpus, r.allocatedGPUs)
	return gpus
}

// GetAllocationID returns the current allocation ID
func (r *GPURouter) GetAllocationID() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.allocationID
}

// SelectGPUForModel selects the best GPU for a model based on size and current utilization
// For now, uses round-robin across allocated GPUs
func (r *GPURouter) SelectGPUForModel(modelName string, modelSize int64) int {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.allocatedGPUs) == 0 {
		return 0 // Default to GPU 0
	}

	// Simple round-robin based on model name hash
	hash := 0
	for _, char := range modelName {
		hash += int(char)
	}
	return r.allocatedGPUs[hash%len(r.allocatedGPUs)]
}

