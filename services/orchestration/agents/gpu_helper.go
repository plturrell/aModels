package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// GPUHelper provides shared GPU allocation functionality for all pipelines (Phase 3)
type GPUHelper struct {
	gpuOrchestratorURL string
	httpClient         *http.Client
	logger             *log.Logger
}

// NewGPUHelper creates a new GPU helper
func NewGPUHelper(gpuOrchestratorURL string, httpClient *http.Client, logger *log.Logger) *GPUHelper {
	if gpuOrchestratorURL == "" {
		gpuOrchestratorURL = "http://gpu-orchestrator:8086"
	}
	return &GPUHelper{
		gpuOrchestratorURL: gpuOrchestratorURL,
		httpClient:         httpClient,
		logger:             logger,
	}
}

// RequestGPUAllocation requests GPU allocation from the GPU orchestrator (Phase 3)
func (gh *GPUHelper) RequestGPUAllocation(ctx context.Context, workloadType, domain string, contentSize int, priority int) (string, error) {
	if gh.gpuOrchestratorURL == "" {
		return "", fmt.Errorf("GPU orchestrator URL not configured")
	}

	// Estimate GPU requirements based on content size and workload type
	requiredGPUs := 1
	minMemoryMB := 4096 // 4GB default
	
	switch workloadType {
	case "inference":
		// Inference operations typically need less memory
		if contentSize > 500000 { // > 500KB
			minMemoryMB = 8192 // 8GB for large inference
		}
	case "domain_learning", "embedding_generation":
		// Learning operations need more memory
		if contentSize > 1000000 { // > 1MB
			minMemoryMB = 8192 // 8GB for large documents
		} else {
			minMemoryMB = 6144 // 6GB for medium documents
		}
	case "batch_processing":
		// Batch operations may need more resources
		minMemoryMB = 12288 // 12GB for batch processing
		requiredGPUs = 1 // Can be scaled up if needed
	}

	if priority == 0 {
		priority = 7 // Default priority
	}

	workloadData := map[string]interface{}{
		"workload_type":    workloadType,
		"domain":           domain,
		"content_size":     contentSize,
		"required_gpus":    requiredGPUs,
		"min_memory_mb":    minMemoryMB,
		"priority":         priority,
	}

	requestBody := map[string]interface{}{
		"service_name":  "orchestration",
		"workload_type": workloadType,
		"workload_data": workloadData,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("marshal GPU allocation request: %w", err)
	}

	url := strings.TrimRight(gh.gpuOrchestratorURL, "/") + "/gpu/allocate"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return "", fmt.Errorf("create GPU allocation request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := gh.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("execute GPU allocation request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("GPU orchestrator returned status %d: %s", resp.StatusCode, string(body))
	}

	var allocation struct {
		ID     string `json:"id"`
		GPUIDs []int  `json:"gpu_ids"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&allocation); err != nil {
		return "", fmt.Errorf("decode GPU allocation response: %w", err)
	}

	if gh.logger != nil {
		gh.logger.Printf("Allocated GPU for %s (domain: %s, allocation ID: %s)", workloadType, domain, allocation.ID)
	}

	return allocation.ID, nil
}

// ReleaseGPUAllocation releases GPU allocation (Phase 3)
func (gh *GPUHelper) ReleaseGPUAllocation(ctx context.Context, allocationID string) error {
	if gh.gpuOrchestratorURL == "" || allocationID == "" {
		return nil // No-op if not configured
	}

	requestBody := map[string]interface{}{
		"allocation_id": allocationID,
		"service_name":  "orchestration",
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("marshal GPU release request: %w", err)
	}

	url := strings.TrimRight(gh.gpuOrchestratorURL, "/") + "/gpu/release"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return fmt.Errorf("create GPU release request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := gh.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute GPU release request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		if gh.logger != nil {
			gh.logger.Printf("Warning: GPU release returned status %d: %s", resp.StatusCode, string(body))
		}
		return fmt.Errorf("GPU orchestrator returned status %d: %s", resp.StatusCode, string(body))
	}

	if gh.logger != nil {
		gh.logger.Printf("Released GPU allocation: %s", allocationID)
	}

	return nil
}

