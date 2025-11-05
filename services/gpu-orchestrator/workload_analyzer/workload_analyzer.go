package workload_analyzer

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// WorkloadRequirements represents GPU requirements for a workload
type WorkloadRequirements struct {
	RequiredGPUs   int     `json:"required_gpus"`
	MinMemoryMB    int64   `json:"min_memory_mb"`
	MaxMemoryMB    int64   `json:"max_memory_mb,omitempty"`
	Priority       int     `json:"priority"`
	MaxUtilization float64 `json:"max_utilization,omitempty"`
	EstimatedDuration *time.Duration `json:"estimated_duration,omitempty"`
}

// WorkloadAnalyzer analyzes workloads to determine GPU requirements
type WorkloadAnalyzer struct {
	graphServiceURL string
	httpClient      *http.Client
	logger          *log.Logger
}

// NewWorkloadAnalyzer creates a new workload analyzer
func NewWorkloadAnalyzer(graphServiceURL string, logger *log.Logger) *WorkloadAnalyzer {
	return &WorkloadAnalyzer{
		graphServiceURL: graphServiceURL,
		httpClient:      &http.Client{Timeout: 30 * time.Second},
		logger:          logger,
	}
}

// AnalyzeWorkload analyzes a workload request and returns GPU requirements
func (w *WorkloadAnalyzer) AnalyzeWorkload(workloadType string, workloadData map[string]interface{}) (*WorkloadRequirements, error) {
	switch workloadType {
	case "training":
		return w.analyzeTrainingWorkload(workloadData)
	case "inference":
		return w.analyzeInferenceWorkload(workloadData)
	case "embedding":
		return w.analyzeEmbeddingWorkload(workloadData)
	case "ocr":
		return w.analyzeOCRWorkload(workloadData)
	case "graph_processing":
		return w.analyzeGraphProcessingWorkload(workloadData)
	default:
		return w.analyzeGenericWorkload(workloadData)
	}
}

// analyzeTrainingWorkload analyzes training workload requirements
func (w *WorkloadAnalyzer) analyzeTrainingWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	// Default requirements for training
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    8192, // 8GB minimum
		Priority:       5,
		MaxUtilization: 90.0,
	}

	// Adjust based on workload data
	if modelSize, ok := data["model_size"].(string); ok {
		switch modelSize {
		case "small":
			req.RequiredGPUs = 1
			req.MinMemoryMB = 4096
		case "medium":
			req.RequiredGPUs = 2
			req.MinMemoryMB = 8192
		case "large":
			req.RequiredGPUs = 4
			req.MinMemoryMB = 16384
		case "xlarge":
			req.RequiredGPUs = 8
			req.MinMemoryMB = 32768
		}
	}

	if batchSize, ok := data["batch_size"].(float64); ok {
		if batchSize > 64 {
			req.RequiredGPUs = 2
		}
		if batchSize > 128 {
			req.RequiredGPUs = 4
		}
	}

	if multiGPU, ok := data["multi_gpu"].(bool); ok && multiGPU {
		req.RequiredGPUs = 2
	}

	return req, nil
}

// analyzeInferenceWorkload analyzes inference workload requirements
func (w *WorkloadAnalyzer) analyzeInferenceWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    4096, // 4GB minimum
		Priority:       7,    // Higher priority for inference
		MaxUtilization: 80.0,
	}

	if modelSize, ok := data["model_size"].(string); ok {
		switch modelSize {
		case "small":
			req.MinMemoryMB = 2048
		case "medium":
			req.MinMemoryMB = 4096
		case "large":
			req.MinMemoryMB = 8192
		case "xlarge":
			req.MinMemoryMB = 16384
		}
	}

	if concurrent, ok := data["concurrent_requests"].(float64); ok {
		if concurrent > 10 {
			req.RequiredGPUs = 2
		}
		if concurrent > 20 {
			req.RequiredGPUs = 4
		}
	}

	return req, nil
}

// analyzeEmbeddingWorkload analyzes embedding generation workload requirements
func (w *WorkloadAnalyzer) analyzeEmbeddingWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    2048,
		Priority:       6,
		MaxUtilization: 85.0,
	}

	if batchSize, ok := data["batch_size"].(float64); ok {
		if batchSize > 32 {
			req.RequiredGPUs = 2
		}
	}

	return req, nil
}

// analyzeOCRWorkload analyzes OCR workload requirements
func (w *WorkloadAnalyzer) analyzeOCRWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    4096,
		Priority:       6,
		MaxUtilization: 80.0,
	}

	if imageCount, ok := data["image_count"].(float64); ok {
		if imageCount > 100 {
			req.RequiredGPUs = 2
		}
	}

	return req, nil
}

// analyzeGraphProcessingWorkload analyzes graph processing workload requirements
func (w *WorkloadAnalyzer) analyzeGraphProcessingWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    4096,
		Priority:       5,
		MaxUtilization: 75.0,
	}

	// Graph processing typically doesn't need multiple GPUs
	// but may need more memory for large graphs
	if nodeCount, ok := data["node_count"].(float64); ok {
		if nodeCount > 1000000 {
			req.MinMemoryMB = 8192
		}
	}

	return req, nil
}

// analyzeGenericWorkload analyzes generic workload requirements
func (w *WorkloadAnalyzer) analyzeGenericWorkload(data map[string]interface{}) (*WorkloadRequirements, error) {
	req := &WorkloadRequirements{
		RequiredGPUs:   1,
		MinMemoryMB:    4096,
		Priority:       5,
		MaxUtilization: 80.0,
	}

	// Try to extract requirements from data
	if requiredGPUs, ok := data["required_gpus"].(float64); ok {
		req.RequiredGPUs = int(requiredGPUs)
	}

	if minMemory, ok := data["min_memory_mb"].(float64); ok {
		req.MinMemoryMB = int64(minMemory)
	}

	if priority, ok := data["priority"].(float64); ok {
		req.Priority = int(priority)
	}

	return req, nil
}

// QueryUnifiedWorkflow queries the unified workflow for workload context
func (w *WorkloadAnalyzer) QueryUnifiedWorkflow(workflowID string) (map[string]interface{}, error) {
	if w.graphServiceURL == "" {
		return nil, fmt.Errorf("graph service URL not configured")
	}

	url := fmt.Sprintf("%s/workflow/%s", w.graphServiceURL, workflowID)
	resp, err := w.httpClient.Get(url)
	if err != nil {
		return nil, fmt.Errorf("failed to query unified workflow: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unified workflow returned status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

