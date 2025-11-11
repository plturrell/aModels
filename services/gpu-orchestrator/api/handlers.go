package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/gpu-orchestrator/gpu_orchestrator"
)

// Handlers provides HTTP handlers for the GPU orchestrator API
type Handlers struct {
	orchestrator *gpu_orchestrator.GPUOrchestrator
	logger       *log.Logger
}

// NewHandlers creates new API handlers
func NewHandlers(orchestrator *gpu_orchestrator.GPUOrchestrator, logger *log.Logger) *Handlers {
	return &Handlers{
		orchestrator: orchestrator,
		logger:       logger,
	}
}

// HandleHealthz handles health check requests
func (h *Handlers) HandleHealthz(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"service": "gpu-orchestrator",
		"timestamp": time.Now(),
	})
}

// AllocationRequest represents an HTTP request to allocate GPUs
type AllocationRequest struct {
	ServiceName   string                 `json:"service_name"`
	WorkloadType  string                 `json:"workload_type"`
	WorkloadData  map[string]interface{} `json:"workload_data,omitempty"`
	WebhookURL    string                 `json:"webhook_url,omitempty"` // Callback URL for async notifications
}

// HandleAllocateGPU handles GPU allocation requests
func (h *Handlers) HandleAllocateGPU(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AllocationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.ServiceName == "" {
		http.Error(w, "service_name is required", http.StatusBadRequest)
		return
	}

	if req.WorkloadType == "" {
		req.WorkloadType = "generic"
	}
	
	// Add webhook URL to workload data if provided
	if req.WebhookURL != "" {
		if req.WorkloadData == nil {
			req.WorkloadData = make(map[string]interface{})
		}
		req.WorkloadData["webhook_url"] = req.WebhookURL
	}

	allocation, err := h.orchestrator.AllocateGPUs(r.Context(), req.ServiceName, req.WorkloadType, req.WorkloadData)
	if err != nil {
		h.logger.Printf("Failed to allocate GPUs: %v", err)
		http.Error(w, fmt.Sprintf("Failed to allocate GPUs: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(allocation)
}

// ReleaseRequest represents an HTTP request to release GPUs
type ReleaseRequest struct {
	AllocationID string `json:"allocation_id,omitempty"`
	ServiceName  string `json:"service_name,omitempty"`
}

// HandleReleaseGPU handles GPU release requests
func (h *Handlers) HandleReleaseGPU(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ReleaseRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	var err error
	if req.AllocationID != "" {
		err = h.orchestrator.ReleaseGPUs(req.AllocationID)
	} else if req.ServiceName != "" {
		err = h.orchestrator.ReleaseGPUsByService(req.ServiceName)
	} else {
		http.Error(w, "allocation_id or service_name is required", http.StatusBadRequest)
		return
	}

	if err != nil {
		h.logger.Printf("Failed to release GPUs: %v", err)
		http.Error(w, fmt.Sprintf("Failed to release GPUs: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "released",
	})
}

// HandleGPUStatus handles GPU status requests
func (h *Handlers) HandleGPUStatus(w http.ResponseWriter, r *http.Request) {
	status, err := h.orchestrator.GetGPUStatus()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to get GPU status: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"gpus": status,
	})
}

// HandleListGPUs handles list GPUs requests
func (h *Handlers) HandleListGPUs(w http.ResponseWriter, r *http.Request) {
	status, err := h.orchestrator.GetGPUStatus()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list GPUs: %v", err), http.StatusInternalServerError)
		return
	}

	allocations := h.orchestrator.ListAllocations()
	queue := h.orchestrator.GetQueueStatus()

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]interface{}{
		"gpus":        status,
		"allocations": allocations,
		"queue":       queue,
	})
}

// HandleWorkloadAnalysis handles workload analysis requests
func (h *Handlers) HandleWorkloadAnalysis(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req AllocationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.WorkloadType == "" {
		req.WorkloadType = "generic"
	}

	requirements, err := h.orchestrator.AnalyzeWorkload(req.WorkloadType, req.WorkloadData)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to analyze workload: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(requirements)
}

