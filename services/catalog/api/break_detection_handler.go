package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/plturrell/aModels/services/catalog/breakdetection"
)

// BreakDetectionHandler handles break detection API requests
type BreakDetectionHandler struct {
	breakDetectionService *breakdetection.BreakDetectionService
	baselineManager       *breakdetection.BaselineManager
	logger                *log.Logger
}

// NewBreakDetectionHandler creates a new break detection handler
func NewBreakDetectionHandler(
	breakDetectionService *breakdetection.BreakDetectionService,
	baselineManager *breakdetection.BaselineManager,
	logger *log.Logger,
) *BreakDetectionHandler {
	return &BreakDetectionHandler{
		breakDetectionService: breakDetectionService,
		baselineManager:       baselineManager,
		logger:                logger,
	}
}

// HandleDetectBreaks handles POST /catalog/break-detection/detect
func (h *BreakDetectionHandler) HandleDetectBreaks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req breakdetection.DetectionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.SystemName == "" {
		http.Error(w, "system_name is required", http.StatusBadRequest)
		return
	}
	if req.BaselineID == "" {
		http.Error(w, "baseline_id is required", http.StatusBadRequest)
		return
	}
	if req.DetectionType == "" {
		http.Error(w, "detection_type is required", http.StatusBadRequest)
		return
	}

	// Perform break detection
	result, err := h.breakDetectionService.DetectBreaks(r.Context(), &req)
	if err != nil {
		h.logger.Printf("Break detection failed: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"result":  result,
		"message": fmt.Sprintf("Break detection completed: %d breaks detected", result.TotalBreaksDetected),
	})
}

// HandleCreateBaseline handles POST /catalog/break-detection/baselines
func (h *BreakDetectionHandler) HandleCreateBaseline(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	var req breakdetection.BaselineRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.SystemName == "" {
		http.Error(w, "system_name is required", http.StatusBadRequest)
		return
	}
	if req.Version == "" {
		http.Error(w, "version is required", http.StatusBadRequest)
		return
	}
	if req.SnapshotType == "" {
		req.SnapshotType = "full" // Default
	}
	if req.SnapshotData == nil {
		http.Error(w, "snapshot_data is required", http.StatusBadRequest)
		return
	}

	// Create baseline
	baseline, err := h.baselineManager.CreateBaseline(ctx, &req)
	if err != nil {
		h.logger.Printf("Failed to create baseline: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusCreated, map[string]interface{}{
		"baseline": baseline,
		"message":  "Baseline created successfully",
	})
}

// HandleGetBaseline handles GET /catalog/break-detection/baselines/{baseline_id}
func (h *BreakDetectionHandler) HandleGetBaseline(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Extract baseline ID from path
	path := r.URL.Path
	baselineID := strings.TrimPrefix(path, "/catalog/break-detection/baselines/")
	if baselineID == "" {
		http.Error(w, "baseline_id is required in path", http.StatusBadRequest)
		return
	}

	baseline, err := h.baselineManager.GetBaseline(ctx, baselineID)
	if err != nil {
		http.Error(w, fmt.Sprintf("Baseline not found: %v", err), http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"baseline": baseline,
	})
}

// HandleListBaselines handles GET /catalog/break-detection/baselines?system={system_name}
func (h *BreakDetectionHandler) HandleListBaselines(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	systemName := r.URL.Query().Get("system")
	if systemName == "" {
		http.Error(w, "system query parameter is required", http.StatusBadRequest)
		return
	}

	limit := 100 // Default limit
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if parsed, err := fmt.Sscanf(limitStr, "%d", &limit); err != nil || parsed != 1 {
			limit = 100
		}
	}

	baselines, err := h.baselineManager.ListBaselines(ctx, breakdetection.SystemName(systemName), limit)
	if err != nil {
		h.logger.Printf("Failed to list baselines: %v", err)
		http.Error(w, fmt.Sprintf("Failed to list baselines: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"baselines": baselines,
		"count":     len(baselines),
	})
}

// HandleListBreaks handles GET /catalog/break-detection/breaks?system={system_name}&status={status}
func (h *BreakDetectionHandler) HandleListBreaks(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	systemName := r.URL.Query().Get("system")
	if systemName == "" {
		http.Error(w, "system query parameter is required", http.StatusBadRequest)
		return
	}

	status := breakdetection.BreakStatusOpen
	if statusStr := r.URL.Query().Get("status"); statusStr != "" {
		status = breakdetection.BreakStatus(statusStr)
	}

	limit := 100
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if parsed, err := fmt.Sscanf(limitStr, "%d", &limit); err != nil || parsed != 1 {
			limit = 100
		}
	}

	breaks, err := h.breakDetectionService.ListBreaks(ctx, breakdetection.SystemName(systemName), limit, status)
	if err != nil {
		h.logger.Printf("Failed to list breaks: %v", err)
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"breaks": breaks,
		"count":  len(breaks),
	})
}

// HandleGetBreak handles GET /catalog/break-detection/breaks/{break_id}
func (h *BreakDetectionHandler) HandleGetBreak(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Extract break_id from URL path
	// Path format: /catalog/break-detection/breaks/{break_id}
	path := r.URL.Path
	pathParts := strings.Split(strings.Trim(path, "/"), "/")

	var breakID string
	for i, part := range pathParts {
		if part == "breaks" && i+1 < len(pathParts) {
			breakID = pathParts[i+1]
			break
		}
	}

	if breakID == "" {
		http.Error(w, "break_id is required", http.StatusBadRequest)
		return
	}

	// Get break from service
	breakRecord, err := h.breakDetectionService.GetBreak(ctx, breakID)
	if err != nil {
		if strings.Contains(err.Error(), "not found") {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}
		h.logger.Printf("Failed to get break %s: %v", breakID, err)
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"break": breakRecord,
	})
}
