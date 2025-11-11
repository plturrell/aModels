package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// MurexHandler provides HTTP handlers for Murex trade processing.
type MurexHandler struct {
	pipeline     *agents.MurexPipeline
	jobProcessor *agents.MurexJobProcessor
	logger       *log.Logger
}

// GetJobProcessor returns the job processor.
func (h *MurexHandler) GetJobProcessor() *agents.MurexJobProcessor {
	return h.jobProcessor
}

// NewMurexHandler creates a new Murex handler.
func NewMurexHandler(logger *log.Logger) (*MurexHandler, error) {
	// Load configuration from environment
	config := agents.MurexPipelineConfig{
		BaseURL:            getEnvOrDefault("MUREX_BASE_URL", "https://api.murex.com"),
		APIKey:             os.Getenv("MUREX_API_KEY"),
		OpenAPISpecURL:     getEnvOrDefault("MUREX_OPENAPI_SPEC_URL", ""),
		UnifiedWorkflowURL: getEnvOrDefault("UNIFIED_WORKFLOW_URL", "http://graph-service:8081"),
		CatalogURL:         getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:        getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:         getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:          getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		SAPGLURL:           getEnvOrDefault("SAP_GL_URL", "http://sap-gl:8080"),
		Logger:             logger,
	}

	// API key is optional (may use other auth methods)
	if config.APIKey == "" {
		logger.Printf("Warning: MUREX_API_KEY not set, some operations may fail")
	}

	pipeline, err := agents.NewMurexPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Murex pipeline: %w", err)
	}

	// Create job processor for async processing
	tracker := pipeline.GetRequestTracker()
	jobProcessor := agents.NewMurexJobProcessor(tracker, pipeline, logger)

	return &MurexHandler{
		pipeline:     pipeline,
		jobProcessor: jobProcessor,
		logger:       logger,
	}, nil
}

// HandleProcessTrades handles POST /api/murex/process.
func (h *MurexHandler) HandleProcessTrades(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Table      string                 `json:"table,omitempty"`
		Filters    map[string]interface{} `json:"filters,omitempty"`
		Async      bool                   `json:"async,omitempty"`
		WebhookURL string                 `json:"webhook_url,omitempty"`
		Config     map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Build query map
	query := make(map[string]interface{})
	if req.Table != "" {
		query["table"] = req.Table
	} else {
		query["table"] = "trades" // Default to trades
	}

	if req.Filters != nil {
		query["filters"] = req.Filters
	}

	// Merge additional config
	if req.Config != nil {
		for k, v := range req.Config {
			query[k] = v
		}
	}

	// Generate request ID
	requestID := agents.GenerateRequestID()

	// Check if async processing is requested
	if req.Async {
		// Submit job for async processing
		if err := h.jobProcessor.SubmitJob(requestID, query, req.WebhookURL); err != nil {
			writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"status":     "failed",
				"request_id": requestID,
				"error":      err.Error(),
			})
			return
		}

		// Return immediately with job info
		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"status":      "queued",
			"request_id":  requestID,
			"message":     "Job submitted for async processing",
			"status_url":  fmt.Sprintf("/api/murex/status/%s", requestID),
			"results_url": fmt.Sprintf("/api/murex/results/%s", requestID),
		})
		return
	}

	// Process trades synchronously with tracking
	processingRequest, err := h.pipeline.ProcessTradesWithTracking(r.Context(), requestID, query)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"status":     "failed",
			"request_id": requestID,
			"error":      err.Error(),
		})
		return
	}

	// Return response
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":            processingRequest.Status,
		"request_id":        processingRequest.RequestID,
		"statistics":        processingRequest.Statistics,
		"processing_time_ms": processingRequest.ProcessingTimeMs,
		"status_url":        fmt.Sprintf("/api/murex/status/%s", requestID),
		"results_url":       fmt.Sprintf("/api/murex/results/%s", requestID),
		"intelligence_url":  fmt.Sprintf("/api/murex/results/%s/intelligence", requestID),
	})
}

// HandleGetStatus handles GET /api/murex/status/{request_id}.
func (h *MurexHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/murex/status/")
	requestID := strings.TrimSuffix(path, "/")

	if requestID == "" {
		http.Error(w, "Request ID is required", http.StatusBadRequest)
		return
	}

	request := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if request == nil {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, request)
}

// HandleGetResults handles GET /api/murex/results/{request_id}.
func (h *MurexHandler) HandleGetResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/murex/results/")
	requestID := strings.TrimSuffix(path, "/")

	if requestID == "" {
		http.Error(w, "Request ID is required", http.StatusBadRequest)
		return
	}

	request := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if request == nil {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	documents := h.pipeline.GetRequestTracker().GetDocuments(requestID)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": request.RequestID,
		"query":      request.Query,
		"status":     request.Status,
		"statistics": request.Statistics,
		"documents":  documents,
		"results": map[string]interface{}{
			"catalog_url": fmt.Sprintf("/api/catalog/documents?source=murex&request_id=%s", requestID),
			"search_url":  fmt.Sprintf("/api/search?query=murex&request_id=%s", requestID),
			"export_url":  fmt.Sprintf("/api/murex/results/%s/export", requestID),
		},
	})
}

// HandleGetIntelligence handles GET /api/murex/results/{request_id}/intelligence.
func (h *MurexHandler) HandleGetIntelligence(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := strings.TrimPrefix(r.URL.Path, "/api/murex/results/")
	path = strings.TrimSuffix(path, "/intelligence")
	requestID := strings.TrimSuffix(path, "/")

	if requestID == "" {
		http.Error(w, "Request ID is required", http.StatusBadRequest)
		return
	}

	request := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if request == nil {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	intelligence := h.pipeline.GetRequestTracker().GetIntelligence(requestID)
	documents := h.pipeline.GetRequestTracker().GetDocuments(requestID)

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id":  request.RequestID,
		"query":       request.Query,
		"status":      request.Status,
		"intelligence": intelligence,
		"documents":   documents,
	})
}

// HandleGetHistory handles GET /api/murex/history.
func (h *MurexHandler) HandleGetHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit <= 0 {
		limit = 50
	}
	if limit > 100 {
		limit = 100
	}

	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))
	if offset < 0 {
		offset = 0
	}

	statusFilter := r.URL.Query().Get("status")
	tableFilter := r.URL.Query().Get("table")

	requests := h.pipeline.GetRequestTracker().GetAllRequests()

	// Apply filters
	filtered := []*agents.ProcessingRequest{}
	for _, req := range requests {
		if statusFilter != "" && req.Status != statusFilter {
			continue
		}
		if tableFilter != "" && !strings.Contains(req.Query, tableFilter) {
			continue
		}
		filtered = append(filtered, req)
	}

	// Sort by created_at descending
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].CreatedAt.After(filtered[j].CreatedAt)
	})

	// Paginate
	start := offset
	end := offset + limit
	if start > len(filtered) {
		start = len(filtered)
	}
	if end > len(filtered) {
		end = len(filtered)
	}

	results := filtered[start:end]

	// Format response
	formatted := make([]map[string]interface{}, len(results))
	for i, req := range results {
		formatted[i] = map[string]interface{}{
			"request_id":    req.RequestID,
			"query":         req.Query,
			"status":        req.Status,
			"created_at":    req.CreatedAt.Format(time.RFC3339),
			"completed_at":  nil,
			"document_count": len(req.DocumentIDs),
		}
		if req.CompletedAt != nil {
			formatted[i]["completed_at"] = req.CompletedAt.Format(time.RFC3339)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"requests": formatted,
		"total":    len(filtered),
		"limit":    limit,
		"offset":   offset,
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// getEnvOrDefault gets an environment variable or returns a default value.
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

