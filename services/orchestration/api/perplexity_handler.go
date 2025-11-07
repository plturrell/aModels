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

// PerplexityHandler provides HTTP handlers for Perplexity document processing.
type PerplexityHandler struct {
	pipeline     *agents.PerplexityPipeline
	jobProcessor *agents.JobProcessor
	logger       *log.Logger
}

// NewPerplexityHandler creates a new Perplexity handler.
func NewPerplexityHandler(logger *log.Logger) (*PerplexityHandler, error) {
	// Load configuration from environment
	config := agents.PerplexityPipelineConfig{
		PerplexityAPIKey:    os.Getenv("PERPLEXITY_API_KEY"),
		PerplexityBaseURL:   getEnvOrDefault("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
		DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
		DeepSeekOCRAPIKey:   os.Getenv("DEEPSEEK_OCR_API_KEY"),
		DeepResearchURL:     getEnvOrDefault("DEEP_RESEARCH_URL", "http://localhost:8085"),
		UnifiedWorkflowURL:  getEnvOrDefault("UNIFIED_WORKFLOW_URL", "http://graph-service:8081"),
		CatalogURL:          getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:         getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:          getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:           getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		ExtractURL:          getEnvOrDefault("EXTRACT_URL", "http://extract:8081"),
		Logger:              logger,
	}

	pipeline, err := agents.NewPerplexityPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Perplexity pipeline: %w", err)
	}

	// Create job processor for async processing
	tracker := pipeline.GetRequestTracker()
	jobProcessor := agents.NewJobProcessor(tracker, pipeline, logger)

	return &PerplexityHandler{
		pipeline:     pipeline,
		jobProcessor: jobProcessor,
		logger:       logger,
	}, nil
}

// HandleProcessDocuments handles POST /api/perplexity/process.
func (h *PerplexityHandler) HandleProcessDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query         string                 `json:"query"`
		Model         string                 `json:"model,omitempty"`
		Limit         int                    `json:"limit,omitempty"`
		IncludeImages bool                   `json:"include_images,omitempty"`
		Async         bool                   `json:"async,omitempty"`
		WebhookURL    string                 `json:"webhook_url,omitempty"`
		Config        map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Build query map
	query := map[string]interface{}{
		"query":         req.Query,
		"include_images": req.IncludeImages,
	}

	if req.Model != "" {
		query["model"] = req.Model
	}

	if req.Limit > 0 {
		query["limit"] = req.Limit
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
				"query":      req.Query,
			})
			return
		}

		// Return immediately with job info
		writeJSON(w, http.StatusAccepted, map[string]interface{}{
			"status":     "queued",
			"request_id": requestID,
			"query":      req.Query,
			"message":    "Job submitted for async processing",
			"status_url": fmt.Sprintf("/api/perplexity/status/%s", requestID),
			"results_url": fmt.Sprintf("/api/perplexity/results/%s", requestID),
		})
		return
	}

	// Process documents synchronously with tracking
	processingRequest, err := h.pipeline.ProcessDocumentsWithTracking(r.Context(), requestID, query)
	if err != nil {
		// Return error but include request ID for tracking
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"status":     "failed",
			"request_id": requestID,
			"error":      err.Error(),
			"query":      req.Query,
		})
		return
	}

	// Build enhanced response
	response := map[string]interface{}{
		"status":    string(processingRequest.Status),
		"request_id": requestID,
		"query":     req.Query,
	}

	// Add statistics if available
	if processingRequest.Statistics != nil {
		response["statistics"] = map[string]interface{}{
			"documents_processed": processingRequest.Statistics.DocumentsProcessed,
			"documents_succeeded":  processingRequest.Statistics.DocumentsSucceeded,
			"documents_failed":     processingRequest.Statistics.DocumentsFailed,
			"steps_completed":     processingRequest.Statistics.StepsCompleted,
		}
	}

	// Add processing time
	if processingRequest.ProcessingTime != nil {
		response["processing_time_ms"] = processingRequest.ProcessingTime.Milliseconds()
	}

	// Add document IDs
	if len(processingRequest.DocumentIDs) > 0 {
		response["document_ids"] = processingRequest.DocumentIDs
		response["document_count"] = len(processingRequest.DocumentIDs)
	}

	// Add results links
	if processingRequest.Results != nil {
		response["results"] = processingRequest.Results
	}

	// Add progress information
	progress := map[string]interface{}{
		"current_step":     processingRequest.CurrentStep,
		"completed_steps": processingRequest.CompletedSteps,
		"total_steps":      processingRequest.TotalSteps,
	}
	if processingRequest.ProgressPercent > 0 {
		progress["progress_percent"] = processingRequest.ProgressPercent
	}
	if processingRequest.EstimatedTimeRemaining != nil {
		progress["estimated_time_remaining_ms"] = processingRequest.EstimatedTimeRemaining.Milliseconds()
	}
	response["progress"] = progress

	// Add errors and warnings
	if len(processingRequest.Errors) > 0 {
		response["errors"] = processingRequest.Errors
	}
	if len(processingRequest.Warnings) > 0 {
		response["warnings"] = processingRequest.Warnings
	}

	// Add intelligence summary if available
	if processingRequest.Intelligence != nil {
		response["intelligence"] = map[string]interface{}{
			"domains":                processingRequest.Intelligence.Domains,
			"total_relationships":    processingRequest.Intelligence.TotalRelationships,
			"total_patterns":         processingRequest.Intelligence.TotalPatterns,
			"knowledge_graph_nodes":  processingRequest.Intelligence.KnowledgeGraphNodes,
			"knowledge_graph_edges":  processingRequest.Intelligence.KnowledgeGraphEdges,
			"workflow_processed":     processingRequest.Intelligence.WorkflowProcessed,
			"summary":                processingRequest.Intelligence.Summary,
		}
	}

	// Add status endpoint link
	response["status_url"] = fmt.Sprintf("/api/perplexity/status/%s", requestID)
	response["results_url"] = fmt.Sprintf("/api/perplexity/results/%s", requestID)
	response["intelligence_url"] = fmt.Sprintf("/api/perplexity/results/%s/intelligence", requestID)

	writeJSON(w, http.StatusOK, response)
}

// HandleProcessWithIngestion handles POST /api/perplexity/process-with-ingestion.
// This endpoint uses the data ingestion agent pattern for consistency with other sources.
func (h *PerplexityHandler) HandleProcessWithIngestion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query  string                 `json:"query"`
		Config map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Build config for ingestion agent
	config := map[string]interface{}{
		"query": req.Query,
	}

	// Merge additional config
	if req.Config != nil {
		for k, v := range req.Config {
			config[k] = v
		}
	}

	// Use the pipeline's ProcessDocuments method
	err := h.pipeline.ProcessDocuments(r.Context(), config)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to process documents: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "completed",
		"query":   req.Query,
		"message": "Documents ingested and processed successfully",
	})
}

// getEnvOrDefault returns the environment variable value or a default.
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// HandleGetStatus handles GET /api/perplexity/status/{request_id}.
func (h *PerplexityHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	// Assuming URL pattern: /api/perplexity/status/{request_id}
	requestID := r.URL.Path[len("/api/perplexity/status/"):]
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	tracker := h.pipeline.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, request)
}

// HandleGetResults handles GET /api/perplexity/results/{request_id}.
func (h *PerplexityHandler) HandleGetResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/results/"):]
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	tracker := h.pipeline.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Build results response
	response := map[string]interface{}{
		"request_id": request.RequestID,
		"query":      request.Query,
		"status":     string(request.Status),
		"statistics": request.Statistics,
		"documents":  request.Documents,
		"results":    request.Results,
	}

	if request.ProcessingTime != nil {
		response["processing_time_ms"] = request.ProcessingTime.Milliseconds()
	}

	// Add intelligence summary if available
	if request.Intelligence != nil {
		response["intelligence"] = request.Intelligence
	}
	
	// Add intelligence URL
	response["intelligence_url"] = fmt.Sprintf("/api/perplexity/results/%s/intelligence", requestID)

	writeJSON(w, http.StatusOK, response)
}

// HandleGetLearningReport handles GET /api/perplexity/learning/report.
func (h *PerplexityHandler) HandleGetLearningReport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	report := h.pipeline.GetLearningReport()
	if report == nil {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"message": "No learning report available",
		})
		return
	}

	writeJSON(w, http.StatusOK, report)
}

// HandleGetIntelligence handles GET /api/perplexity/results/{request_id}/intelligence.
func (h *PerplexityHandler) HandleGetIntelligence(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/results/"):]
	if idx := strings.Index(requestID, "/intelligence"); idx != -1 {
		requestID = requestID[:idx]
	}
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	tracker := h.pipeline.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Build intelligence response
	response := map[string]interface{}{
		"request_id":   request.RequestID,
		"query":        request.Query,
		"status":       string(request.Status),
		"intelligence": request.Intelligence,
		"documents":    make([]map[string]interface{}, 0),
	}

	// Include document-level intelligence
	for _, doc := range request.Documents {
		docIntelligence := map[string]interface{}{
			"id":          doc.ID,
			"title":       doc.Title,
			"intelligence": doc.Intelligence,
		}
		response["documents"] = append(response["documents"].([]map[string]interface{}), docIntelligence)
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleCancelJob handles DELETE /api/perplexity/jobs/{request_id}.
func (h *PerplexityHandler) HandleCancelJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/jobs/"):]
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	if err := h.jobProcessor.CancelJob(requestID); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "cancelled",
		"request_id": requestID,
		"message":    "Job cancelled successfully",
	})
}

// HandleExportResults handles GET /api/perplexity/results/{request_id}/export.
func (h *PerplexityHandler) HandleExportResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/results/"):]
	if idx := strings.Index(requestID, "/export"); idx != -1 {
		requestID = requestID[:idx]
	}
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	// Get format (json or csv)
	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}

	tracker := h.pipeline.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	switch format {
	case "csv":
		h.exportCSV(w, request)
	case "json":
		fallthrough
	default:
		h.exportJSON(w, request)
	}
}

// exportJSON exports results as JSON.
func (h *PerplexityHandler) exportJSON(w http.ResponseWriter, request *agents.ProcessingRequest) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=perplexity_results_%s.json", request.RequestID))
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": request.RequestID,
		"query":      request.Query,
		"status":     string(request.Status),
		"created_at": request.CreatedAt,
		"statistics": request.Statistics,
		"documents":  request.Documents,
		"results":    request.Results,
		"errors":     request.Errors,
		"warnings":   request.Warnings,
	})
}

// exportCSV exports results as CSV.
func (h *PerplexityHandler) exportCSV(w http.ResponseWriter, request *agents.ProcessingRequest) {
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=perplexity_results_%s.csv", request.RequestID))
	w.WriteHeader(http.StatusOK)

	// Write CSV header
	fmt.Fprintf(w, "Document ID,Title,Status,Processed At,Catalog ID,Training Task ID,LocalAI ID,Search ID,Error\n")

	// Write document rows
	for _, doc := range request.Documents {
		fmt.Fprintf(w, "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
			escapeCSV(doc.ID),
			escapeCSV(doc.Title),
			doc.Status,
			doc.ProcessedAt.Format(time.RFC3339),
			escapeCSV(doc.CatalogID),
			escapeCSV(doc.TrainingTaskID),
			escapeCSV(doc.LocalAIID),
			escapeCSV(doc.SearchID),
			escapeCSV(doc.Error),
		)
	}
}

// escapeCSV escapes CSV special characters.
func escapeCSV(s string) string {
	if strings.Contains(s, ",") || strings.Contains(s, "\"") || strings.Contains(s, "\n") {
		return fmt.Sprintf("\"%s\"", strings.ReplaceAll(s, "\"", "\"\""))
	}
	return s
}

// HandleBatchProcess handles POST /api/perplexity/batch.
func (h *PerplexityHandler) HandleBatchProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Queries    []map[string]interface{} `json:"queries"`
		Async      bool                      `json:"async,omitempty"`
		WebhookURL string                    `json:"webhook_url,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "queries array is required", http.StatusBadRequest)
		return
	}

	if len(req.Queries) > 50 {
		http.Error(w, "maximum 50 queries per batch", http.StatusBadRequest)
		return
	}

	// Generate batch ID
	batchID := agents.GenerateRequestID()
	requestIDs := make([]string, 0, len(req.Queries))

	// Process each query
	for _, query := range req.Queries {
		requestID := agents.GenerateRequestID()
		requestIDs = append(requestIDs, requestID)

		if req.Async {
			// Submit async job
			if err := h.jobProcessor.SubmitJob(requestID, query, req.WebhookURL); err != nil {
				if h.logger != nil {
					h.logger.Printf("Failed to submit batch job %s: %v", requestID, err)
				}
			}
		} else {
			// Process synchronously (in background to avoid blocking)
			go func(id string, q map[string]interface{}) {
				_, _ = h.pipeline.ProcessDocumentsWithTracking(r.Context(), id, q)
			}(requestID, query)
		}
	}

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"batch_id":     batchID,
		"total_queries": len(req.Queries),
		"request_ids":  requestIDs,
		"status":       "processing",
		"status_url":   fmt.Sprintf("/api/perplexity/batch/%s", batchID),
	})
}

// HandleGetHistory handles GET /api/perplexity/history.
func (h *PerplexityHandler) HandleGetHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse query parameters
	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 && parsed <= 1000 {
			limit = parsed
		}
	}

	offset := 0
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil && parsed >= 0 {
			offset = parsed
		}
	}

	statusFilter := r.URL.Query().Get("status")
	queryFilter := r.URL.Query().Get("query")

	// Get all requests from tracker
	tracker := h.pipeline.GetRequestTracker()
	allRequests := tracker.GetAllRequests()

	// Filter requests
	filtered := make([]*agents.ProcessingRequest, 0)
	for _, req := range allRequests {
		// Status filter
		if statusFilter != "" && string(req.Status) != statusFilter {
			continue
		}
		// Query text filter
		if queryFilter != "" && !strings.Contains(strings.ToLower(req.Query), strings.ToLower(queryFilter)) {
			continue
		}
		filtered = append(filtered, req)
	}

	// Sort by created_at (newest first)
	sort.Slice(filtered, func(i, j int) bool {
		return filtered[i].CreatedAt.After(filtered[j].CreatedAt)
	})

	// Paginate
	total := len(filtered)
	start := offset
	end := offset + limit
	if end > total {
		end = total
	}
	if start > total {
		start = total
	}

	paginated := filtered[start:end]

	// Build history items
	historyItems := make([]map[string]interface{}, 0, len(paginated))
	for _, req := range paginated {
		item := map[string]interface{}{
			"request_id":    req.RequestID,
			"query":         req.Query,
			"status":        string(req.Status),
			"created_at":    req.CreatedAt,
			"document_count": len(req.DocumentIDs),
		}
		if req.CompletedAt != nil {
			item["completed_at"] = req.CompletedAt
		}
		historyItems = append(historyItems, item)
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"requests": historyItems,
		"total":    total,
		"limit":    limit,
		"offset":   offset,
	})
}

// HandleSearchQuery handles POST /api/perplexity/search.
func (h *PerplexityHandler) HandleSearchQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query   string                 `json:"query"`
		RequestID string                `json:"request_id,omitempty"` // Optional: filter to specific request
		TopK    int                    `json:"top_k,omitempty"`
		Filters map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	if req.TopK <= 0 {
		req.TopK = 10
	}

	// Query search service
	results, err := h.pipeline.QuerySearch(r.Context(), req.Query, req.RequestID, req.TopK, req.Filters)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"query":   req.Query,
		"results": results,
		"count":   len(results),
	})
}

// HandleKnowledgeGraphQuery handles POST /api/perplexity/graph/{request_id}/query.
func (h *PerplexityHandler) HandleKnowledgeGraphQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/graph/"):]
	if idx := strings.Index(requestID, "/query"); idx != -1 {
		requestID = requestID[:idx]
	}
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	var req struct {
		Query  string                 `json:"query"` // Cypher query
		Params map[string]interface{} `json:"params,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Query knowledge graph
	results, err := h.pipeline.QueryKnowledgeGraph(r.Context(), requestID, req.Query, req.Params)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": requestID,
		"query":      req.Query,
		"results":    results,
	})
}

// HandleDomainQuery handles GET /api/perplexity/domains/{domain}/documents.
func (h *PerplexityHandler) HandleDomainQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract domain from URL path
	domain := r.URL.Path[len("/api/perplexity/domains/"):]
	if idx := strings.Index(domain, "/documents"); idx != -1 {
		domain = domain[:idx]
	}
	if domain == "" {
		http.Error(w, "domain is required", http.StatusBadRequest)
		return
	}

	// Parse query parameters
	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 && parsed <= 100 {
			limit = parsed
		}
	}

	offset := 0
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil && parsed >= 0 {
			offset = parsed
		}
	}

	// Query domain documents
	documents, err := h.pipeline.QueryDomainDocuments(r.Context(), domain, limit, offset)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"domain":    domain,
		"documents": documents,
		"count":     len(documents),
		"limit":     limit,
		"offset":    offset,
	})
}

// HandleCatalogSearch handles POST /api/perplexity/catalog/search.
func (h *PerplexityHandler) HandleCatalogSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query       string                 `json:"query"`
		ObjectClass string                 `json:"object_class,omitempty"`
		Property    string                 `json:"property,omitempty"`
		Source      string                 `json:"source,omitempty"`
		Filters     map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Query catalog semantic search
	results, err := h.pipeline.QueryCatalogSemantic(r.Context(), req.Query, req.ObjectClass, req.Property, req.Source, req.Filters)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"query":   req.Query,
		"results": results,
	})
}

// HandleGetRelationships handles GET /api/perplexity/graph/{request_id}/relationships.
func (h *PerplexityHandler) HandleGetRelationships(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/perplexity/graph/"):]
	if idx := strings.Index(requestID, "/relationships"); idx != -1 {
		requestID = requestID[:idx]
	}
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	tracker := h.pipeline.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Collect all relationships from documents
	allRelationships := make([]agents.Relationship, 0)
	for _, doc := range request.Documents {
		if doc.Intelligence != nil && len(doc.Intelligence.Relationships) > 0 {
			allRelationships = append(allRelationships, doc.Intelligence.Relationships...)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id":  requestID,
		"query":       request.Query,
		"relationships": allRelationships,
		"count":       len(allRelationships),
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

