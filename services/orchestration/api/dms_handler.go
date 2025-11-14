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

// DMSHandler provides HTTP handlers for DMS document processing.
type DMSHandler struct {
	pipeline     *agents.DMSPipeline
	jobProcessor *agents.DMSJobProcessor
	logger       *log.Logger
}

// GetJobProcessor returns the job processor.
func (h *DMSHandler) GetJobProcessor() *agents.DMSJobProcessor {
	return h.jobProcessor
}

// NewDMSHandler creates a new document handler (migrated from DMS to Extract service).
func NewDMSHandler(logger *log.Logger) (*DMSHandler, error) {
	// Load configuration from environment
	// ExtractURL is now the primary service (replaces DMS)
	extractURL := getEnvOrDefault("EXTRACT_URL", "")
	if extractURL == "" {
		// Fallback to DMS_URL for backward compatibility
		extractURL = getEnvOrDefault("DMS_URL", "http://localhost:8083")
	}
	
	config := agents.DMSPipelineConfig{
		ExtractURL:          extractURL, // Primary service
		DMSURL:              extractURL, // Backward compatibility
		DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
		DeepSeekOCRAPIKey:   os.Getenv("DEEPSEEK_OCR_API_KEY"),
		DeepResearchURL:     getEnvOrDefault("DEEP_RESEARCH_URL", "http://localhost:8085"),
		UnifiedWorkflowURL:  getEnvOrDefault("UNIFIED_WORKFLOW_URL", "http://graph-service:8081"),
		CatalogURL:          getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:         getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:          getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:           getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		Logger:              logger,
	}

	pipeline, err := agents.NewDMSPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create DMS pipeline: %w", err)
	}

	// Create job processor for async processing
	tracker := pipeline.GetRequestTracker()
	jobProcessor := agents.NewDMSJobProcessor(tracker, pipeline, logger)

	return &DMSHandler{
		pipeline:     pipeline,
		jobProcessor: jobProcessor,
		logger:       logger,
	}, nil
}

// HandleProcessDocuments handles POST /api/dms/process.
func (h *DMSHandler) HandleProcessDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DocumentID  string                   `json:"document_id,omitempty"`
		DocumentIDs []string                  `json:"document_ids,omitempty"`
		Async       bool                      `json:"async,omitempty"`
		WebhookURL  string                    `json:"webhook_url,omitempty"`
		Config      map[string]interface{}    `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Build query map
	query := make(map[string]interface{})
	if req.DocumentID != "" {
		query["document_id"] = req.DocumentID
	} else if len(req.DocumentIDs) > 0 {
		// Convert []string to []interface{}
		docIDs := make([]interface{}, len(req.DocumentIDs))
		for i, id := range req.DocumentIDs {
			docIDs[i] = id
		}
		query["document_ids"] = docIDs
	} else {
		http.Error(w, "document_id or document_ids is required", http.StatusBadRequest)
		return
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
			"status_url":  fmt.Sprintf("/api/dms/status/%s", requestID),
			"results_url": fmt.Sprintf("/api/dms/results/%s", requestID),
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
		})
		return
	}

	// Build enhanced response
	response := map[string]interface{}{
		"status":     string(processingRequest.Status),
		"request_id": requestID,
	}

	// Add statistics if available
	if processingRequest.Statistics != nil {
		response["statistics"] = map[string]interface{}{
			"documents_processed": processingRequest.Statistics.DocumentsProcessed,
			"documents_succeeded": processingRequest.Statistics.DocumentsSucceeded,
			"documents_failed":    processingRequest.Statistics.DocumentsFailed,
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
		"current_step":    processingRequest.CurrentStep,
		"completed_steps": processingRequest.CompletedSteps,
		"total_steps":     processingRequest.TotalSteps,
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
			"total_relationships":   processingRequest.Intelligence.TotalRelationships,
			"total_patterns":         processingRequest.Intelligence.TotalPatterns,
			"knowledge_graph_nodes":  processingRequest.Intelligence.KnowledgeGraphNodes,
			"knowledge_graph_edges":  processingRequest.Intelligence.KnowledgeGraphEdges,
			"workflow_processed":     processingRequest.Intelligence.WorkflowProcessed,
			"summary":               processingRequest.Intelligence.Summary,
		}
	}

	// Add status endpoint link
	response["status_url"] = fmt.Sprintf("/api/dms/status/%s", requestID)
	response["results_url"] = fmt.Sprintf("/api/dms/results/%s", requestID)
	response["intelligence_url"] = fmt.Sprintf("/api/dms/results/%s/intelligence", requestID)

	writeJSON(w, http.StatusOK, response)
}

// HandleGetStatus handles GET /api/dms/status/{request_id}.
func (h *DMSHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/status/"):]
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

// HandleGetResults handles GET /api/dms/results/{request_id}.
func (h *DMSHandler) HandleGetResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/results/"):]
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
	response["intelligence_url"] = fmt.Sprintf("/api/dms/results/%s/intelligence", requestID)

	writeJSON(w, http.StatusOK, response)
}

// HandleGetIntelligence handles GET /api/dms/results/{request_id}/intelligence.
func (h *DMSHandler) HandleGetIntelligence(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/results/"):]
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
			"id":           doc.ID,
			"title":        doc.Title,
			"intelligence": doc.Intelligence,
		}
		response["documents"] = append(response["documents"].([]map[string]interface{}), docIntelligence)
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleCancelJob handles DELETE /api/dms/jobs/{request_id}.
func (h *DMSHandler) HandleCancelJob(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/jobs/"):]
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

// HandleExportResults handles GET /api/dms/results/{request_id}/export.
func (h *DMSHandler) HandleExportResults(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/results/"):]
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
func (h *DMSHandler) exportJSON(w http.ResponseWriter, request *agents.ProcessingRequest) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=dms_results_%s.json", request.RequestID))
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
func (h *DMSHandler) exportCSV(w http.ResponseWriter, request *agents.ProcessingRequest) {
	w.Header().Set("Content-Type", "text/csv")
	w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=dms_results_%s.csv", request.RequestID))
	w.WriteHeader(http.StatusOK)

	// Write CSV header
	fmt.Fprintf(w, "Document ID,Title,Status,Processed At,Catalog ID,Training Task ID,LocalAI ID,Search ID,Error\n")

	// Write document rows
	for _, doc := range request.Documents {
		fmt.Fprintf(w, "%s,%s,%s,%s,%s,%s,%s,%s,%s\n",
			escapeCSV(doc.ID),
			escapeCSV(doc.Title),
			doc.Status,
			doc.ProcessedAt,
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

// HandleBatchProcess handles POST /api/dms/batch.
func (h *DMSHandler) HandleBatchProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DocumentIDs []string               `json:"document_ids"`
		Async       bool                   `json:"async,omitempty"`
		WebhookURL  string                 `json:"webhook_url,omitempty"`
		Config      map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.DocumentIDs) == 0 {
		http.Error(w, "document_ids array is required", http.StatusBadRequest)
		return
	}

	if len(req.DocumentIDs) > 50 {
		http.Error(w, "maximum 50 documents per batch", http.StatusBadRequest)
		return
	}

	// Generate batch ID
	batchID := agents.GenerateRequestID()
	requestIDs := make([]string, 0, len(req.DocumentIDs))

	// Process each document
	for _, docID := range req.DocumentIDs {
		requestID := agents.GenerateRequestID()
		requestIDs = append(requestIDs, requestID)

		query := map[string]interface{}{
			"document_id": docID,
		}

		// Merge additional config
		if req.Config != nil {
			for k, v := range req.Config {
				query[k] = v
			}
		}

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
		"batch_id":      batchID,
		"total_documents": len(req.DocumentIDs),
		"request_ids":   requestIDs,
		"status":        "processing",
		"status_url":    fmt.Sprintf("/api/dms/batch/%s", batchID),
	})
}

// HandleGetHistory handles GET /api/dms/history.
func (h *DMSHandler) HandleGetHistory(w http.ResponseWriter, r *http.Request) {
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
	documentIDFilter := r.URL.Query().Get("document_id")

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
		// Document ID filter
		if documentIDFilter != "" {
			found := false
			for _, docID := range req.DocumentIDs {
				if docID == documentIDFilter {
					found = true
					break
				}
			}
			if !found {
				continue
			}
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
			"request_id":     req.RequestID,
			"query":          req.Query,
			"status":         string(req.Status),
			"created_at":     req.CreatedAt,
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

// HandleSearchQuery handles POST /api/dms/search.
func (h *DMSHandler) HandleSearchQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query     string                 `json:"query"`
		RequestID string                 `json:"request_id,omitempty"`
		TopK      int                    `json:"top_k,omitempty"`
		Filters   map[string]interface{} `json:"filters,omitempty"`
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

// HandleKnowledgeGraphQuery handles POST /api/dms/graph/{request_id}/query.
func (h *DMSHandler) HandleKnowledgeGraphQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from URL path
	requestID := r.URL.Path[len("/api/dms/graph/"):]
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

// HandleDomainQuery handles GET /api/dms/domains/{domain}/documents.
func (h *DMSHandler) HandleDomainQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract domain from URL path
	domain := r.URL.Path[len("/api/dms/domains/"):]
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

// HandleCatalogSearch handles POST /api/dms/catalog/search.
func (h *DMSHandler) HandleCatalogSearch(w http.ResponseWriter, r *http.Request) {
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

	// Set default source to "dms"
	if req.Source == "" {
		req.Source = "dms"
	}

	// Query catalog semantic search
	results, err := h.pipeline.QueryCatalogSemantic(r.Context(), req.Query, req.ObjectClass, req.Property, req.Source, req.Filters)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]interface{}{
			"error": err.Error(),
		})
		return
	}

	writeJSON(w, http.StatusOK, results)
}

// HandleGetDocument handles GET /api/dms/documents/{document_id}.
func (h *DMSHandler) HandleGetDocument(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract document ID from URL path
	documentID := r.URL.Path[len("/api/dms/documents/"):]
	if documentID == "" {
		http.Error(w, "document_id is required", http.StatusBadRequest)
		return
	}

	// Query DMS connector to get document
	query := map[string]interface{}{
		"document_id": documentID,
	}

	// Get DMS connector
	connector := h.pipeline.GetDMSConnector()

	// Connect to DMS
	if err := connector.Connect(r.Context(), query); err != nil {
		http.Error(w, fmt.Sprintf("Failed to connect to DMS: %v", err), http.StatusInternalServerError)
		return
	}
	defer connector.Close()

	// Fetch document
	documents, err := connector.ExtractData(r.Context(), query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to fetch document: %v", err), http.StatusInternalServerError)
		return
	}

	if len(documents) == 0 {
		http.Error(w, "Document not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, documents[0])
}

