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

// RelationalHandler provides HTTP handlers for relational table processing.
type RelationalHandler struct {
	pipeline     *agents.RelationalPipeline
	jobProcessor *agents.RelationalJobProcessor
	logger       *log.Logger
}

// GetJobProcessor returns the job processor.
func (h *RelationalHandler) GetJobProcessor() *agents.RelationalJobProcessor {
	return h.jobProcessor
}

// NewRelationalHandler creates a new relational handler.
func NewRelationalHandler(logger *log.Logger) (*RelationalHandler, error) {
	// Load configuration from environment
	config := agents.RelationalPipelineConfig{
		DatabaseURL:        getEnvOrDefault("DATABASE_URL", ""),
		DatabaseType:       getEnvOrDefault("DATABASE_TYPE", "postgres"),
		UnifiedWorkflowURL: getEnvOrDefault("UNIFIED_WORKFLOW_URL", "http://graph-service:8081"),
		CatalogURL:         getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:        getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:         getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:          getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		ExtractURL:         getEnvOrDefault("EXTRACT_URL", "http://extract:8081"),
		Logger:             logger,
	}

	pipeline, err := agents.NewRelationalPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create relational pipeline: %w", err)
	}

	// Create job processor for async processing
	tracker := pipeline.GetRequestTracker()
	jobProcessor := agents.NewRelationalJobProcessor(tracker, pipeline, logger)

	return &RelationalHandler{
		pipeline:     pipeline,
		jobProcessor: jobProcessor,
		logger:       logger,
	}, nil
}

// HandleProcessTables handles POST /api/relational/process.
func (h *RelationalHandler) HandleProcessTables(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Table       string                   `json:"table,omitempty"`
		Tables      []string                 `json:"tables,omitempty"`
		Schema      string                   `json:"schema,omitempty"`
		DatabaseURL string                   `json:"database_url,omitempty"`
		DatabaseType string                  `json:"database_type,omitempty"`
		Async       bool                     `json:"async,omitempty"`
		WebhookURL  string                   `json:"webhook_url,omitempty"`
		Config      map[string]interface{}   `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Build query map
	query := make(map[string]interface{})
	if req.Table != "" {
		query["table"] = req.Table
	} else if len(req.Tables) > 0 {
		// Convert []string to []interface{}
		tables := make([]interface{}, len(req.Tables))
		for i, t := range req.Tables {
			tables[i] = t
		}
		query["tables"] = tables
	}

	if req.Schema != "" {
		query["schema"] = req.Schema
	}

	if req.DatabaseURL != "" {
		query["database_url"] = req.DatabaseURL
	}

	if req.DatabaseType != "" {
		query["database_type"] = req.DatabaseType
		query["db_type"] = req.DatabaseType
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
			"status_url":  fmt.Sprintf("/api/relational/status/%s", requestID),
			"results_url": fmt.Sprintf("/api/relational/results/%s", requestID),
		})
		return
	}

	// Process tables synchronously with tracking
	processingRequest, err := h.pipeline.ProcessTablesWithTracking(r.Context(), requestID, query)
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
		response["statistics"] = processingRequest.Statistics
	}

	// Add processing time
	if processingRequest.CompletedAt != nil && processingRequest.CreatedAt != nil {
		completed, _ := time.Parse(time.RFC3339, *processingRequest.CompletedAt)
		created, _ := time.Parse(time.RFC3339, *processingRequest.CreatedAt)
		processingTime := completed.Sub(created)
		response["processing_time_ms"] = processingTime.Milliseconds()
	}

	// Add links
	response["status_url"] = fmt.Sprintf("/api/relational/status/%s", requestID)
	response["results_url"] = fmt.Sprintf("/api/relational/results/%s", requestID)
	response["intelligence_url"] = fmt.Sprintf("/api/relational/results/%s/intelligence", requestID)

	writeJSON(w, http.StatusOK, response)
}

// HandleGetStatus handles GET /api/relational/status/{request_id}.
func (h *RelationalHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/status/")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	request, exists := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, request)
}

// HandleGetResults handles GET /api/relational/results/{request_id}.
func (h *RelationalHandler) HandleGetResults(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/results/")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	request, exists := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Get documents (tables)
	documents := h.pipeline.GetRequestTracker().GetDocuments(requestID)

	response := map[string]interface{}{
		"request_id": requestID,
		"query":      request.Query,
		"status":     string(request.Status),
		"statistics": request.Statistics,
		"documents":  documents,
		"results":    request.Results,
	}

	// Add intelligence summary if available
	if intelligence := h.pipeline.GetRequestTracker().GetRequestIntelligence(requestID); intelligence != nil {
		response["intelligence"] = map[string]interface{}{
			"domains":                intelligence.Domains,
			"total_relationships":    intelligence.TotalRelationships,
			"total_patterns":         intelligence.TotalPatterns,
			"knowledge_graph_nodes":  intelligence.KnowledgeGraphNodes,
			"knowledge_graph_edges": intelligence.KnowledgeGraphEdges,
		}
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleGetIntelligence handles GET /api/relational/results/{request_id}/intelligence.
func (h *RelationalHandler) HandleGetIntelligence(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/results/")
	requestID = strings.TrimSuffix(requestID, "/intelligence")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	request, exists := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	// Get intelligence data
	intelligence := h.pipeline.GetRequestTracker().GetRequestIntelligence(requestID)
	documents := h.pipeline.GetRequestTracker().GetDocuments(requestID)

	// Build document intelligence list
	docIntelligence := make([]map[string]interface{}, 0, len(documents))
	for _, doc := range documents {
		docIntel := h.pipeline.GetRequestTracker().GetDocumentIntelligence(requestID, doc.ID)
		if docIntel != nil {
			docIntelligence = append(docIntelligence, map[string]interface{}{
				"id":          doc.ID,
				"title":       doc.Metadata["title"],
				"intelligence": docIntel,
			})
		}
	}

	response := map[string]interface{}{
		"request_id": requestID,
		"query":      request.Query,
		"status":     string(request.Status),
		"intelligence": intelligence,
		"documents":   docIntelligence,
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleGetHistory handles GET /api/relational/history.
func (h *RelationalHandler) HandleGetHistory(w http.ResponseWriter, r *http.Request) {
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

	statusFilter := r.URL.Query().Get("status")
	tableFilter := r.URL.Query().Get("table")

	// Get all requests
	allRequests := h.pipeline.GetRequestTracker().GetAllRequests()

	// Filter by status if provided
	filtered := make([]*agents.ProcessingRequest, 0)
	for _, req := range allRequests {
		if statusFilter != "" && string(req.Status) != statusFilter {
			continue
		}
		if tableFilter != "" {
			// Check if query contains table filter
			queryStr := req.Query
			if !strings.Contains(strings.ToLower(queryStr), strings.ToLower(tableFilter)) {
				continue
			}
		}
		filtered = append(filtered, req)
	}

	// Sort by created_at descending
	sort.Slice(filtered, func(i, j int) bool {
		if filtered[i].CreatedAt == nil || filtered[j].CreatedAt == nil {
			return false
		}
		ti, _ := time.Parse(time.RFC3339, *filtered[i].CreatedAt)
		tj, _ := time.Parse(time.RFC3339, *filtered[j].CreatedAt)
		return ti.After(tj)
	})

	// Apply pagination
	total := len(filtered)
	start := offset
	end := offset + limit
	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	requests := filtered[start:end]

	// Build response
	responseRequests := make([]map[string]interface{}, len(requests))
	for i, req := range requests {
		responseRequests[i] = map[string]interface{}{
			"request_id":    req.RequestID,
			"query":         req.Query,
			"status":        string(req.Status),
			"created_at":    req.CreatedAt,
			"completed_at":  req.CompletedAt,
			"document_count": 0,
		}
		if req.Statistics != nil {
			responseRequests[i]["document_count"] = req.Statistics.DocumentsTotal
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"requests": responseRequests,
		"total":    total,
		"limit":    limit,
		"offset":   offset,
	})
}

// HandleSearchQuery handles POST /api/relational/search.
func (h *RelationalHandler) HandleSearchQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query     string `json:"query"`
		RequestID string `json:"request_id,omitempty"`
		TopK      int    `json:"top_k,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	if req.TopK == 0 {
		req.TopK = 10
	}

	results, err := h.pipeline.QuerySearch(r.Context(), req.Query, req.RequestID, req.TopK)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"query":   req.Query,
		"results": results,
		"count":   len(results),
	})
}

// HandleExportResults handles GET /api/relational/results/{request_id}/export.
func (h *RelationalHandler) HandleExportResults(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/results/")
	requestID = strings.TrimSuffix(requestID, "/export")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	format := r.URL.Query().Get("format")
	if format == "" {
		format = "json"
	}

	request, exists := h.pipeline.GetRequestTracker().GetRequest(requestID)
	if !exists {
		http.Error(w, "Request not found", http.StatusNotFound)
		return
	}

	documents := h.pipeline.GetRequestTracker().GetDocuments(requestID)

	switch format {
	case "json":
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"request_id": requestID,
			"documents":  documents,
			"statistics": request.Statistics,
		})
	case "csv":
		// Simple CSV export
		w.Header().Set("Content-Type", "text/csv")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=relational_%s.csv", requestID))
		w.WriteHeader(http.StatusOK)
		// CSV export implementation would go here
		fmt.Fprintf(w, "request_id,table_id,status,processed_at\n")
		for _, doc := range documents {
			fmt.Fprintf(w, "%s,%s,%s,%s\n", requestID, doc.ID, doc.Status, doc.ProcessedAt)
		}
	default:
		http.Error(w, "Unsupported format. Use 'json' or 'csv'", http.StatusBadRequest)
	}
}

// HandleBatchProcess handles POST /api/relational/batch.
func (h *RelationalHandler) HandleBatchProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Tables     []string               `json:"tables"`
		Schema     string                 `json:"schema,omitempty"`
		DatabaseURL string                `json:"database_url,omitempty"`
		DatabaseType string               `json:"database_type,omitempty"`
		Async      bool                   `json:"async,omitempty"`
		WebhookURL string                `json:"webhook_url,omitempty"`
		Config     map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Tables) == 0 {
		http.Error(w, "tables array is required", http.StatusBadRequest)
		return
	}

	// Generate batch ID
	batchID := agents.GenerateRequestID()

	// Process each table
	requestIDs := make([]string, 0, len(req.Tables))
	for _, table := range req.Tables {
		query := map[string]interface{}{
			"table": table,
		}
		if req.Schema != "" {
			query["schema"] = req.Schema
		}
		if req.DatabaseURL != "" {
			query["database_url"] = req.DatabaseURL
		}
		if req.DatabaseType != "" {
			query["database_type"] = req.DatabaseType
		}
		if req.Config != nil {
			for k, v := range req.Config {
				query[k] = v
			}
		}

		requestID := agents.GenerateRequestID()
		requestIDs = append(requestIDs, requestID)

		if req.Async {
			h.jobProcessor.SubmitJob(requestID, query, req.WebhookURL)
		} else {
			_, _ = h.pipeline.ProcessTablesWithTracking(r.Context(), requestID, query)
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"batch_id":        batchID,
		"total_tables":    len(req.Tables),
		"request_ids":    requestIDs,
		"status":          "processing",
		"status_url":      fmt.Sprintf("/api/relational/batch/%s", batchID),
	})
}

// HandleCancelJob handles DELETE /api/relational/jobs/{request_id}.
func (h *RelationalHandler) HandleCancelJob(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/jobs/")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	if err := h.jobProcessor.CancelJob(requestID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":     "cancelled",
		"request_id": requestID,
		"message":    "Job cancelled successfully",
	})
}

// HandleKnowledgeGraphQuery handles POST /api/relational/graph/{request_id}/query.
func (h *RelationalHandler) HandleKnowledgeGraphQuery(w http.ResponseWriter, r *http.Request) {
	requestID := strings.TrimPrefix(r.URL.Path, "/api/relational/graph/")
	requestID = strings.TrimSuffix(requestID, "/query")
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	var req struct {
		Query string `json:"query"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	result, err := h.pipeline.QueryKnowledgeGraph(r.Context(), requestID, req.Query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Query failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": requestID,
		"query":      req.Query,
		"results":    result,
	})
}

// HandleDomainQuery handles GET /api/relational/domains/{domain}/tables.
func (h *RelationalHandler) HandleDomainQuery(w http.ResponseWriter, r *http.Request) {
	domain := strings.TrimPrefix(r.URL.Path, "/api/relational/domains/")
	domain = strings.TrimSuffix(domain, "/tables")
	if domain == "" {
		http.Error(w, "domain is required", http.StatusBadRequest)
		return
	}

	limit := 50
	if l := r.URL.Query().Get("limit"); l != "" {
		if parsed, err := strconv.Atoi(l); err == nil && parsed > 0 {
			limit = parsed
		}
	}

	offset := 0
	if o := r.URL.Query().Get("offset"); o != "" {
		if parsed, err := strconv.Atoi(o); err == nil && parsed >= 0 {
			offset = parsed
		}
	}

	results, err := h.pipeline.QueryDomainTables(r.Context(), domain, limit, offset)
	if err != nil {
		http.Error(w, fmt.Sprintf("Query failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"domain":    domain,
		"tables":    results,
		"count":     len(results),
		"limit":     limit,
		"offset":    offset,
	})
}

// HandleCatalogSearch handles POST /api/relational/catalog/search.
func (h *RelationalHandler) HandleCatalogSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query       string `json:"query"`
		ObjectClass string `json:"object_class,omitempty"`
		Property    string `json:"property,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	results, err := h.pipeline.QueryCatalogSemantic(r.Context(), req.Query, req.ObjectClass, req.Property)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"results": results,
		"count":   len(results),
	})
}

