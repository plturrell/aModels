package agents

import (
	"fmt"
	"log"
	"strings"
	"sync"
	"time"
)

// RequestTracker tracks processing requests and their results.
type RequestTracker struct {
	requests map[string]*ProcessingRequest
	mu       sync.RWMutex
	logger   *log.Logger
}

// ProcessingRequest represents a processing request and its results.
type ProcessingRequest struct {
	RequestID      string                 `json:"request_id"`
	Query          string                 `json:"query"`
	Status         RequestStatus          `json:"status"`
	CreatedAt      time.Time              `json:"created_at"`
	StartedAt      *time.Time             `json:"started_at,omitempty"`
	CompletedAt    *time.Time             `json:"completed_at,omitempty"`
	ProcessingTime *time.Duration         `json:"processing_time_ms,omitempty"`
	
	// Statistics
	Statistics *ProcessingStatistics `json:"statistics,omitempty"`
	
	// Documents
	DocumentIDs []string               `json:"document_ids,omitempty"`
	Documents   []ProcessedDocument    `json:"documents,omitempty"`
	
	// Progress
	CurrentStep    string   `json:"current_step,omitempty"`
	CompletedSteps []string `json:"completed_steps,omitempty"`
	TotalSteps     int      `json:"total_steps,omitempty"`
	ProgressPercent float64 `json:"progress_percent,omitempty"`
	EstimatedTimeRemaining *time.Duration `json:"estimated_time_remaining_ms,omitempty"`
	
	// Results
	Results *ProcessingResults `json:"results,omitempty"`
	
	// Errors and warnings
	Errors   []ProcessingError `json:"errors,omitempty"`
	Warnings []string           `json:"warnings,omitempty"`
	
	// Intelligence (aggregated across all documents)
	Intelligence *RequestIntelligence `json:"intelligence,omitempty"`
}

// RequestIntelligence contains aggregated intelligence data for a request.
type RequestIntelligence struct {
	Domains            []string               `json:"domains,omitempty"`
	TotalRelationships int                    `json:"total_relationships,omitempty"`
	TotalPatterns      int                    `json:"total_patterns,omitempty"`
	KnowledgeGraphNodes int                   `json:"knowledge_graph_nodes,omitempty"`
	KnowledgeGraphEdges int                   `json:"knowledge_graph_edges,omitempty"`
	WorkflowProcessed  bool                   `json:"workflow_processed,omitempty"`
	Summary            map[string]interface{} `json:"summary,omitempty"`
}

// RequestStatus represents the status of a processing request.
type RequestStatus string

const (
	RequestStatusPending   RequestStatus = "pending"
	RequestStatusProcessing RequestStatus = "processing"
	RequestStatusCompleted  RequestStatus = "completed"
	RequestStatusFailed     RequestStatus = "failed"
	RequestStatusPartial    RequestStatus = "partial" // Some documents failed
)

// ProcessingStatistics contains processing statistics.
type ProcessingStatistics struct {
	DocumentsProcessed int `json:"documents_processed"`
	DocumentsSucceeded int `json:"documents_succeeded"`
	DocumentsFailed    int `json:"documents_failed"`
	StepsCompleted    int `json:"steps_completed"`
}

// ProcessedDocument represents a processed document.
type ProcessedDocument struct {
	ID              string                 `json:"id"`
	Title           string                 `json:"title,omitempty"`
	Status          string                 `json:"status"` // "success", "failed"
	ProcessedAt     time.Time              `json:"processed_at"`
	CatalogID       string                 `json:"catalog_id,omitempty"`
	TrainingTaskID  string                 `json:"training_task_id,omitempty"`
	LocalAIID       string                 `json:"localai_id,omitempty"`
	SearchID        string                 `json:"search_id,omitempty"`
	Error           string                 `json:"error,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	Intelligence    *DocumentIntelligence  `json:"intelligence,omitempty"`
}

// DocumentIntelligence contains intelligence data for a document.
type DocumentIntelligence struct {
	Domain            string                 `json:"domain,omitempty"`
	DomainConfidence  float64                `json:"domain_confidence,omitempty"`
	KnowledgeGraph    map[string]interface{} `json:"knowledge_graph,omitempty"`
	WorkflowResults   map[string]interface{} `json:"workflow_results,omitempty"`
	Relationships     []Relationship         `json:"relationships,omitempty"`
	LearnedPatterns   []Pattern              `json:"learned_patterns,omitempty"`
	CatalogPatterns   map[string]interface{} `json:"catalog_patterns,omitempty"`
	TrainingPatterns  map[string]interface{} `json:"training_patterns,omitempty"`
	DomainPatterns    map[string]interface{} `json:"domain_patterns,omitempty"`
	SearchPatterns    map[string]interface{} `json:"search_patterns,omitempty"`
	MetadataEnrichment map[string]interface{} `json:"metadata_enrichment,omitempty"`
}

// Relationship represents a discovered relationship between documents.
type Relationship struct {
	Type        string                 `json:"type"`
	TargetID    string                 `json:"target_id"`
	TargetTitle string                 `json:"target_title,omitempty"`
	Strength    float64                `json:"strength,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// Pattern represents a learned pattern.
type Pattern struct {
	Type        string                 `json:"type"`
	Description string                 `json:"description,omitempty"`
	Confidence  float64                `json:"confidence,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ProcessingResults contains links to results in various services.
type ProcessingResults struct {
	CatalogURL   string `json:"catalog_url,omitempty"`
	SearchURL    string `json:"search_url,omitempty"`
	TrainingURL  string `json:"training_url,omitempty"`
	LocalAIURL   string `json:"localai_url,omitempty"`
	ExportURL    string `json:"export_url,omitempty"`
}

// ProcessingError represents an error during processing.
type ProcessingError struct {
	Step          string    `json:"step"`
	Message       string    `json:"message"`
	Timestamp     time.Time `json:"timestamp"`
	DocumentID    string    `json:"document_id,omitempty"`
	ErrorCode     string    `json:"error_code,omitempty"`
	RecoverySteps []string  `json:"recovery_steps,omitempty"`
	Retryable     bool      `json:"retryable,omitempty"`
}

// NewRequestTracker creates a new request tracker.
func NewRequestTracker(logger *log.Logger) *RequestTracker {
	return &RequestTracker{
		requests: make(map[string]*ProcessingRequest),
		logger:   logger,
	}
}

// CreateRequest creates a new processing request.
func (rt *RequestTracker) CreateRequest(requestID, query string) *ProcessingRequest {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	now := time.Now()
	request := &ProcessingRequest{
		RequestID:      requestID,
		Query:          query,
		Status:         RequestStatusPending,
		CreatedAt:      now,
		DocumentIDs:    []string{},
		Documents:      []ProcessedDocument{},
		CompletedSteps: []string{},
		Errors:         []ProcessingError{},
		Warnings:       []string{},
		Statistics: &ProcessingStatistics{},
		TotalSteps:     10, // Default: connect, extract, unified_workflow, deep_research, ocr, catalog, training, localai, search, learning
	}

	rt.requests[requestID] = request
	return request
}

// GetRequest retrieves a processing request by ID.
func (rt *RequestTracker) GetRequest(requestID string) (*ProcessingRequest, bool) {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	request, exists := rt.requests[requestID]
	return request, exists
}

// GetAllRequests retrieves all processing requests.
func (rt *RequestTracker) GetAllRequests() []*ProcessingRequest {
	rt.mu.RLock()
	defer rt.mu.RUnlock()

	requests := make([]*ProcessingRequest, 0, len(rt.requests))
	for _, req := range rt.requests {
		requests = append(requests, req)
	}
	return requests
}

// UpdateStatus updates the status of a request.
func (rt *RequestTracker) UpdateStatus(requestID string, status RequestStatus) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		request.Status = status
		now := time.Now()
		switch status {
		case RequestStatusProcessing:
			if request.StartedAt == nil {
				request.StartedAt = &now
			}
		case RequestStatusCompleted, RequestStatusFailed, RequestStatusPartial:
			if request.CompletedAt == nil {
				request.CompletedAt = &now
				if request.StartedAt != nil {
					duration := now.Sub(*request.StartedAt)
					request.ProcessingTime = &duration
				}
			}
		}
	}
}

// UpdateStep updates the current processing step.
func (rt *RequestTracker) UpdateStep(requestID, step string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		request.CurrentStep = step
		// Add to completed steps if not already there
		found := false
		for _, s := range request.CompletedSteps {
			if s == step {
				found = true
				break
			}
		}
		if !found {
			request.CompletedSteps = append(request.CompletedSteps, step)
			if request.Statistics != nil {
				request.Statistics.StepsCompleted = len(request.CompletedSteps)
			}
		}
		
		// Calculate progress percentage
		if request.TotalSteps > 0 {
			request.ProgressPercent = float64(len(request.CompletedSteps)) / float64(request.TotalSteps) * 100.0
		}
		
		// Estimate time remaining (simple linear estimation)
		if request.StartedAt != nil && request.Statistics != nil && request.Statistics.StepsCompleted > 0 {
			elapsed := time.Since(*request.StartedAt)
			avgTimePerStep := elapsed / time.Duration(request.Statistics.StepsCompleted)
			remainingSteps := request.TotalSteps - request.Statistics.StepsCompleted
			estimatedRemaining := avgTimePerStep * time.Duration(remainingSteps)
			request.EstimatedTimeRemaining = &estimatedRemaining
		}
	}
}

// AddDocument adds a processed document to the request.
func (rt *RequestTracker) AddDocument(requestID string, doc ProcessedDocument) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		request.Documents = append(request.Documents, doc)
		request.DocumentIDs = append(request.DocumentIDs, doc.ID)
		if request.Statistics != nil {
			request.Statistics.DocumentsProcessed++
			if doc.Status == "success" {
				request.Statistics.DocumentsSucceeded++
			} else {
				request.Statistics.DocumentsFailed++
			}
		}
		
		// Aggregate intelligence data
		rt.aggregateIntelligence(request, doc)
	}
}

// aggregateIntelligence aggregates intelligence data from documents into request-level intelligence.
func (rt *RequestTracker) aggregateIntelligence(request *ProcessingRequest, doc ProcessedDocument) {
	if doc.Intelligence == nil {
		return
	}
	
	if request.Intelligence == nil {
		request.Intelligence = &RequestIntelligence{
			Domains: make([]string, 0),
			Summary: make(map[string]interface{}),
		}
	}
	
	// Aggregate domains
	if doc.Intelligence.Domain != "" {
		// Check if domain already in list
		found := false
		for _, d := range request.Intelligence.Domains {
			if d == doc.Intelligence.Domain {
				found = true
				break
			}
		}
		if !found {
			request.Intelligence.Domains = append(request.Intelligence.Domains, doc.Intelligence.Domain)
		}
	}
	
	// Aggregate relationships
	if len(doc.Intelligence.Relationships) > 0 {
		request.Intelligence.TotalRelationships += len(doc.Intelligence.Relationships)
	}
	
	// Aggregate patterns
	if len(doc.Intelligence.LearnedPatterns) > 0 {
		request.Intelligence.TotalPatterns += len(doc.Intelligence.LearnedPatterns)
	}
	
	// Aggregate knowledge graph
	if doc.Intelligence.KnowledgeGraph != nil {
		if nodes, ok := doc.Intelligence.KnowledgeGraph["nodes"].([]interface{}); ok {
			request.Intelligence.KnowledgeGraphNodes += len(nodes)
		}
		if edges, ok := doc.Intelligence.KnowledgeGraph["edges"].([]interface{}); ok {
			request.Intelligence.KnowledgeGraphEdges += len(edges)
		}
	}
	
	// Track workflow processing
	if doc.Intelligence.WorkflowResults != nil {
		request.Intelligence.WorkflowProcessed = true
	}
}

// SetDocumentIntelligence sets intelligence data for a specific document.
func (rt *RequestTracker) SetDocumentIntelligence(requestID, documentID string, intelligence *DocumentIntelligence) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		for i := range request.Documents {
			if request.Documents[i].ID == documentID {
				request.Documents[i].Intelligence = intelligence
				// Re-aggregate intelligence
				rt.aggregateIntelligence(request, request.Documents[i])
				break
			}
		}
	}
}

// AddError adds an error to the request.
func (rt *RequestTracker) AddError(requestID string, step, message, documentID string) {
	rt.AddErrorWithDetails(requestID, step, message, documentID, "", nil, false)
}

// AddErrorWithDetails adds an error with enhanced details to the request.
func (rt *RequestTracker) AddErrorWithDetails(requestID string, step, message, documentID, errorCode string, recoverySteps []string, retryable bool) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		error := ProcessingError{
			Step:          step,
			Message:       message,
			Timestamp:     time.Now(),
			DocumentID:    documentID,
			ErrorCode:     errorCode,
			RecoverySteps: recoverySteps,
			Retryable:     retryable,
		}
		
		// Auto-detect error type and add recovery steps if not provided
		if len(recoverySteps) == 0 {
			error.RecoverySteps = rt.generateRecoverySteps(step, message, errorCode)
		}
		
		// Auto-detect if retryable
		if !retryable {
			error.Retryable = rt.isRetryableError(message, errorCode)
		}
		
		request.Errors = append(request.Errors, error)
	}
}

// generateRecoverySteps generates recovery steps based on error context.
func (rt *RequestTracker) generateRecoverySteps(step, message, errorCode string) []string {
	steps := []string{}
	
	// Network errors
	if strings.Contains(message, "connection") || strings.Contains(message, "timeout") || strings.Contains(message, "network") {
		steps = append(steps, "Check network connectivity")
		steps = append(steps, "Verify service URLs are accessible")
		steps = append(steps, "Retry the request")
	}
	
	// Authentication errors
	if strings.Contains(message, "unauthorized") || strings.Contains(message, "authentication") || strings.Contains(message, "api key") {
		steps = append(steps, "Verify API key is correct")
		steps = append(steps, "Check API key permissions")
		steps = append(steps, "Regenerate API key if needed")
	}
	
	// Rate limiting
	if strings.Contains(message, "rate limit") || strings.Contains(message, "too many requests") {
		steps = append(steps, "Wait before retrying")
		steps = append(steps, "Reduce request frequency")
		steps = append(steps, "Check rate limit quotas")
	}
	
	// Validation errors
	if strings.Contains(message, "invalid") || strings.Contains(message, "validation") {
		steps = append(steps, "Review request parameters")
		steps = append(steps, "Check required fields are present")
		steps = append(steps, "Validate data format")
	}
	
	// Service unavailable
	if strings.Contains(message, "service unavailable") || strings.Contains(message, "503") {
		steps = append(steps, "Check service status")
		steps = append(steps, "Retry after a short delay")
		steps = append(steps, "Contact support if issue persists")
	}
	
	// Default recovery steps
	if len(steps) == 0 {
		steps = append(steps, "Review error message for details")
		steps = append(steps, "Check service logs")
		steps = append(steps, "Retry the operation")
	}
	
	return steps
}

// isRetryableError determines if an error is retryable.
func (rt *RequestTracker) isRetryableError(message, errorCode string) bool {
	retryablePatterns := []string{
		"timeout",
		"connection",
		"network",
		"rate limit",
		"503",
		"502",
		"500",
		"temporary",
		"unavailable",
	}
	
	messageLower := strings.ToLower(message)
	for _, pattern := range retryablePatterns {
		if strings.Contains(messageLower, pattern) {
			return true
		}
	}
	
	return false
}

// AddWarning adds a warning to the request.
func (rt *RequestTracker) AddWarning(requestID, warning string) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		request.Warnings = append(request.Warnings, warning)
	}
}

// SetResults sets the results links for a request.
func (rt *RequestTracker) SetResults(requestID string, results *ProcessingResults) {
	rt.mu.Lock()
	defer rt.mu.Unlock()

	if request, exists := rt.requests[requestID]; exists {
		request.Results = results
	}
}

// GenerateRequestID generates a unique request ID.
func GenerateRequestID() string {
	return fmt.Sprintf("req_%d", time.Now().UnixNano())
}

