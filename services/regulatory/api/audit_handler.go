package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/graph"
	"github.com/plturrell/aModels/services/orchestration/agents"
	"github.com/plturrell/aModels/services/regulatory"
)

// AuditHandler handles BCBS239 audit requests via HTTP.
type AuditHandler struct {
	pipeline   *regulatory.BCBS239AuditPipeline
	logger     *log.Logger
	
	// Track running audits
	runningAudits map[string]*regulatory.AuditResult
}

// NewAuditHandler creates a new audit handler.
func NewAuditHandler(
	driver neo4j.DriverWithContext,
	localAIURL string,
	gnnURL string,
	gooseURL string,
	deepAgentsURL string,
	logger *log.Logger,
) *AuditHandler {
	// Setup compliance stack
	graphClient := graph.NewNeo4jGraphClient(driver, logger)
	bcbs239GraphClient := regulatory.NewBCBS239GraphClient(driver, graphClient, logger)
	
	localAIClient := agents.NewLocalAIClient(localAIURL, nil, logger)
	
	// Create reasoning agent with all models
	reasoningAgent := regulatory.NewComplianceReasoningAgent(
		localAIClient,
		bcbs239GraphClient,
		logger,
		"gemma-2b-q4_k_m.gguf",
	)
	
	// Wire model adapters
	if gnnURL != "" {
		reasoningAgent.WithGNNAdapter(regulatory.NewGNNAdapter(gnnURL, logger))
	}
	if gooseURL != "" {
		reasoningAgent.WithGooseAdapter(regulatory.NewGooseAdapter(gooseURL, logger))
	}
	if deepAgentsURL != "" {
		reasoningAgent.WithDeepResearchAdapter(regulatory.NewDeepResearchAdapter(deepAgentsURL, logger))
	}
	
	pipeline := regulatory.NewBCBS239AuditPipeline(
		reasoningAgent,
		bcbs239GraphClient,
		logger,
	)
	
	return &AuditHandler{
		pipeline:      pipeline,
		logger:        logger,
		runningAudits: make(map[string]*regulatory.AuditResult),
	}
}

// ServeHTTP implements http.Handler for the audit API.
func (h *AuditHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	
	switch r.URL.Path {
	case "/api/compliance/audit/start":
		h.handleStartAudit(w, r)
	case "/api/compliance/audit/status":
		h.handleAuditStatus(w, r)
	case "/api/compliance/audit/list":
		h.handleListAudits(w, r)
	default:
		http.NotFound(w, r)
	}
}

// handleStartAudit starts a new BCBS239 audit.
func (h *AuditHandler) handleStartAudit(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req regulatory.AuditRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.sendError(w, "Invalid request body", http.StatusBadRequest)
		return
	}
	
	// Generate audit ID if not provided
	if req.AuditID == "" {
		req.AuditID = generateAuditID()
	}
	
	// Start audit in background
	go h.runAuditAsync(req)
	
	h.sendJSON(w, map[string]interface{}{
		"audit_id": req.AuditID,
		"status":   "started",
		"message":  "Audit started successfully",
	}, http.StatusAccepted)
}

// handleAuditStatus retrieves the status of a running audit.
func (h *AuditHandler) handleAuditStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	auditID := r.URL.Query().Get("audit_id")
	if auditID == "" {
		h.sendError(w, "audit_id parameter required", http.StatusBadRequest)
		return
	}
	
	result, exists := h.pipeline.GetAuditStatus(auditID)
	if !exists {
		h.sendError(w, "Audit not found", http.StatusNotFound)
		return
	}
	
	h.sendJSON(w, result, http.StatusOK)
}

// handleListAudits lists all audits.
func (h *AuditHandler) handleListAudits(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	// Return summary of all audits
	audits := []map[string]interface{}{}
	
	// In production, this would query from persistent storage
	h.sendJSON(w, map[string]interface{}{
		"audits": audits,
		"total":  len(audits),
	}, http.StatusOK)
}

// runAuditAsync runs an audit in the background.
func (h *AuditHandler) runAuditAsync(req regulatory.AuditRequest) {
	ctx := context.Background()
	
	if h.logger != nil {
		h.logger.Printf("Starting audit: %s", req.AuditID)
	}
	
	result, err := h.pipeline.RunAudit(ctx, req)
	if err != nil {
		if h.logger != nil {
			h.logger.Printf("Audit %s failed: %v", req.AuditID, err)
		}
		return
	}
	
	if h.logger != nil {
		h.logger.Printf("Audit %s completed: %s (%.2f%%)", 
			req.AuditID, result.ComplianceStatus, result.OverallScore*100)
	}
}

// Helper methods

func (h *AuditHandler) sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		if h.logger != nil {
			h.logger.Printf("Error encoding JSON: %v", err)
		}
	}
}

func (h *AuditHandler) sendError(w http.ResponseWriter, message string, status int) {
	h.sendJSON(w, map[string]string{"error": message}, status)
}

func generateAuditID() string {
	return "audit-" + time.Now().Format("20060102-150405")
}
