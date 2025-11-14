package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"

	"github.com/plturrell/aModels/services/catalog/integration"
)

// HANAInboundHandler handles HANA inbound integration requests
type HANAInboundHandler struct {
	hanaIntegration *integration.HANAInboundIntegration
	logger          *log.Logger
}

// NewHANAInboundHandler creates a new HANA inbound handler
func NewHANAInboundHandler(
	hanaIntegration *integration.HANAInboundIntegration,
	logger *log.Logger,
) *HANAInboundHandler {
	return &HANAInboundHandler{
		hanaIntegration: hanaIntegration,
		logger:          logger,
	}
}

// HandleProcessHANATables handles POST /catalog/integration/hana/process
func (h *HANAInboundHandler) HandleProcessHANATables(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req integration.HANAInboundRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		h.logger.Printf("[HANA_INBOUND] Invalid request: %v", err)
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Schema == "" {
		http.Error(w, "schema is required", http.StatusBadRequest)
		return
	}
	if req.ProjectID == "" {
		req.ProjectID = "default"
	}

	// Store request in context for token forwarding
	ctx := r.Context()
	ctx = context.WithValue(ctx, "http_request", r)

	// Process HANA tables through the pipeline
	response, err := h.hanaIntegration.ProcessHANATables(ctx, req)
	if err != nil {
		h.logger.Printf("[HANA_INBOUND] Processing failed: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, response)
}

// HandleGetStatus handles GET /catalog/integration/hana/status/:request_id
func (h *HANAInboundHandler) HandleGetStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract request ID from path
	requestID := r.URL.Path[len("/catalog/integration/hana/status/"):]
	if requestID == "" {
		http.Error(w, "request_id is required", http.StatusBadRequest)
		return
	}

	// TODO: Implement status retrieval from storage
	// For now, return a placeholder response
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": requestID,
		"status":     "processing",
		"message":    "Status retrieval not yet implemented",
	})
}
