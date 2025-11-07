package api

import (
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

	// Process HANA tables through the pipeline
	response, err := h.hanaIntegration.ProcessHANATables(r.Context(), req)
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

// Helper function to write JSON response
func writeJSON(w http.ResponseWriter, statusCode int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		// Log error but can't change status code at this point
	}
}

