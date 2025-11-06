package api

import (
	// "context" // unused
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"

	"github.com/plturrell/aModels/services/extract/regulatory"
)

// RegulatoryHandler provides HTTP handlers for regulatory spec extraction.
type RegulatoryHandler struct {
	regSystem *regulatory.RegulatorySpecSystem
	logger    *log.Logger
}

// NewRegulatoryHandler creates a new regulatory handler.
func NewRegulatoryHandler(regSystem *regulatory.RegulatorySpecSystem, logger *log.Logger) *RegulatoryHandler {
	return &RegulatoryHandler{
		regSystem: regSystem,
		logger:    logger,
	}
}

// HandleExtractMAS610 handles POST /api/regulatory/extract/mas610.
func (h *RegulatoryHandler) HandleExtractMAS610(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DocumentContent string `json:"document_content"`
		DocumentSource  string `json:"document_source"`
		DocumentVersion string `json:"document_version"`
		User            string `json:"user"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.DocumentContent == "" {
		http.Error(w, "document_content is required", http.StatusBadRequest)
		return
	}

	if req.User == "" {
		req.User = "system"
	}

	// Extract MAS 610 spec
	result, err := h.regSystem.ExtractMAS610(r.Context(), req.DocumentContent, req.DocumentSource, req.DocumentVersion, req.User)
	if err != nil {
		http.Error(w, fmt.Sprintf("Extraction failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"extraction_id": result.ExtractionID,
		"spec": map[string]interface{}{
			"id":              result.Spec.ID,
			"regulatory_type": result.Spec.RegulatoryType,
			"version":         result.Spec.Version,
			"report_name":     result.Spec.ReportStructure.ReportName,
			"field_count":     result.Spec.ReportStructure.TotalFields,
		},
		"confidence":      result.Confidence,
		"processing_time_ms": result.ProcessingTime.Milliseconds(),
		"message":         "MAS 610 specification extracted successfully",
	})
}

// HandleExtractBCBS239 handles POST /api/regulatory/extract/bcbs239.
func (h *RegulatoryHandler) HandleExtractBCBS239(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DocumentContent string `json:"document_content"`
		DocumentSource  string `json:"document_source"`
		DocumentVersion string `json:"document_version"`
		User            string `json:"user"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.DocumentContent == "" {
		http.Error(w, "document_content is required", http.StatusBadRequest)
		return
	}

	if req.User == "" {
		req.User = "system"
	}

	// Extract BCBS 239 spec
	result, err := h.regSystem.ExtractBCBS239(r.Context(), req.DocumentContent, req.DocumentSource, req.DocumentVersion, req.User)
	if err != nil {
		http.Error(w, fmt.Sprintf("Extraction failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"extraction_id": result.ExtractionID,
		"spec": map[string]interface{}{
			"id":              result.Spec.ID,
			"regulatory_type": result.Spec.RegulatoryType,
			"version":         result.Spec.Version,
			"report_name":     result.Spec.ReportStructure.ReportName,
			"field_count":     result.Spec.ReportStructure.TotalFields,
		},
		"confidence":      result.Confidence,
		"processing_time_ms": result.ProcessingTime.Milliseconds(),
		"message":         "BCBS 239 specification extracted successfully",
	})
}

// HandleValidateSpec handles POST /api/regulatory/validate.
func (h *RegulatoryHandler) HandleValidateSpec(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Spec *regulatory.RegulatorySpec `json:"spec"`
		User string                     `json:"user"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Spec == nil {
		http.Error(w, "spec is required", http.StatusBadRequest)
		return
	}

	if req.User == "" {
		req.User = "system"
	}

	// Validate and save
	validationResult, err := h.regSystem.ValidateAndSave(r.Context(), req.Spec, req.User)
	if err != nil {
		http.Error(w, fmt.Sprintf("Validation failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"valid":        validationResult.Valid,
		"errors":       validationResult.Errors,
		"warnings":     validationResult.Warnings,
		"completeness": validationResult.Completeness,
		"field_count":  validationResult.FieldCount,
		"message":      "Validation completed",
	})
}

// HandleListSchemas handles GET /api/regulatory/schemas.
func (h *RegulatoryHandler) HandleListSchemas(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	_ = r.URL.Query().Get("regulatory_type") // regulatoryType unused
	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit == 0 {
		limit = 100
	}

	// List schemas (schema repo access would need to be exposed)
	// For now, return empty list
	schemas := []*regulatory.RegulatorySpec{}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"schemas": schemas,
		"count":   len(schemas),
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

