package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/plturrell/aModels/services/catalog/workflows"
)

// DataProductHandler provides handlers for complete data products.
type DataProductHandler struct {
	unifiedWorkflow *workflows.UnifiedWorkflowIntegration
	logger          *log.Logger
}

// NewDataProductHandler creates a new data product handler.
func NewDataProductHandler(unifiedWorkflow *workflows.UnifiedWorkflowIntegration, logger *log.Logger) *DataProductHandler {
	return &DataProductHandler{
		unifiedWorkflow: unifiedWorkflow,
		logger:          logger,
	}
}

// HandleBuildDataProduct handles POST /catalog/data-products/build.
// This is the "thin slice" endpoint - builds one complete data product for a customer.
func (h *DataProductHandler) HandleBuildDataProduct(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Topic        string `json:"topic"`
		CustomerNeed string `json:"customer_need"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Topic == "" || req.CustomerNeed == "" {
		http.Error(w, "topic and customer_need are required", http.StatusBadRequest)
		return
	}

	// Build complete data product
	product, err := h.unifiedWorkflow.BuildCompleteDataProduct(r.Context(), req.Topic, req.CustomerNeed)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to build data product: %v", err), http.StatusInternalServerError)
		return
	}

	// Return complete product
	writeJSON(w, http.StatusOK, map[string]any{
		"data_product": map[string]any{
			"identifier":      product.DataElement.Identifier,
			"name":            product.DataElement.Name,
			"definition":      product.DataElement.Definition,
			"quality_score":   product.QualityMetrics.QualityScore,
			"quality_level":   product.QualityMetrics.ValidationStatus,
			"lifecycle_state": product.EnhancedElement.LifecycleState,
			"lineage":         product.Lineage,
			"usage_examples":  product.UsageExamples,
			"documentation_url": product.DocumentationURL,
			"sample_data_url":   product.SampleDataURL,
			"research_report":  product.ResearchReport,
		},
		"message": "Complete data product built successfully",
	})
}

// HandleGetDataProduct handles GET /catalog/data-products/{id}.
func (h *DataProductHandler) HandleGetDataProduct(w http.ResponseWriter, r *http.Request) {
	// Extract ID from path
	id := r.URL.Path[len("/catalog/data-products/"):]
	
	// In production, would fetch from registry
	// For now, return placeholder
	writeJSON(w, http.StatusOK, map[string]any{
		"id":     id,
		"status": "not_implemented",
		"message": "Data product retrieval coming soon",
	})
}

