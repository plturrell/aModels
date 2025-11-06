package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/workflows"
)

// DataProductHandler provides handlers for complete data products.
type DataProductHandler struct {
	unifiedWorkflow *workflows.UnifiedWorkflowIntegration
	registry        *iso11179.MetadataRegistry
	versionManager  *workflows.VersionManager
	logger          *log.Logger
}

// NewDataProductHandler creates a new data product handler.
func NewDataProductHandler(unifiedWorkflow *workflows.UnifiedWorkflowIntegration, registry *iso11179.MetadataRegistry, versionManager *workflows.VersionManager, logger *log.Logger) *DataProductHandler {
	return &DataProductHandler{
		unifiedWorkflow: unifiedWorkflow,
		registry:        registry,
		versionManager: versionManager,
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
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract ID from path
	path := r.URL.Path[len("/catalog/data-products/"):]
	
	// Handle versioned requests: /catalog/data-products/{id}?version=v1.0.0
	// or /catalog/data-products/{id}/versions/{version}
	var productID string
	var version string
	
	if strings.Contains(path, "/versions/") {
		parts := strings.Split(path, "/versions/")
		productID = parts[0]
		if len(parts) > 1 {
			version = parts[1]
		}
	} else {
		productID = path
		version = r.URL.Query().Get("version")
	}

	ctx := r.Context()

	// Try to get from version manager first (if available)
	if h.versionManager != nil {
		var versionData *workflows.DataProductVersion
		var err error
		
		if version != "" {
			versionData, err = h.versionManager.GetVersion(ctx, productID, version)
		} else {
			versionData, err = h.versionManager.GetLatestVersion(ctx, productID)
		}
		
		if err == nil && versionData != nil {
			// Unmarshal product snapshot
			var product workflows.CompleteDataProduct
			if err := json.Unmarshal(versionData.ProductSnapshot, &product); err == nil {
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
						"version":           versionData.Version,
						"created_at":        versionData.CreatedAt,
						"created_by":        versionData.CreatedBy,
					},
					"message": "Data product retrieved successfully",
				})
				return
			}
		}
	}

	// Fallback to registry lookup
	if h.registry != nil {
		element, ok := h.registry.GetDataElement(productID)
		if ok {
			// Get enhanced element if available
			enhanced := iso11179.NewEnhancedDataElement(element)
			
			// Build response from registry data
			writeJSON(w, http.StatusOK, map[string]any{
				"data_product": map[string]any{
					"identifier":      element.Identifier,
					"name":            element.Name,
					"definition":      element.Definition,
					"lifecycle_state": enhanced.LifecycleState,
					"documentation_url": fmt.Sprintf("/catalog/data-elements/%s/docs", element.Identifier),
					"sample_data_url":   fmt.Sprintf("/catalog/data-elements/%s/sample", element.Identifier),
				},
				"message": "Data product retrieved from registry",
			})
			return
		}
	}

	// Not found
	http.Error(w, fmt.Sprintf("Data product not found: %s", productID), http.StatusNotFound)
}

// HandleGetSampleData handles GET /catalog/data-elements/{id}/sample.
// Returns sample data preview for a data product.
func (h *DataProductHandler) HandleGetSampleData(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract ID from path
	path := r.URL.Path
	var elementID string
	
	if strings.Contains(path, "/data-elements/") {
		parts := strings.Split(path, "/data-elements/")
		if len(parts) > 1 {
			elementID = strings.TrimSuffix(parts[1], "/sample")
		}
	} else if strings.Contains(path, "/data-products/") {
		parts := strings.Split(path, "/data-products/")
		if len(parts) > 1 {
			elementID = strings.TrimSuffix(parts[1], "/sample")
		}
	}

	if elementID == "" {
		http.Error(w, "Element ID required", http.StatusBadRequest)
		return
	}

	ctx := r.Context()

	// Try to get from version manager first
	if h.versionManager != nil {
		versionData, err := h.versionManager.GetLatestVersion(ctx, elementID)
		if err == nil && versionData != nil {
			var product workflows.CompleteDataProduct
			if err := json.Unmarshal(versionData.ProductSnapshot, &product); err == nil {
				sampleData := h.generateSampleData(&product)
				writeJSON(w, http.StatusOK, map[string]any{
					"element_id": elementID,
					"sample_data": sampleData,
					"message":     "Sample data generated from data product",
				})
				return
			}
		}
	}

	// Fallback to registry
	if h.registry != nil {
		element, ok := h.registry.GetDataElement(elementID)
		if ok {
			sampleData := h.generateSampleDataFromElement(element)
			writeJSON(w, http.StatusOK, map[string]any{
				"element_id": elementID,
				"sample_data": sampleData,
				"message":     "Sample data generated from registry",
			})
			return
		}
	}

	http.Error(w, fmt.Sprintf("Data element not found: %s", elementID), http.StatusNotFound)
}

// generateSampleData generates sample data from a complete data product.
func (h *DataProductHandler) generateSampleData(product *workflows.CompleteDataProduct) []map[string]any {
	samples := make([]map[string]any, 0, 5)
	
	if product.DataElement == nil {
		return samples
	}

	// Generate 5 sample records
	for i := 0; i < 5; i++ {
		sample := make(map[string]any)
		sample["id"] = fmt.Sprintf("sample_%d", i+1)
		sample["name"] = fmt.Sprintf("%s Sample %d", product.DataElement.Name, i+1)
		sample["description"] = product.DataElement.Definition
		
		// Add quality metrics if available
		if product.QualityMetrics != nil {
			sample["quality_score"] = product.QualityMetrics.QualityScore
			sample["freshness"] = product.QualityMetrics.FreshnessScore
			sample["completeness"] = product.QualityMetrics.CompletenessScore
		}
		
		// Add metadata fields
		if product.DataElement.Metadata != nil {
			for k, v := range product.DataElement.Metadata {
				sample[k] = v
			}
		}
		
		samples = append(samples, sample)
	}
	
	return samples
}

// generateSampleDataFromElement generates sample data from a data element.
func (h *DataProductHandler) generateSampleDataFromElement(element *iso11179.DataElement) []map[string]any {
	samples := make([]map[string]any, 0, 5)
	
	// Generate 5 sample records
	for i := 0; i < 5; i++ {
		sample := make(map[string]any)
		sample["id"] = fmt.Sprintf("sample_%d", i+1)
		sample["name"] = fmt.Sprintf("%s Sample %d", element.Name, i+1)
		sample["description"] = element.Definition
		sample["identifier"] = element.Identifier
		sample["version"] = element.Version
		
		// Add metadata fields
		if element.Metadata != nil {
			for k, v := range element.Metadata {
				sample[k] = v
			}
		}
		
		samples = append(samples, sample)
	}
	
	return samples
}

