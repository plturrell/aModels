package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/research"
)

// CacheInterface defines the cache interface.
type CacheInterface interface {
	Get(ctx interface{}, key string, dest interface{}) error
	Set(ctx interface{}, key string, value interface{}, ttl interface{}) error
}

// CatalogHandlers provides HTTP handlers for the catalog API.
type CatalogHandlers struct {
	registry          *iso11179.MetadataRegistry
	logger            *log.Logger
	cache             CacheInterface
	deepAgentsClient  *DeepAgentsClient
	deepResearchClient *research.DeepResearchClient
	aiDeduplicationEnabled bool
	aiValidationEnabled    bool
	aiResearchEnabled      bool
}

// NewCatalogHandlers creates new catalog handlers.
func NewCatalogHandlers(registry *iso11179.MetadataRegistry, logger *log.Logger) *CatalogHandlers {
	handlers := &CatalogHandlers{
		registry: registry,
		logger:   logger,
		aiDeduplicationEnabled: os.Getenv("CATALOG_AI_DEDUPLICATION_ENABLED") == "true",
		aiValidationEnabled:    os.Getenv("CATALOG_AI_VALIDATION_ENABLED") == "true",
		aiResearchEnabled:      os.Getenv("CATALOG_AI_RESEARCH_ENABLED") == "true",
	}

	// Initialize DeepAgents client if any AI feature is enabled
	if handlers.aiDeduplicationEnabled || handlers.aiValidationEnabled {
		handlers.deepAgentsClient = NewDeepAgentsClient(logger)
	}

	// Initialize DeepResearch client if research is enabled
	if handlers.aiResearchEnabled {
		deepResearchURL := os.Getenv("DEEP_RESEARCH_URL")
		if deepResearchURL == "" {
			deepResearchURL = "http://localhost:8085"
		}
		handlers.deepResearchClient = research.NewDeepResearchClient(deepResearchURL, logger)
	}

	return handlers
}

// SetCache sets the cache for the handlers.
func (h *CatalogHandlers) SetCache(cache CacheInterface) {
	h.cache = cache
}

// HandleListDataElements handles GET /catalog/data-elements.
func (h *CatalogHandlers) HandleListDataElements(w http.ResponseWriter, r *http.Request) {
	var elements []map[string]any
	for _, element := range h.registry.DataElements {
		elements = append(elements, h.dataElementToMap(element))
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"data_elements": elements,
		"count":         len(elements),
	})
}

// HandleGetDataElement handles GET /catalog/data-elements/{id}.
func (h *CatalogHandlers) HandleGetDataElement(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path, "/catalog/data-elements/")
	
	element, ok := h.registry.GetDataElement(id)
	if !ok {
		http.Error(w, "Data element not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, h.dataElementToMap(element))
}

// HandleCreateDataElement handles POST /catalog/data-elements.
func (h *CatalogHandlers) HandleCreateDataElement(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Name                 string `json:"name"`
		DataElementConceptID string `json:"data_element_concept_id"`
		RepresentationID     string `json:"representation_id"`
		Definition           string `json:"definition"`
		Identifier           string `json:"identifier,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Name == "" || req.DataElementConceptID == "" || req.RepresentationID == "" {
		http.Error(w, "name, data_element_concept_id, and representation_id are required", http.StatusBadRequest)
		return
	}

	// Generate identifier if not provided
	identifier := req.Identifier
	if identifier == "" {
		identifier = h.registry.GenerateURI(req.Name)
	}

	// Create data element
	element := iso11179.NewDataElement(
		identifier,
		req.Name,
		req.DataElementConceptID,
		req.RepresentationID,
		req.Definition,
	)

	// Validate
	if errors := h.registry.ValidateDataElement(element); len(errors) > 0 {
		writeJSON(w, http.StatusBadRequest, map[string]any{
			"error":   "Validation failed",
			"details": errors,
		})
		return
	}

	// Register
	h.registry.RegisterDataElement(element)

	writeJSON(w, http.StatusCreated, h.dataElementToMap(element))
}

// checkDuplicatesWithDeepAgents checks for duplicates using DeepAgents.
func (h *CatalogHandlers) checkDuplicatesWithDeepAgents(ctx context.Context, candidates []CandidateElement) (map[int]DeduplicationSuggestion, error) {
	if h.deepAgentsClient == nil || !h.aiDeduplicationEnabled {
		return nil, nil
	}

	// Get existing elements for context
	existing := h.getExistingElementsForDeduplication()

	response, err := h.deepAgentsClient.CheckDuplicates(ctx, candidates, existing)
	if err != nil || response == nil {
		return nil, err
	}

	// Convert to map by index
	suggestions := make(map[int]DeduplicationSuggestion)
	for _, suggestion := range response.Suggestions {
		suggestions[suggestion.Index] = suggestion
	}

	return suggestions, nil
}

// validateWithDeepAgents validates definitions using DeepAgents.
func (h *CatalogHandlers) validateWithDeepAgents(ctx context.Context, candidates []CandidateElement) (map[int]ValidationSuggestion, error) {
	if h.deepAgentsClient == nil || !h.aiValidationEnabled {
		return nil, nil
	}

	response, err := h.deepAgentsClient.ValidateDefinitions(ctx, candidates)
	if err != nil || response == nil {
		return nil, err
	}

	// Convert to map by index
	suggestions := make(map[int]ValidationSuggestion)
	for _, suggestion := range response.Suggestions {
		suggestions[suggestion.Index] = suggestion
	}

	return suggestions, nil
}

// researchSimilarElements researches similar elements using Open Deep Research.
func (h *CatalogHandlers) researchSimilarElements(ctx context.Context, name, definition string) (*research.ResearchReport, error) {
	if h.deepResearchClient == nil || !h.aiResearchEnabled {
		return nil, nil
	}

	query := fmt.Sprintf("Find data elements similar to '%s' with definition: %s", name, definition)
	req := &research.ResearchRequest{
		Query: query,
		Context: map[string]interface{}{
			"name":       name,
			"definition": definition,
		},
		Tools: []string{"sparql_query", "catalog_search"},
	}

	report, err := h.deepResearchClient.Research(ctx, req)
	if err != nil {
		if h.logger != nil {
			h.logger.Printf("Deep Research failed: %v", err)
		}
		return nil, nil // Non-fatal
	}

	return report, nil
}

// getExistingElementsForDeduplication gets existing elements for deduplication context.
func (h *CatalogHandlers) getExistingElementsForDeduplication() []ExistingElement {
	var existing []ExistingElement
	for _, element := range h.registry.DataElements {
		existing = append(existing, ExistingElement{
			Identifier:           element.Identifier,
			Name:                 element.Name,
			Definition:           element.Definition,
			DataElementConceptID: element.DataElementConceptID,
			RepresentationID:     element.RepresentationID,
		})
	}
	return existing
}

// HandleCreateDataElementsBulk handles POST /catalog/data-elements/bulk.
func (h *CatalogHandlers) HandleCreateDataElementsBulk(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	
	var req []struct {
		Name                 string            `json:"name"`
		DataElementConceptID string            `json:"data_element_concept_id"`
		RepresentationID     string            `json:"representation_id"`
		Definition           string            `json:"definition"`
		Identifier           string            `json:"identifier,omitempty"`
		Metadata             map[string]string `json:"metadata,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req) == 0 {
		http.Error(w, "No data elements provided", http.StatusBadRequest)
		return
	}

	// Convert to candidate elements for AI processing
	candidates := make([]CandidateElement, len(req))
	for i, elemReq := range req {
		candidates[i] = CandidateElement{
			Name:                 elemReq.Name,
			Definition:           elemReq.Definition,
			DataElementConceptID: elemReq.DataElementConceptID,
			RepresentationID:     elemReq.RepresentationID,
			Metadata:             elemReq.Metadata,
		}
	}

	// AI Processing: Deduplication
	var duplicateSuggestions map[int]DeduplicationSuggestion
	if h.aiDeduplicationEnabled {
		var err error
		duplicateSuggestions, err = h.checkDuplicatesWithDeepAgents(ctx, candidates)
		if err != nil && h.logger != nil {
			h.logger.Printf("Warning: DeepAgents deduplication failed: %v", err)
		}
	}

	// AI Processing: Validation
	var validationSuggestions map[int]ValidationSuggestion
	if h.aiValidationEnabled {
		var err error
		validationSuggestions, err = h.validateWithDeepAgents(ctx, candidates)
		if err != nil && h.logger != nil {
			h.logger.Printf("Warning: DeepAgents validation failed: %v", err)
		}
	}

	var created []map[string]any
	var errors []map[string]any
	var aiSuggestions []map[string]any
	var duplicatesDetected int
	var researchFindings []map[string]any

	for i, elemReq := range req {
		if elemReq.Name == "" || elemReq.DataElementConceptID == "" || elemReq.RepresentationID == "" {
			errors = append(errors, map[string]any{
				"index": i,
				"error": "name, data_element_concept_id, and representation_id are required",
			})
			continue
		}

		// Check for duplicates (AI-powered)
		if suggestion, ok := duplicateSuggestions[i]; ok && suggestion.Action == "skip" {
			duplicatesDetected++
			if h.logger != nil {
				h.logger.Printf("Skipping duplicate element %d: %s (similar to %s)", i, elemReq.Name, suggestion.SimilarTo)
			}
			aiSuggestions = append(aiSuggestions, map[string]any{
				"index":  i,
				"action": "skipped_duplicate",
				"reason": suggestion.Reason,
				"similar_to": suggestion.SimilarTo,
			})
			continue
		}

		// Research similar elements (optional, non-blocking)
		var researchReport *research.ResearchReport
		if h.aiResearchEnabled {
			var err error
			researchReport, err = h.researchSimilarElements(ctx, elemReq.Name, elemReq.Definition)
			if err == nil && researchReport != nil && researchReport.Report != nil {
				researchFindings = append(researchFindings, map[string]any{
					"index":   i,
					"topic":   researchReport.Report.Topic,
					"summary": researchReport.Report.Summary,
				})
			}
		}

		// Generate identifier if not provided
		identifier := elemReq.Identifier
		if identifier == "" {
			identifier = h.registry.GenerateURI(elemReq.Name)
		}

		// Create data element
		element := iso11179.NewDataElement(
			identifier,
			elemReq.Name,
			elemReq.DataElementConceptID,
			elemReq.RepresentationID,
			elemReq.Definition,
		)

		// Add metadata if provided
		if elemReq.Metadata != nil {
			for k, v := range elemReq.Metadata {
				element.AddMetadata(k, v)
			}
		}

		// Apply AI validation suggestions (if any)
		if suggestion, ok := validationSuggestions[i]; ok {
			element.AddMetadata("ai_validation_score", fmt.Sprintf("%.2f", suggestion.Score))
			if len(suggestion.Improvements) > 0 {
				element.AddMetadata("ai_validation_improvements", strings.Join(suggestion.Improvements, "; "))
				aiSuggestions = append(aiSuggestions, map[string]any{
					"index":        i,
					"type":         "validation",
					"score":        suggestion.Score,
					"improvements": suggestion.Improvements,
				})
			}
		}

		// Validate
		if validationErrors := h.registry.ValidateDataElement(element); len(validationErrors) > 0 {
			errors = append(errors, map[string]any{
				"index":   i,
				"error":   "Validation failed",
				"details": validationErrors,
			})
			continue
		}

		// Register
		h.registry.RegisterDataElement(element)
		created = append(created, h.dataElementToMap(element))
	}

	// Build response
	response := map[string]any{
		"created": len(created),
		"errors":  len(errors),
		"results": created,
		"failures": errors,
	}

	// Add AI enhancements if available
	if len(aiSuggestions) > 0 || duplicatesDetected > 0 || len(researchFindings) > 0 {
		response["ai_suggestions"] = aiSuggestions
		response["duplicates_detected"] = duplicatesDetected
		if len(researchFindings) > 0 {
			response["research_findings"] = researchFindings
		}
	}

	writeJSON(w, http.StatusCreated, response)
}

// HandleGetOntology handles GET /catalog/ontology.
func (h *CatalogHandlers) HandleGetOntology(w http.ResponseWriter, r *http.Request) {
	// Return ontology metadata
	stats := h.registry.GetRegistryStats()
	writeJSON(w, http.StatusOK, map[string]any{
		"namespace":              h.registry.Namespace,
		"data_elements":          stats.DataElementCount,
		"data_element_concepts":  stats.DataElementConceptCount,
		"representations":        stats.RepresentationCount,
		"value_domains":          stats.ValueDomainCount,
		"created_at":             h.registry.CreatedAt,
		"updated_at":             h.registry.UpdatedAt,
	})
}

// HandleSemanticSearch handles POST /catalog/semantic-search.
func (h *CatalogHandlers) HandleSemanticSearch(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query   string `json:"query"`
		ObjectClass string `json:"object_class,omitempty"`
		Property    string `json:"property,omitempty"`
		Source      string `json:"source,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	var results []map[string]any

	// Search by object class
	if req.ObjectClass != "" {
		elements := h.registry.FindDataElementsByObjectClass(req.ObjectClass)
		for _, element := range elements {
			results = append(results, h.dataElementToMap(element))
		}
	}

	// Search by property
	if req.Property != "" {
		elements := h.registry.FindDataElementsByProperty(req.Property)
		for _, element := range elements {
			// Avoid duplicates
			found := false
			for _, existing := range results {
				if existing["identifier"] == element.Identifier {
					found = true
					break
				}
			}
			if !found {
				results = append(results, h.dataElementToMap(element))
			}
		}
	}

	// Search by source
	if req.Source != "" {
		elements := h.registry.FindDataElementsBySource(req.Source)
		for _, element := range elements {
			// Avoid duplicates
			found := false
			for _, existing := range results {
				if existing["identifier"] == element.Identifier {
					found = true
					break
				}
			}
			if !found {
				results = append(results, h.dataElementToMap(element))
			}
		}
	}

	// If no specific filters, return all (or implement text search)
	if req.ObjectClass == "" && req.Property == "" && req.Source == "" {
		// Simple text search on query
		queryLower := strings.ToLower(req.Query)
		for _, element := range h.registry.DataElements {
			if strings.Contains(strings.ToLower(element.Name), queryLower) ||
				strings.Contains(strings.ToLower(element.Definition), queryLower) {
				results = append(results, h.dataElementToMap(element))
			}
		}
	}

	writeJSON(w, http.StatusOK, map[string]any{
		"results": results,
		"count":   len(results),
	})
}

// dataElementToMap converts a data element to a map for JSON serialization.
func (h *CatalogHandlers) dataElementToMap(element *iso11179.DataElement) map[string]any {
	return map[string]any{
		"identifier":              element.Identifier,
		"name":                     element.Name,
		"data_element_concept_id":  element.DataElementConceptID,
		"representation_id":        element.RepresentationID,
		"definition":               element.Definition,
		"version":                  element.Version,
		"created_at":               element.CreatedAt,
		"updated_at":               element.UpdatedAt,
		"registration_status":      element.RegistrationStatus,
		"steward":                  element.Steward,
		"source":                   element.Source,
		"metadata":                 element.Metadata,
	}
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, statusCode int, data any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	json.NewEncoder(w).Encode(data)
}

