package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// CacheInterface defines the cache interface.
type CacheInterface interface {
	Get(ctx interface{}, key string, dest interface{}) error
	Set(ctx interface{}, key string, value interface{}, ttl interface{}) error
}

// CatalogHandlers provides HTTP handlers for the catalog API.
type CatalogHandlers struct {
	registry *iso11179.MetadataRegistry
	logger   *log.Logger
	cache    CacheInterface
}

// NewCatalogHandlers creates new catalog handlers.
func NewCatalogHandlers(registry *iso11179.MetadataRegistry, logger *log.Logger) *CatalogHandlers {
	return &CatalogHandlers{
		registry: registry,
		logger:   logger,
	}
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

