package murex

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// MurexHandler provides HTTP handlers for Murex integration.
type MurexHandler struct {
	integration           *MurexIntegration
	terminologyExtractor  *MurexTerminologyExtractor
	catalogPopulator      *MurexCatalogPopulator
	terminologyLearner    *MurexTerminologyLearnerIntegration
	logger                *log.Logger
}

// NewMurexHandler creates a new Murex handler.
func NewMurexHandler(
	integration *MurexIntegration,
	terminologyExtractor *MurexTerminologyExtractor,
	catalogPopulator *MurexCatalogPopulator,
	terminologyLearner *MurexTerminologyLearnerIntegration,
	logger *log.Logger,
) *MurexHandler {
	return &MurexHandler{
		integration:          integration,
		terminologyExtractor: terminologyExtractor,
		catalogPopulator:    catalogPopulator,
		terminologyLearner:   terminologyLearner,
		logger:               logger,
	}
}

// HandleSync handles POST /integrations/murex/sync - Full synchronization.
func (h *MurexHandler) HandleSync(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	if h.logger != nil {
		h.logger.Printf("Starting Murex full synchronization")
	}

	if err := h.integration.SyncFullSync(ctx); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Synchronization failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Murex synchronization completed successfully",
	})
}

// HandleIngestTrades handles POST /integrations/murex/trades - Ingest trades.
func (h *MurexHandler) HandleIngestTrades(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Filters map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		h.respondError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if h.logger != nil {
		h.logger.Printf("Ingesting Murex trades with filters: %v", req.Filters)
	}

	if err := h.integration.IngestTrades(ctx, req.Filters); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Trade ingestion failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Trades ingested successfully",
	})
}

// HandleIngestCashflows handles POST /integrations/murex/cashflows - Ingest cashflows.
func (h *MurexHandler) HandleIngestCashflows(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Filters map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		h.respondError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if err := h.integration.IngestCashflows(ctx, req.Filters); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Cashflow ingestion failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Cashflows ingested successfully",
	})
}

// HandleDiscoverSchema handles GET /integrations/murex/schema - Discover schema.
func (h *MurexHandler) HandleDiscoverSchema(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	schema, err := h.integration.DiscoverSchema(ctx)
	if err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Schema discovery failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, schema)
}

// HandleExtractTerminology handles POST /integrations/murex/terminology/extract - Extract terminology.
func (h *MurexHandler) HandleExtractTerminology(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		FromOpenAPI bool `json:"from_openapi"`
		FromAPIData bool `json:"from_api_data"`
		SampleSize  int  `json:"sample_size"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		// Defaults
		req.FromOpenAPI = true
		req.FromAPIData = true
		req.SampleSize = 100
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	if h.terminologyExtractor == nil {
		h.respondError(w, http.StatusServiceUnavailable, "Terminology extractor not configured")
		return
	}

	if req.FromOpenAPI {
		if err := h.terminologyExtractor.ExtractFromOpenAPISpec(ctx); err != nil {
			h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to extract from OpenAPI: %v", err))
			return
		}
	}

	if req.FromAPIData {
		if req.SampleSize == 0 {
			req.SampleSize = 100
		}
		if err := h.terminologyExtractor.ExtractFromAPIData(ctx, req.SampleSize); err != nil {
			h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to extract from API data: %v", err))
			return
		}
	}

	terminology := h.terminologyExtractor.GetTerminology()
	trainingData := h.terminologyExtractor.GetTrainingData()

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":      "success",
		"message":     "Terminology extracted successfully",
		"terminology": map[string]interface{}{
			"domains":        len(terminology.Domains),
			"roles":          len(terminology.Roles),
			"patterns":       len(terminology.NamingPatterns),
			"entity_types":   len(terminology.EntityTypes),
			"relationships":  len(terminology.Relationships),
		},
		"training_data": map[string]interface{}{
			"schema_examples":      len(trainingData.SchemaExamples),
			"field_examples":       len(trainingData.FieldExamples),
			"relationship_examples": len(trainingData.RelationshipExamples),
			"value_patterns":       len(trainingData.ValuePatterns),
		},
	})
}

// HandlePopulateCatalog handles POST /integrations/murex/catalog/populate - Populate catalog.
func (h *MurexHandler) HandlePopulateCatalog(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if h.catalogPopulator == nil {
		h.respondError(w, http.StatusServiceUnavailable, "Catalog populator not configured")
		return
	}

	if err := h.catalogPopulator.PopulateAll(ctx); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to populate catalog: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Catalog populated successfully from Murex terminology and training data",
	})
}

// HandleTrainTerminology handles POST /integrations/murex/terminology/train - Train terminology learner.
func (h *MurexHandler) HandleTrainTerminology(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	if h.terminologyLearner == nil {
		h.respondError(w, http.StatusServiceUnavailable, "Terminology learner integration not configured")
		return
	}

	// Train from extracted terminology
	if err := h.terminologyLearner.TrainFromExtractedTerminology(ctx); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to train terminology learner: %v", err))
		return
	}

	// Also train from schema examples
	if err := h.terminologyLearner.TrainFromSchemaExamples(ctx); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Failed to train from schema examples: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Terminology learner trained successfully from Murex data",
	})
}

// HandleExportTrainingData handles GET /integrations/murex/terminology/export - Export training data.
func (h *MurexHandler) HandleExportTrainingData(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h.terminologyLearner == nil {
		h.respondError(w, http.StatusServiceUnavailable, "Terminology learner integration not configured")
		return
	}

	trainingData := h.terminologyLearner.ExportTrainingData()
	h.respondJSON(w, http.StatusOK, trainingData)
}

// respondJSON writes a JSON response.
func (h *MurexHandler) respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		if h.logger != nil {
			h.logger.Printf("Failed to encode JSON response: %v", err)
		}
	}
}

// respondError writes an error response.
func (h *MurexHandler) respondError(w http.ResponseWriter, status int, message string) {
	h.respondJSON(w, status, map[string]interface{}{
		"error":   true,
		"message": message,
	})
}

