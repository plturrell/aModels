package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// PerplexityHandler provides HTTP handlers for Perplexity document processing.
type PerplexityHandler struct {
	pipeline *agents.PerplexityPipeline
	logger   *log.Logger
}

// NewPerplexityHandler creates a new Perplexity handler.
func NewPerplexityHandler(logger *log.Logger) (*PerplexityHandler, error) {
	// Load configuration from environment
	config := agents.PerplexityPipelineConfig{
		PerplexityAPIKey:    os.Getenv("PERPLEXITY_API_KEY"),
		PerplexityBaseURL:   getEnvOrDefault("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
		DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
		DeepSeekOCRAPIKey:   os.Getenv("DEEPSEEK_OCR_API_KEY"),
		DeepResearchURL:     getEnvOrDefault("DEEP_RESEARCH_URL", "http://localhost:8085"),
		UnifiedWorkflowURL:  getEnvOrDefault("UNIFIED_WORKFLOW_URL", "http://graph-service:8081"),
		CatalogURL:          getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:         getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:          getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:           getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		ExtractURL:          getEnvOrDefault("EXTRACT_URL", "http://extract:8081"),
		Logger:              logger,
	}

	pipeline, err := agents.NewPerplexityPipeline(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create Perplexity pipeline: %w", err)
	}

	return &PerplexityHandler{
		pipeline: pipeline,
		logger:   logger,
	}, nil
}

// HandleProcessDocuments handles POST /api/perplexity/process.
func (h *PerplexityHandler) HandleProcessDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query         string                 `json:"query"`
		Model         string                 `json:"model,omitempty"`
		Limit         int                    `json:"limit,omitempty"`
		IncludeImages bool                   `json:"include_images,omitempty"`
		Config        map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Build query map
	query := map[string]interface{}{
		"query":         req.Query,
		"include_images": req.IncludeImages,
	}

	if req.Model != "" {
		query["model"] = req.Model
	}

	if req.Limit > 0 {
		query["limit"] = req.Limit
	}

	// Merge additional config
	if req.Config != nil {
		for k, v := range req.Config {
			query[k] = v
		}
	}

	// Process documents
	err := h.pipeline.ProcessDocuments(r.Context(), query)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to process documents: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "completed",
		"query":   req.Query,
		"message": "Documents processed successfully through OCR, catalog, training, local AI, and search with Deep Research integration",
	})
}

// HandleProcessWithIngestion handles POST /api/perplexity/process-with-ingestion.
// This endpoint uses the data ingestion agent pattern for consistency with other sources.
func (h *PerplexityHandler) HandleProcessWithIngestion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query  string                 `json:"query"`
		Config map[string]interface{} `json:"config,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	// Build config for ingestion agent
	config := map[string]interface{}{
		"query": req.Query,
	}

	// Merge additional config
	if req.Config != nil {
		for k, v := range req.Config {
			config[k] = v
		}
	}

	// Use the pipeline's ProcessDocuments method
	err := h.pipeline.ProcessDocuments(r.Context(), config)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to process documents: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "completed",
		"query":   req.Query,
		"message": "Documents ingested and processed successfully",
	})
}

// getEnvOrDefault returns the environment variable value or a default.
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

