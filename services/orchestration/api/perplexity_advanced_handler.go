package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
	"golang.org/x/net/websocket"
)

// PerplexityAdvancedHandler provides HTTP handlers for advanced Perplexity features.
type PerplexityAdvancedHandler struct {
	advancedPipeline *agents.PerplexityAdvancedPipeline
	logger           *log.Logger
}

// NewPerplexityAdvancedHandler creates a new advanced handler.
func NewPerplexityAdvancedHandler(logger *log.Logger) (*PerplexityAdvancedHandler, error) {
	// Create base pipeline config
	baseConfig := agents.PerplexityPipelineConfig{
		PerplexityAPIKey:    os.Getenv("PERPLEXITY_API_KEY"),
		PerplexityBaseURL:   getEnvOrDefault("PERPLEXITY_BASE_URL", "https://api.perplexity.ai"),
		DeepSeekOCREndpoint: os.Getenv("DEEPSEEK_OCR_ENDPOINT"),
		DeepSeekOCRAPIKey:   os.Getenv("DEEPSEEK_OCR_API_KEY"),
		DeepResearchURL:     getEnvOrDefault("DEEP_RESEARCH_URL", "http://localhost:8085"),
		CatalogURL:          getEnvOrDefault("CATALOG_URL", "http://catalog:8080"),
		TrainingURL:         getEnvOrDefault("TRAINING_URL", "http://training:8080"),
		LocalAIURL:          getEnvOrDefault("LOCALAI_URL", "http://localai:8080"),
		SearchURL:           getEnvOrDefault("SEARCH_URL", "http://search:8080"),
		ExtractURL:          getEnvOrDefault("EXTRACT_URL", "http://extract:8081"),
		Logger:              logger,
	}

	// Create advanced config
	advancedConfig := agents.PerplexityAdvancedConfig{
		PipelineConfig:      baseConfig,
		EnableStreaming:     getEnvOrDefault("ENABLE_STREAMING", "true") == "true",
		EnableCaching:       getEnvOrDefault("ENABLE_CACHING", "true") == "true",
		EnableAutoScaling:   getEnvOrDefault("ENABLE_AUTO_SCALING", "false") == "true",
		CacheTTL:            30 * time.Minute,
		MaxConcurrentQueries: 10,
		Logger:              logger,
	}

	advancedPipeline, err := agents.NewPerplexityAdvancedPipeline(advancedConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create advanced pipeline: %w", err)
	}

	return &PerplexityAdvancedHandler{
		advancedPipeline: advancedPipeline,
		logger:           logger,
	}, nil
}

// HandleStreamingProcess handles WebSocket streaming for real-time processing.
func (h *PerplexityAdvancedHandler) HandleStreamingProcess(ws *websocket.Conn) {
	defer ws.Close()

	var req struct {
		Query         string                 `json:"query"`
		Config        map[string]interface{} `json:"config,omitempty"`
	}

	if err := websocket.JSON.Receive(ws, &req); err != nil {
		writeStreamError(ws, fmt.Sprintf("Failed to receive request: %v", err))
		return
	}

	if req.Query == "" {
		writeStreamError(ws, "query is required")
		return
	}

	// Build query map
	query := map[string]interface{}{
		"query": req.Query,
	}
	if req.Config != nil {
		for k, v := range req.Config {
			query[k] = v
		}
	}

	// Create streaming channel
	streamChan := make(chan agents.StreamEvent, 100)
	ctx := ws.Request().Context()

	// Process in background
	go func() {
		defer close(streamChan)
		if err := h.advancedPipeline.ProcessDocumentsStreaming(ctx, query, streamChan); err != nil {
			streamChan <- agents.StreamEvent{
				Type:    "error",
				Message: err.Error(),
				Time:    time.Now(),
			}
		}
	}()

	// Stream events to client
	for event := range streamChan {
		if err := websocket.JSON.Send(ws, event); err != nil {
			h.logger.Printf("Failed to send stream event: %v", err)
			return
		}
	}
}

// HandleBatchProcess handles POST /api/perplexity/advanced/batch.
func (h *PerplexityAdvancedHandler) HandleBatchProcess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Queries []map[string]interface{} `json:"queries"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.Queries) == 0 {
		http.Error(w, "queries array is required", http.StatusBadRequest)
		return
	}

	// Process batch
	result, err := h.advancedPipeline.ProcessDocumentsBatch(r.Context(), req.Queries)
	if err != nil {
		http.Error(w, fmt.Sprintf("Batch processing failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, result)
}

// HandleAnalytics handles GET /api/perplexity/advanced/analytics.
func (h *PerplexityAdvancedHandler) HandleAnalytics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	analytics := h.advancedPipeline.GetAnalytics()
	writeJSON(w, http.StatusOK, analytics)
}

// HandleOptimizeQuery handles POST /api/perplexity/advanced/optimize.
func (h *PerplexityAdvancedHandler) HandleOptimizeQuery(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Query map[string]interface{} `json:"query"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	optimized := h.advancedPipeline.OptimizeQuery(req.Query)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"original":  req.Query,
		"optimized": optimized,
	})
}

// writeStreamError writes an error to the WebSocket stream.
func writeStreamError(ws *websocket.Conn, message string) {
	event := agents.StreamEvent{
		Type:    "error",
		Message: message,
		Time:    time.Now(),
	}
	websocket.JSON.Send(ws, event)
}

