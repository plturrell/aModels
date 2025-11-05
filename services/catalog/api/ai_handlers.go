package api

import (
	"encoding/json"
	"log"
	"net/http"
	"strconv"

	"github.com/plturrell/aModels/services/catalog/ai"
)

// AIHandlers provides HTTP handlers for AI capabilities.
type AIHandlers struct {
	discoverer   *ai.MetadataDiscoverer
	predictor    *ai.QualityPredictor
	recommender  *ai.Recommender
	logger       *log.Logger
}

// NewAIHandlers creates new AI handlers.
func NewAIHandlers(
	discoverer *ai.MetadataDiscoverer,
	predictor *ai.QualityPredictor,
	recommender *ai.Recommender,
	logger *log.Logger,
) *AIHandlers {
	return &AIHandlers{
		discoverer:  discoverer,
		predictor:   predictor,
		recommender: recommender,
		logger:      logger,
	}
}

// HandleDiscoverMetadata handles POST /catalog/ai/discover.
func (h *AIHandlers) HandleDiscoverMetadata(w http.ResponseWriter, r *http.Request) {
	var req ai.DiscoveryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	discovered, err := h.discoverer.DiscoverMetadata(r.Context(), req)
	if err != nil {
		h.logger.Printf("Error discovering metadata: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, discovered)
}

// HandlePredictQuality handles POST /catalog/ai/predict-quality.
func (h *AIHandlers) HandlePredictQuality(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ElementID    string `json:"element_id"`
		ForecastDays int    `json:"forecast_days,omitempty"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.ElementID == "" {
		http.Error(w, "element_id is required", http.StatusBadRequest)
		return
	}

	if req.ForecastDays == 0 {
		req.ForecastDays = 7 // Default to 7 days
	}

	prediction, err := h.predictor.PredictQuality(r.Context(), req.ElementID, req.ForecastDays)
	if err != nil {
		h.logger.Printf("Error predicting quality: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, prediction)
}

// HandleGetRecommendations handles POST /catalog/ai/recommendations.
func (h *AIHandlers) HandleGetRecommendations(w http.ResponseWriter, r *http.Request) {
	var req ai.RecommendationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	if req.Limit == 0 {
		req.Limit = 10 // Default to 10 recommendations
	}

	recommendations, err := h.recommender.GetRecommendations(r.Context(), req)
	if err != nil {
		h.logger.Printf("Error getting recommendations: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"recommendations": recommendations,
		"count":           len(recommendations),
	})
}

// HandleRecordUsage handles POST /catalog/ai/usage.
func (h *AIHandlers) HandleRecordUsage(w http.ResponseWriter, r *http.Request) {
	var event ai.UsageEvent
	if err := json.NewDecoder(r.Body).Decode(&event); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	h.recommender.RecordUsage(event)
	writeJSON(w, http.StatusOK, map[string]string{"status": "recorded"})
}

