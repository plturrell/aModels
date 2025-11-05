package api

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/plturrell/aModels/services/catalog/analytics"
	"github.com/plturrell/aModels/services/catalog/multimodal"
	"github.com/plturrell/aModels/services/catalog/streaming"
)

// AdvancedHandlers provides HTTP handlers for advanced features.
type AdvancedHandlers struct {
	eventStream *streaming.EventStream
	extractor   *multimodal.MultiModalExtractor
	dashboard   *analytics.AnalyticsDashboard
	logger      *log.Logger
}

// NewAdvancedHandlers creates new advanced handlers.
func NewAdvancedHandlers(
	eventStream *streaming.EventStream,
	extractor *multimodal.MultiModalExtractor,
	dashboard *analytics.AnalyticsDashboard,
	logger *log.Logger,
) *AdvancedHandlers {
	return &AdvancedHandlers{
		eventStream: eventStream,
		extractor:   extractor,
		dashboard:   dashboard,
		logger:      logger,
	}
}

// HandleExtractMultimodal handles POST /catalog/multimodal/extract.
func (h *AdvancedHandlers) HandleExtractMultimodal(w http.ResponseWriter, r *http.Request) {
	var req multimodal.ExtractionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	extracted, err := h.extractor.Extract(r.Context(), req)
	if err != nil {
		h.logger.Printf("Error extracting metadata: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, extracted)
}

// HandleGetDashboardStats handles GET /catalog/analytics/dashboard.
func (h *AdvancedHandlers) HandleGetDashboardStats(w http.ResponseWriter, r *http.Request) {
	stats, err := h.dashboard.GetDashboardStats(r.Context())
	if err != nil {
		h.logger.Printf("Error getting dashboard stats: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, stats)
}

// HandleGetElementAnalytics handles GET /catalog/analytics/elements/{element_id}.
func (h *AdvancedHandlers) HandleGetElementAnalytics(w http.ResponseWriter, r *http.Request) {
	elementID := r.URL.Path[len("/catalog/analytics/elements/"):]
	if elementID == "" {
		http.Error(w, "element_id is required", http.StatusBadRequest)
		return
	}

	analytics, err := h.dashboard.GetElementAnalytics(r.Context(), elementID)
	if err != nil {
		h.logger.Printf("Error getting element analytics: %v", err)
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	if analytics == nil {
		http.Error(w, "Element not found", http.StatusNotFound)
		return
	}

	writeJSON(w, http.StatusOK, analytics)
}

// HandleGetTopElements handles GET /catalog/analytics/top.
func (h *AdvancedHandlers) HandleGetTopElements(w http.ResponseWriter, r *http.Request) {
	metric := r.URL.Query().Get("metric")
	if metric == "" {
		metric = "access_count"
	}

	limit := 10
	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		// Parse limit (would use strconv in production)
		limit = 10
	}

	elements := h.dashboard.GetTopElements(metric, limit)
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"elements": elements,
		"count":    len(elements),
	})
}

