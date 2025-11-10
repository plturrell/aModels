package rest

import (
	"encoding/json"
	"net/http"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
)

// Handler exposes runtime analytics data over HTTP.
type Handler struct {
	orchestrator          *orchestrator.Orchestrator
	unifiedAnalytics     *UnifiedAnalyticsHandler
}

// NewHandler constructs a runtime analytics HTTP handler.
func NewHandler(orch *orchestrator.Orchestrator) *Handler {
	return &Handler{
		orchestrator:      orch,
		unifiedAnalytics: NewUnifiedAnalyticsHandler(orch),
	}
}

// SetTrainingServiceURL sets the training service URL for unified analytics
func (h *Handler) SetTrainingServiceURL(url string) {
	if h.unifiedAnalytics != nil {
		h.unifiedAnalytics.SetTrainingServiceURL(url)
	}
}

// SetSearchServiceURL sets the search service URL for unified analytics
func (h *Handler) SetSearchServiceURL(url string) {
	if h.unifiedAnalytics != nil {
		h.unifiedAnalytics.SetSearchServiceURL(url)
	}
}

// ServeHTTP responds with aggregated dashboard analytics.
func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	if h == nil || h.orchestrator == nil {
		http.Error(w, "runtime orchestrator unavailable", http.StatusServiceUnavailable)
		return
	}

	stats, templates, err := h.orchestrator.FetchDashboardData(r.Context())
	if err != nil {
		status := http.StatusInternalServerError
		if err == orchestrator.ErrUnavailable {
			status = http.StatusServiceUnavailable
		}
		http.Error(w, err.Error(), status)
		return
	}

	response := struct {
		Stats     *analytics.DashboardStats `json:"stats"`
		Templates []dashboard.Template      `json:"templates"`
	}{
		Stats:     stats,
		Templates: templates,
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	}
}
