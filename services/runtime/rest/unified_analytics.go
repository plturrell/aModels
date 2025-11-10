package rest

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
)

// UnifiedAnalyticsHandler handles unified /api/v1/analytics endpoint
type UnifiedAnalyticsHandler struct {
	orchestrator      *orchestrator.Orchestrator
	trainingClient    *TrainingServiceClient
	searchClient      *SearchServiceClient
	version           string
}

// NewUnifiedAnalyticsHandler creates a new unified analytics handler
func NewUnifiedAnalyticsHandler(orch *orchestrator.Orchestrator) *UnifiedAnalyticsHandler {
	return &UnifiedAnalyticsHandler{
		orchestrator:   orch,
		trainingClient: NewTrainingServiceClient(""),
		searchClient:   NewSearchServiceClient(""),
		version:        "v1",
	}
}

// SetTrainingServiceURL sets the training service URL
func (h *UnifiedAnalyticsHandler) SetTrainingServiceURL(url string) {
	h.trainingClient = NewTrainingServiceClient(url)
}

// SetSearchServiceURL sets the search service URL
func (h *UnifiedAnalyticsHandler) SetSearchServiceURL(url string) {
	h.searchClient = NewSearchServiceClient(url)
}

// UnifiedAnalyticsRequest represents a request to the unified analytics API
type UnifiedAnalyticsRequest struct {
	Service    string                 `json:"service,omitempty"`    // "catalog", "training", "search", "all"
	Metric     string                 `json:"metric,omitempty"`     // Specific metric to retrieve
	TimeRange  *TimeRange             `json:"time_range,omitempty"` // Time range filter
	Filters    map[string]interface{} `json:"filters,omitempty"`    // Additional filters
	Format     string                 `json:"format,omitempty"`     // "json", "csv", "summary"
	Aggregation string                `json:"aggregation,omitempty"` // "sum", "avg", "count", "max", "min"
}

// TimeRange represents a time range filter
type TimeRange struct {
	Start time.Time `json:"start"`
	End   time.Time `json:"end"`
}

// UnifiedAnalyticsResponse represents the unified analytics API response
type UnifiedAnalyticsResponse struct {
	Version     string                 `json:"version"`
	Timestamp   time.Time              `json:"timestamp"`
	Service     string                 `json:"service"`
	Data        map[string]interface{} `json:"data"`
	Metadata    ResponseMetadata       `json:"metadata"`
	Links       map[string]string      `json:"links,omitempty"`
}

// ResponseMetadata contains metadata about the response
type ResponseMetadata struct {
	TotalRecords   int64     `json:"total_records,omitempty"`
	TimeRange      *TimeRange `json:"time_range,omitempty"`
	Aggregation    string    `json:"aggregation,omitempty"`
	GeneratedAt    time.Time `json:"generated_at"`
	CacheHit       bool      `json:"cache_hit,omitempty"`
	ProcessingTime string    `json:"processing_time,omitempty"`
}

// SystemWideAnalyticsResponse represents system-wide analytics overview
type SystemWideAnalyticsResponse struct {
	Version     string                 `json:"version"`
	Timestamp   time.Time              `json:"timestamp"`
	Services    map[string]interface{} `json:"services"`
	Summary     SystemSummary          `json:"summary"`
	Health      SystemHealth           `json:"health"`
}

// SystemSummary provides high-level system metrics
type SystemSummary struct {
	TotalRequests    int64   `json:"total_requests"`
	TotalErrors      int64   `json:"total_errors"`
	AverageLatency   float64 `json:"average_latency_ms"`
	SuccessRate      float64 `json:"success_rate"`
	ActiveServices   int     `json:"active_services"`
	TotalDataElements int64   `json:"total_data_elements"`
}

// SystemHealth represents health status of all services
type SystemHealth struct {
	Overall    string              `json:"overall"` // "healthy", "degraded", "down"
	Services   map[string]string   `json:"services"`
	LastCheck  time.Time           `json:"last_check"`
}

// ServeUnifiedAnalytics handles the unified /api/v1/analytics endpoint
func (h *UnifiedAnalyticsHandler) ServeUnifiedAnalytics(w http.ResponseWriter, r *http.Request) {
	startTime := time.Now()

	if r.Method != http.MethodGet && r.Method != http.MethodPost {
		w.Header().Set("Allow", "GET, POST")
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	var req UnifiedAnalyticsRequest
	if r.Method == http.MethodPost {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}
	} else {
		// Parse query parameters for GET requests
		req.Service = r.URL.Query().Get("service")
		req.Metric = r.URL.Query().Get("metric")
		req.Format = r.URL.Query().Get("format")
		req.Aggregation = r.URL.Query().Get("aggregation")
	}

	// Default to "all" if no service specified
	if req.Service == "" {
		req.Service = "all"
	}

	// Fetch analytics data
	var data map[string]interface{}
	var err error

	switch req.Service {
	case "catalog", "all":
		if h.orchestrator != nil {
			ctx := r.Context()
			if ctx == nil {
				ctx = context.Background()
			}
			stats, templates, fetchErr := h.orchestrator.FetchDashboardData(ctx)
			if fetchErr == nil {
				data = map[string]interface{}{
					"dashboard_stats": stats,
					"templates":        templates,
				}
			} else {
				err = fetchErr
			}
		}
	case "training":
		// Fetch training service analytics
		if h.trainingClient != nil {
			ctx := r.Context()
			if ctx == nil {
				ctx = context.Background()
			}
			trainingMetrics, fetchErr := h.trainingClient.FetchAnalytics(ctx)
			if fetchErr == nil {
				data = map[string]interface{}{
					"training_metrics": trainingMetrics,
				}
			} else {
				// Log error but return default metrics
				data = map[string]interface{}{
					"training_metrics": map[string]interface{}{
						"active_experiments": 0,
						"completed_runs":     0,
						"average_accuracy":   0.0,
						"error":              fetchErr.Error(),
					},
				}
			}
		} else {
			data = map[string]interface{}{
				"training_metrics": map[string]interface{}{
					"active_experiments": 0,
					"completed_runs":     0,
					"average_accuracy":   0.0,
				},
			}
		}
	case "search":
		// Fetch search service analytics
		if h.searchClient != nil {
			ctx := r.Context()
			if ctx == nil {
				ctx = context.Background()
			}
			searchMetrics, fetchErr := h.searchClient.FetchAnalytics(ctx)
			if fetchErr == nil {
				data = map[string]interface{}{
					"search_metrics": searchMetrics,
				}
			} else {
				// Log error but return default metrics
				data = map[string]interface{}{
					"search_metrics": map[string]interface{}{
						"total_queries":     0,
						"average_latency_ms": 0.0,
						"cache_hit_rate":    0.0,
						"error":             fetchErr.Error(),
					},
				}
			}
		} else {
			data = map[string]interface{}{
				"search_metrics": map[string]interface{}{
					"total_queries":     0,
					"average_latency_ms": 0.0,
					"cache_hit_rate":    0.0,
				},
			}
		}
	default:
		http.Error(w, "unknown service: "+req.Service, http.StatusBadRequest)
		return
	}

	if err != nil {
		http.Error(w, "failed to fetch analytics: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// Apply filters and aggregation if specified
	if req.Filters != nil {
		data = h.applyFilters(data, req.Filters)
	}

	if req.Aggregation != "" {
		data = h.applyAggregation(data, req.Aggregation)
	}

	// Format response
	processingTime := time.Since(startTime)
	response := UnifiedAnalyticsResponse{
		Version:   h.version,
		Timestamp: time.Now(),
		Service:   req.Service,
		Data:      data,
		Metadata: ResponseMetadata{
			GeneratedAt:    time.Now(),
			ProcessingTime: processingTime.String(),
		},
		Links: map[string]string{
			"self":     r.URL.String(),
			"documentation": "/api/v1/analytics/docs",
		},
	}

	// Set response format
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-API-Version", h.version)
	w.Header().Set("X-Response-Time", processingTime.String())

	if req.Format == "csv" {
		// Convert to CSV format
		w.Header().Set("Content-Type", "text/csv; charset=utf-8")
		w.Header().Set("Content-Disposition", "attachment; filename=analytics.csv")
		// CSV conversion would be implemented here
		json.NewEncoder(w).Encode(response) // Fallback to JSON for now
		return
	}

	json.NewEncoder(w).Encode(response)
}

// ServeSystemWideAnalytics provides system-wide analytics overview
func (h *UnifiedAnalyticsHandler) ServeSystemWideAnalytics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Collect analytics from all services
	services := make(map[string]interface{})
	health := SystemHealth{
		Services:  make(map[string]string),
		LastCheck: time.Now(),
	}

	// Catalog service
	if h.orchestrator != nil {
		ctx := r.Context()
		if ctx == nil {
			ctx = context.Background()
		}
		stats, _, fetchErr := h.orchestrator.FetchDashboardData(ctx)
		if fetchErr == nil {
			services["catalog"] = map[string]interface{}{
				"total_elements": stats.TotalDataElements,
				"total_products": stats.TotalDataProducts,
				"usage":          stats.UsageStatistics,
			}
			health.Services["catalog"] = "healthy"
		} else {
			health.Services["catalog"] = "unavailable"
		}
	}

	// Training service
	if h.trainingClient != nil {
		ctx := r.Context()
		if ctx == nil {
			ctx = context.Background()
		}
		trainingMetrics, fetchErr := h.trainingClient.FetchAnalytics(ctx)
		if fetchErr == nil {
			services["training"] = map[string]interface{}{
				"active_experiments": trainingMetrics.ActiveExperiments,
				"completed_runs":    trainingMetrics.CompletedRuns,
				"average_accuracy":  trainingMetrics.AverageAccuracy,
			}
			health.Services["training"] = "healthy"
		} else {
			services["training"] = map[string]interface{}{
				"active_experiments": 0,
				"completed_runs":    0,
			}
			health.Services["training"] = "unavailable"
		}
	} else {
		services["training"] = map[string]interface{}{
			"active_experiments": 0,
			"completed_runs":    0,
		}
		health.Services["training"] = "unknown"
	}

	// Search service
	if h.searchClient != nil {
		ctx := r.Context()
		if ctx == nil {
			ctx = context.Background()
		}
		searchMetrics, fetchErr := h.searchClient.FetchAnalytics(ctx)
		if fetchErr == nil {
			services["search"] = map[string]interface{}{
				"total_queries": searchMetrics.TotalQueries,
				"cache_hits":   int(float64(searchMetrics.TotalQueries) * searchMetrics.CacheHitRate),
				"cache_hit_rate": searchMetrics.CacheHitRate,
			}
			health.Services["search"] = "healthy"
		} else {
			services["search"] = map[string]interface{}{
				"total_queries": 0,
				"cache_hits":    0,
			}
			health.Services["search"] = "unavailable"
		}
	} else {
		services["search"] = map[string]interface{}{
			"total_queries": 0,
			"cache_hits":    0,
		}
		health.Services["search"] = "unknown"
	}

	// Calculate overall health
	overallHealth := "healthy"
	for _, status := range health.Services {
		if status == "unavailable" || status == "down" {
			overallHealth = "degraded"
			break
		}
	}
	health.Overall = overallHealth

	// Calculate summary
	summary := SystemSummary{
		TotalRequests:    0,
		TotalErrors:      0,
		AverageLatency:   0.0,
		SuccessRate:       1.0,
		ActiveServices:   len(services),
		TotalDataElements: 0,
	}

	if catalogData, ok := services["catalog"].(map[string]interface{}); ok {
		if total, ok := catalogData["total_elements"].(int); ok {
			summary.TotalDataElements = int64(total)
		}
	}

	response := SystemWideAnalyticsResponse{
		Version:   h.version,
		Timestamp: time.Now(),
		Services:  services,
		Summary:   summary,
		Health:    health,
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("X-API-Version", h.version)
	json.NewEncoder(w).Encode(response)
}

// applyFilters applies filters to the data
func (h *UnifiedAnalyticsHandler) applyFilters(data map[string]interface{}, filters map[string]interface{}) map[string]interface{} {
	// Filtering logic would be implemented here
	// For now, return data as-is
	return data
}

// applyAggregation applies aggregation to the data
func (h *UnifiedAnalyticsHandler) applyAggregation(data map[string]interface{}, aggregation string) map[string]interface{} {
	// Aggregation logic would be implemented here
	// For now, return data as-is
	return data
}

