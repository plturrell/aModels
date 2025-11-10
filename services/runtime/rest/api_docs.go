package rest

import (
	"encoding/json"
	"net/http"
)

// ServeAPIDocumentation serves API documentation
func (h *Handler) ServeAPIDocumentation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	docs := map[string]interface{}{
		"version": "v1",
		"title":   "Unified Analytics API",
		"description": "Unified API for accessing analytics data across all services",
		"endpoints": map[string]interface{}{
			"/api/v1/analytics": map[string]interface{}{
				"method":      []string{"GET", "POST"},
				"description": "Get analytics data for a specific service or all services",
				"parameters": map[string]interface{}{
					"service": map[string]interface{}{
						"type":        "string",
						"required":    false,
						"default":     "all",
						"description": "Service name: 'catalog', 'training', 'search', or 'all'",
						"enum":        []string{"catalog", "training", "search", "all"},
					},
					"metric": map[string]interface{}{
						"type":        "string",
						"required":    false,
						"description": "Specific metric to retrieve",
					},
					"format": map[string]interface{}{
						"type":        "string",
						"required":    false,
						"default":     "json",
						"description": "Response format: 'json' or 'csv'",
						"enum":        []string{"json", "csv"},
					},
					"aggregation": map[string]interface{}{
						"type":        "string",
						"required":    false,
						"description": "Aggregation function: 'sum', 'avg', 'count', 'max', 'min'",
						"enum":        []string{"sum", "avg", "count", "max", "min"},
					},
					"time_range": map[string]interface{}{
						"type":        "object",
						"required":    false,
						"description": "Time range filter with 'start' and 'end' ISO 8601 timestamps",
					},
					"filters": map[string]interface{}{
						"type":        "object",
						"required":    false,
						"description": "Additional filters as key-value pairs",
					},
				},
				"examples": map[string]interface{}{
					"get_all": map[string]interface{}{
						"url":    "/api/v1/analytics?service=all",
						"method": "GET",
					},
					"get_catalog": map[string]interface{}{
						"url":    "/api/v1/analytics?service=catalog",
						"method": "GET",
					},
					"post_with_filters": map[string]interface{}{
						"url":    "/api/v1/analytics",
						"method": "POST",
						"body": map[string]interface{}{
							"service": "catalog",
							"filters": map[string]interface{}{
								"element_type": "table",
							},
							"time_range": map[string]interface{}{
								"start": "2024-01-01T00:00:00Z",
								"end":   "2024-01-31T23:59:59Z",
							},
						},
					},
				},
			},
			"/api/v1/analytics/system": map[string]interface{}{
				"method":      "GET",
				"description": "Get system-wide analytics overview across all services",
				"response": map[string]interface{}{
					"services": "Map of service names to their analytics data",
					"summary":  "Aggregated system-wide metrics",
					"health":   "Health status of all services",
				},
			},
		},
		"response_format": map[string]interface{}{
			"version":   "API version (e.g., 'v1')",
			"timestamp": "ISO 8601 timestamp of response generation",
			"service":   "Service name or 'all'",
			"data":      "Analytics data specific to the service",
			"metadata": map[string]interface{}{
				"total_records":   "Total number of records (if applicable)",
				"time_range":      "Time range of the data",
				"aggregation":      "Aggregation function applied",
				"generated_at":    "Timestamp when data was generated",
				"cache_hit":       "Whether response was served from cache",
				"processing_time": "Time taken to process the request",
			},
			"links": map[string]interface{}{
				"self":         "Link to this endpoint",
				"documentation": "Link to API documentation",
			},
		},
		"authentication": map[string]interface{}{
			"type":        "API Key or Bearer Token",
			"header":      "X-API-Key or Authorization",
			"description": "Authentication required for all endpoints",
		},
		"rate_limiting": map[string]interface{}{
			"limit":       "100 requests per minute per API key",
			"header":      "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset",
			"description": "Rate limiting applied to prevent abuse",
		},
		"versioning": map[string]interface{}{
			"current":     "v1",
			"header":      "X-API-Version",
			"description": "API version is specified in the URL path and response header",
		},
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(docs)
}

