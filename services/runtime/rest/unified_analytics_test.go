package rest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
)

// Mock orchestrator for testing
type mockOrchestrator struct {
	stats     *analytics.DashboardStats
	templates []dashboard.Template
	err       error
}

func (m *mockOrchestrator) FetchDashboardData(ctx context.Context) (*analytics.DashboardStats, []dashboard.Template, error) {
	return m.stats, m.templates, m.err
}

func TestUnifiedAnalyticsHandler_ServeUnifiedAnalytics(t *testing.T) {
	tests := []struct {
		name           string
		method         string
		service        string
		expectedStatus int
		expectedService string
	}{
		{
			name:           "GET request with catalog service",
			method:         "GET",
			service:        "catalog",
			expectedStatus: http.StatusOK,
			expectedService: "catalog",
		},
		{
			name:           "GET request with all services",
			method:         "GET",
			service:        "all",
			expectedStatus: http.StatusOK,
			expectedService: "all",
		},
		{
			name:           "POST request with catalog service",
			method:         "POST",
			service:        "catalog",
			expectedStatus: http.StatusOK,
			expectedService: "catalog",
		},
		{
			name:           "GET request without service (defaults to all)",
			method:         "GET",
			service:        "",
			expectedStatus: http.StatusOK,
			expectedService: "all",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockOrch := &mockOrchestrator{
				stats: &analytics.DashboardStats{
					TotalDataElements: 100,
					TotalDataProducts: 10,
				},
				templates: []dashboard.Template{},
				err:       nil,
			}

			handler := NewUnifiedAnalyticsHandler(mockOrch)

			var req *http.Request
			if tt.method == "POST" {
				body := map[string]interface{}{
					"service": tt.service,
				}
				jsonBody, _ := json.Marshal(body)
				req = httptest.NewRequest("POST", "/api/v1/analytics", bytes.NewReader(jsonBody))
			} else {
				url := "/api/v1/analytics"
				if tt.service != "" {
					url += "?service=" + tt.service
				}
				req = httptest.NewRequest("GET", url, nil)
			}

			w := httptest.NewRecorder()
			handler.ServeUnifiedAnalytics(w, req)

			if w.Code != tt.expectedStatus {
				t.Errorf("expected status %d, got %d", tt.expectedStatus, w.Code)
			}

			if tt.expectedStatus == http.StatusOK {
				var response UnifiedAnalyticsResponse
				if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
					t.Fatalf("failed to unmarshal response: %v", err)
				}

				if response.Service != tt.expectedService {
					t.Errorf("expected service %s, got %s", tt.expectedService, response.Service)
				}

				if response.Version != "v1" {
					t.Errorf("expected version v1, got %s", response.Version)
				}

				if response.Metadata.GeneratedAt.IsZero() {
					t.Error("expected generated_at to be set")
				}
			}
		})
	}
}

func TestUnifiedAnalyticsHandler_ServeSystemWideAnalytics(t *testing.T) {
	mockOrch := &mockOrchestrator{
		stats: &analytics.DashboardStats{
			TotalDataElements: 100,
			TotalDataProducts: 10,
		},
		templates: []dashboard.Template{},
		err:       nil,
	}

	handler := NewUnifiedAnalyticsHandler(mockOrch)
	req := httptest.NewRequest("GET", "/api/v1/analytics/system", nil)
	w := httptest.NewRecorder()

	handler.ServeSystemWideAnalytics(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status %d, got %d", http.StatusOK, w.Code)
	}

	var response SystemWideAnalyticsResponse
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("failed to unmarshal response: %v", err)
	}

	if response.Version != "v1" {
		t.Errorf("expected version v1, got %s", response.Version)
	}

	if len(response.Services) == 0 {
		t.Error("expected services to be populated")
	}

	if response.Health.Overall == "" {
		t.Error("expected health overall to be set")
	}
}


