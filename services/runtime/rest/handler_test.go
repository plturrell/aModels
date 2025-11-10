package rest

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
	"github.com/plturrell/aModels/services/runtime/orchestrator"
)

type mockAnalyticsClient struct {
	stats *analytics.DashboardStats
	err   error
}

func (m *mockAnalyticsClient) DashboardStats(ctx context.Context) (*analytics.DashboardStats, error) {
	return m.stats, m.err
}

func TestHandler_ServeHTTP(t *testing.T) {
	mockStats := &analytics.DashboardStats{
		TotalDataElements: 100,
		TotalDataProducts: 50,
	}
	templateCount := len(dashboard.StandardTemplates(time.Now()))

	tests := []struct {
		name          string
		clientStats   *analytics.DashboardStats
		clientErr     error
		wantStatus    int
		expectJSON    bool
		wantErrorBody string
	}{
		{
			name:        "success",
			clientStats: mockStats,
			clientErr:   nil,
			wantStatus:  http.StatusOK,
			expectJSON: true,
		},
		{
			name:          "orchestrator unavailable",
			clientStats:  nil,
			clientErr:    orchestrator.ErrUnavailable,
			wantStatus:   http.StatusServiceUnavailable,
			expectJSON:   false,
			wantErrorBody: orchestrator.ErrUnavailable.Error(),
		},
		{
			name:        "empty data",
			clientStats: &analytics.DashboardStats{},
			clientErr:   nil,
			wantStatus:  http.StatusOK,
			expectJSON: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockAnalyticsClient{stats: tt.clientStats, err: tt.clientErr}
			handler := NewHandler(orchestrator.New(mockClient, nil))

			req := httptest.NewRequest(http.MethodGet, "/analytics/dashboard", nil)
			rec := httptest.NewRecorder()

			handler.ServeHTTP(rec, req)

			if rec.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d", rec.Code, tt.wantStatus)
			}

			if tt.expectJSON {
				var resp struct {
					Stats     *analytics.DashboardStats `json:"stats"`
					Templates []dashboard.Template      `json:"templates"`
				}
				if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
					t.Fatalf("invalid JSON response: %v", err)
				}

				if tt.clientStats != nil {
					if resp.Stats == nil {
						t.Fatalf("expected stats in response")
					}
					if resp.Stats.TotalDataElements != tt.clientStats.TotalDataElements {
						t.Errorf("TotalDataElements = %d, want %d", resp.Stats.TotalDataElements, tt.clientStats.TotalDataElements)
					}
				} else if resp.Stats != nil {
					t.Fatalf("expected no stats, got %+v", resp.Stats)
				}

				if len(resp.Templates) != templateCount {
					t.Fatalf("templates count = %d, want %d", len(resp.Templates), templateCount)
				}
			} else {
				body := strings.TrimSpace(rec.Body.String())
				if body != tt.wantErrorBody {
					t.Errorf("error body = %q, want %q", body, tt.wantErrorBody)
				}
			}
		})
	}
}
