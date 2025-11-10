package orchestrator

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
)

type mockAnalyticsClient struct {
	stats *analytics.DashboardStats
	err   error
}

func (m *mockAnalyticsClient) DashboardStats(ctx context.Context) (*analytics.DashboardStats, error) {
	return m.stats, m.err
}

func TestOrchestrator_FetchDashboardData(t *testing.T) {
	mockStats := &analytics.DashboardStats{
		TotalDataElements: 100,
		TotalDataProducts: 50,
		PopularElements: []analytics.PopularElement{
			{ElementID: "e1", ElementName: "Element 1", AccessCount: 75},
		},
	}

	expectedTemplates := len(dashboard.StandardTemplates(time.Now()))

	tests := []struct {
		name          string
		clientStats   *analytics.DashboardStats
		clientErr     error
		wantStats     *analytics.DashboardStats
		wantTemplates int
		wantErr       bool
	}{
		{
			name:          "happy path",
			clientStats:   mockStats,
			clientErr:     nil,
			wantStats:     mockStats,
			wantTemplates: expectedTemplates,
			wantErr:       false,
		},
		{
			name:          "client error",
			clientStats:   nil,
			clientErr:     errors.New("catalog down"),
			wantStats:     nil,
			wantTemplates: 0,
			wantErr:       true,
		},
		{
			name:          "empty stats",
			clientStats:   &analytics.DashboardStats{},
			clientErr:     nil,
			wantStats:     &analytics.DashboardStats{},
			wantTemplates: expectedTemplates,
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockClient := &mockAnalyticsClient{stats: tt.clientStats, err: tt.clientErr}
			orch := New(mockClient, nil)

			stats, templates, err := orch.FetchDashboardData(context.Background())

			if (err != nil) != tt.wantErr {
				t.Fatalf("FetchDashboardData() error = %v, wantErr %v", err, tt.wantErr)
			}

			if tt.wantStats != nil {
				if stats == nil {
					t.Fatal("expected stats, got nil")
				}
				if stats.TotalDataElements != tt.wantStats.TotalDataElements {
					t.Errorf("TotalDataElements = %d, want %d", stats.TotalDataElements, tt.wantStats.TotalDataElements)
				}
			} else if stats != nil {
				t.Fatalf("expected nil stats, got %+v", stats)
			}

			if len(templates) != tt.wantTemplates {
				t.Fatalf("templates count = %d, want %d", len(templates), tt.wantTemplates)
			}
		})
	}
}
