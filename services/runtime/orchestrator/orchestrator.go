package orchestrator

import (
	"context"
	"errors"
	"log"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
)

// Orchestrator coordinates analytics fetches and template binding for downstream consumption.
type analyticsClient interface {
	DashboardStats(ctx context.Context) (*analytics.DashboardStats, error)
}

type Orchestrator struct {
	client analyticsClient
	logger *log.Logger
}

// New creates a new orchestrator with the provided analytics client.
func New(client analyticsClient, logger *log.Logger) *Orchestrator {
	return &Orchestrator{
		client: client,
		logger: logger,
	}
}

// FetchDashboardData retrieves dashboard statistics and attaches them to standard templates.
func (o *Orchestrator) FetchDashboardData(ctx context.Context) (*analytics.DashboardStats, []dashboard.Template, error) {
	if o == nil || o.client == nil {
		return nil, nil, ErrUnavailable
	}

	stats, err := o.client.DashboardStats(ctx)
	if err != nil {
		return nil, nil, err
	}

	templates := dashboard.StandardTemplates(time.Now())
	for i := range templates {
		templates[i].Spec.TotalDataElements = stats.TotalDataElements
		templates[i].Spec.TotalDataProducts = stats.TotalDataProducts
		templates[i].Spec.PopularElements = stats.PopularElements
		templates[i].Spec.RecentActivity = stats.RecentActivity
		templates[i].Spec.QualityTrends = stats.QualityTrends
		templates[i].Spec.UsageStatistics = stats.UsageStatistics
		templates[i].Spec.Predictions = stats.Predictions
	}

	return stats, templates, nil
}

// ErrUnavailable indicates the orchestrator is not ready to serve requests.
var ErrUnavailable = errors.New("runtime orchestrator unavailable")
