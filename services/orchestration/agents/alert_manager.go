package agents

import (
	"context"
	"fmt"
	"log"
	"time"
)

// DefaultAlertManager implements AlertManager with basic alerting.
type DefaultAlertManager struct {
	alerts []Anomaly
	logger *log.Logger
}

// NewDefaultAlertManager creates a new default alert manager.
func NewDefaultAlertManager(logger *log.Logger) *DefaultAlertManager {
	return &DefaultAlertManager{
		alerts: []Anomaly{},
		logger: logger,
	}
}

// SendAlert sends an alert.
func (am *DefaultAlertManager) SendAlert(ctx context.Context, anomaly Anomaly) error {
	if am.logger != nil {
		am.logger.Printf("ALERT [%s] %s: %s", anomaly.Severity, anomaly.Type, anomaly.Description)
	}

	am.alerts = append(am.alerts, anomaly)

	// In production, would send to alerting system (email, Slack, PagerDuty, etc.)
	return nil
}

// GetAlerts retrieves alerts matching filters.
func (am *DefaultAlertManager) GetAlerts(ctx context.Context, filters AlertFilters) ([]Anomaly, error) {
	var filtered []Anomaly

	for _, alert := range am.alerts {
		// Apply filters
		if filters.Type != "" && alert.Type != filters.Type {
			continue
		}
		if filters.Severity != "" && alert.Severity != filters.Severity {
			continue
		}
		if filters.StartTime != nil && alert.DetectedAt.Before(*filters.StartTime) {
			continue
		}
		if filters.EndTime != nil && alert.DetectedAt.After(*filters.EndTime) {
			continue
		}

		filtered = append(filtered, alert)

		if filters.Limit > 0 && len(filtered) >= filters.Limit {
			break
		}
	}

	return filtered, nil
}

