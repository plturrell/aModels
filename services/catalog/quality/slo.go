package quality

import (
	"time"
)

// SLO represents a Service Level Objective for a data product.
type SLO struct {
	// Name is the name of the SLO (e.g., "Freshness", "Completeness", "Accuracy")
	Name string

	// Target is the target value (e.g., 0.95 for 95% freshness)
	Target float64

	// Current is the current value
	Current float64

	// Window is the time window for measurement (e.g., "24h", "7d")
	Window string

	// LastUpdated is when this SLO was last updated
	LastUpdated time.Time

	// Status is the current status ("met", "warning", "violated")
	Status string
}

// QualityMetrics represents quality metrics for a data product.
type QualityMetrics struct {
	// Overall quality score (0-1)
	QualityScore float64

	// Freshness: How recent is the data?
	FreshnessScore float64
	LastUpdated    time.Time
	ExpectedUpdateFrequency string // e.g., "daily", "hourly", "real-time"

	// Completeness: How much data is present vs expected?
	CompletenessScore float64
	TotalRecords      int64
	NullRecords       int64
	MissingRecords    int64

	// Accuracy: How correct is the data?
	AccuracyScore float64
	ErrorRate     float64

	// Consistency: How consistent is the data across sources?
	ConsistencyScore float64

	// Validity: Does the data conform to schema/constraints?
	ValidityScore float64
	InvalidRecords int64

	// SLOs are the service level objectives
	SLOs []SLO

	// LastValidated is when quality was last validated
	LastValidated time.Time

	// ValidationStatus is the overall validation status
	ValidationStatus string // "passed", "warning", "failed"
}

// CalculateOverallScore calculates the overall quality score from individual metrics.
func (qm *QualityMetrics) CalculateOverallScore() float64 {
	// Weighted average of individual scores
	weights := map[string]float64{
		"freshness":   0.20,
		"completeness": 0.25,
		"accuracy":    0.25,
		"consistency": 0.15,
		"validity":    0.15,
	}

	score := qm.FreshnessScore*weights["freshness"] +
		qm.CompletenessScore*weights["completeness"] +
		qm.AccuracyScore*weights["accuracy"] +
		qm.ConsistencyScore*weights["consistency"] +
		qm.ValidityScore*weights["validity"]

	qm.QualityScore = score
	return score
}

// CheckSLOs checks if all SLOs are being met.
func (qm *QualityMetrics) CheckSLOs() {
	for i := range qm.SLOs {
		slo := &qm.SLOs[i]
		
		if slo.Current >= slo.Target {
			slo.Status = "met"
		} else if slo.Current >= slo.Target*0.9 {
			slo.Status = "warning"
		} else {
			slo.Status = "violated"
		}
		
		slo.LastUpdated = time.Now()
	}

	// Update overall validation status
	allMet := true
	hasWarning := false
	for _, slo := range qm.SLOs {
		if slo.Status == "violated" {
			allMet = false
			break
		}
		if slo.Status == "warning" {
			hasWarning = true
		}
	}

	if allMet && !hasWarning {
		qm.ValidationStatus = "passed"
	} else if allMet && hasWarning {
		qm.ValidationStatus = "warning"
	} else {
		qm.ValidationStatus = "failed"
	}
}

// NewQualityMetrics creates a new QualityMetrics instance.
func NewQualityMetrics() *QualityMetrics {
	return &QualityMetrics{
		SLOs: []SLO{},
		ValidationStatus: "unknown",
	}
}

// AddSLO adds a new SLO to the quality metrics.
func (qm *QualityMetrics) AddSLO(name string, target float64, window string) {
	slo := SLO{
		Name:        name,
		Target:      target,
		Current:     0.0,
		Window:      window,
		LastUpdated: time.Now(),
		Status:      "unknown",
	}
	qm.SLOs = append(qm.SLOs, slo)
}

// UpdateMetric updates a specific metric.
func (qm *QualityMetrics) UpdateMetric(metricName string, value float64) {
	switch metricName {
	case "freshness":
		qm.FreshnessScore = value
	case "completeness":
		qm.CompletenessScore = value
	case "accuracy":
		qm.AccuracyScore = value
	case "consistency":
		qm.ConsistencyScore = value
	case "validity":
		qm.ValidityScore = value
	}
	qm.LastValidated = time.Now()
	qm.CalculateOverallScore()
	qm.CheckSLOs()
}

