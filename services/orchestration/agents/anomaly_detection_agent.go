package agents

import (
	"context"
	"fmt"
	"log"
	"math"
	"time"
)

// AnomalyDetectionAgent detects anomalies in data patterns and quality.
type AnomalyDetectionAgent struct {
	ID            string
	Detectors     []AnomalyDetector
	AlertManager  AlertManager
	GraphClient   GraphClient
	logger        *log.Logger
	lastRun       time.Time
	stats         AnomalyStats
}

// AnomalyDetector detects anomalies using different methods.
type AnomalyDetector interface {
	Detect(ctx context.Context, data []DataPoint) ([]Anomaly, error)
	Name() string
}

// AlertManager manages anomaly alerts.
type AlertManager interface {
	SendAlert(ctx context.Context, anomaly Anomaly) error
	GetAlerts(ctx context.Context, filters AlertFilters) ([]Anomaly, error)
}

// DataPoint represents a data point for anomaly detection.
type DataPoint struct {
	Timestamp   time.Time
	Value       float64
	Dimensions  map[string]interface{}
	Metadata    map[string]interface{}
}

// Anomaly represents a detected anomaly.
type Anomaly struct {
	ID            string
	Type          string // "statistical", "pattern", "quality", "relationship"
	Severity      string // "low", "medium", "high", "critical"
	Description   string
	DetectedAt    time.Time
	DataPoint     DataPoint
	Confidence    float64
	Recommendation string
	Metadata      map[string]interface{}
}

// AlertFilters filters alerts.
type AlertFilters struct {
	Type     string
	Severity  string
	StartTime *time.Time
	EndTime   *time.Time
	Limit     int
}

// AnomalyStats tracks anomaly detection statistics.
type AnomalyStats struct {
	TotalRuns        int
	AnomaliesDetected int
	AlertsSent       int
	LastDetection    time.Time
	DetectionsByType map[string]int
}

// NewAnomalyDetectionAgent creates a new anomaly detection agent.
func NewAnomalyDetectionAgent(
	id string,
	detectors []AnomalyDetector,
	alertManager AlertManager,
	graphClient GraphClient,
	logger *log.Logger,
) *AnomalyDetectionAgent {
	return &AnomalyDetectionAgent{
		ID:           id,
		Detectors:    detectors,
		AlertManager: alertManager,
		GraphClient:  graphClient,
		logger:       logger,
		stats: AnomalyStats{
			DetectionsByType: make(map[string]int),
		},
	}
}

// DetectAnomalies performs anomaly detection on data.
func (agent *AnomalyDetectionAgent) DetectAnomalies(ctx context.Context, data []DataPoint) ([]Anomaly, error) {
	agent.stats.TotalRuns++

	if agent.logger != nil {
		agent.logger.Printf("Detecting anomalies in %d data points", len(data))
	}

	var allAnomalies []Anomaly

	// Run each detector
	for _, detector := range agent.Detectors {
		anomalies, err := detector.Detect(ctx, data)
		if err != nil {
			agent.logger.Printf("Warning: Detector %s failed: %v", detector.Name(), err)
			continue
		}

		allAnomalies = append(allAnomalies, anomalies...)
	}

	// Remove duplicates and merge similar anomalies
	uniqueAnomalies := agent.deduplicateAnomalies(allAnomalies)

	// Send alerts for high-severity anomalies
	for _, anomaly := range uniqueAnomalies {
		if anomaly.Severity == "high" || anomaly.Severity == "critical" {
			if err := agent.AlertManager.SendAlert(ctx, anomaly); err != nil {
				agent.logger.Printf("Warning: Failed to send alert: %v", err)
			} else {
				agent.stats.AlertsSent++
			}
		}

		// Update statistics
		agent.stats.AnomaliesDetected++
		agent.stats.DetectionsByType[anomaly.Type]++
	}

	agent.stats.LastDetection = time.Now()
	agent.lastRun = time.Now()

	if agent.logger != nil {
		agent.logger.Printf("Detected %d anomalies (%d unique)", len(allAnomalies), len(uniqueAnomalies))
	}

	return uniqueAnomalies, nil
}

// deduplicateAnomalies removes duplicate and merges similar anomalies.
func (agent *AnomalyDetectionAgent) deduplicateAnomalies(anomalies []Anomaly) []Anomaly {
	seen := make(map[string]bool)
	var unique []Anomaly

	for _, anomaly := range anomalies {
		key := fmt.Sprintf("%s-%s-%v", anomaly.Type, anomaly.Description, anomaly.DataPoint.Timestamp.Unix())
		if !seen[key] {
			seen[key] = true
			unique = append(unique, anomaly)
		}
	}

	return unique
}

// GetStats returns anomaly detection statistics.
func (agent *AnomalyDetectionAgent) GetStats() AnomalyStats {
	return agent.stats
}

// StatisticalAnomalyDetector detects anomalies using statistical methods.
type StatisticalAnomalyDetector struct {
	threshold float64 // Z-score threshold
	logger    *log.Logger
}

// NewStatisticalAnomalyDetector creates a new statistical detector.
func NewStatisticalAnomalyDetector(threshold float64, logger *log.Logger) *StatisticalAnomalyDetector {
	return &StatisticalAnomalyDetector{
		threshold: threshold,
		logger:    logger,
	}
}

// Name returns the detector name.
func (sd *StatisticalAnomalyDetector) Name() string {
	return "statistical"
}

// Detect detects anomalies using Z-score method.
func (sd *StatisticalAnomalyDetector) Detect(ctx context.Context, data []DataPoint) ([]Anomaly, error) {
	if len(data) < 3 {
		return []Anomaly{}, nil // Need at least 3 points for statistics
	}

	// Calculate mean and standard deviation
	values := make([]float64, len(data))
	for i, point := range data {
		values[i] = point.Value
	}

	mean := sd.calculateMean(values)
	stdDev := sd.calculateStdDev(values, mean)

	if stdDev == 0 {
		return []Anomaly{}, nil // No variation
	}

	var anomalies []Anomaly

	// Detect outliers using Z-score
	for i, point := range data {
		zScore := math.Abs((point.Value - mean) / stdDev)

		if zScore > sd.threshold {
			severity := "medium"
			if zScore > sd.threshold*2 {
				severity = "high"
			}

			anomaly := Anomaly{
				ID:          fmt.Sprintf("statistical-%d-%d", point.Timestamp.Unix(), i),
				Type:        "statistical",
				Severity:    severity,
				Description: fmt.Sprintf("Statistical outlier detected (Z-score: %.2f)", zScore),
				DetectedAt:  time.Now(),
				DataPoint:   point,
				Confidence:  math.Min(zScore/sd.threshold, 1.0),
				Recommendation: "Review data point for data quality issues",
			}

			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies, nil
}

// calculateMean calculates the mean of values.
func (sd *StatisticalAnomalyDetector) calculateMean(values []float64) float64 {
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

// calculateStdDev calculates the standard deviation.
func (sd *StatisticalAnomalyDetector) calculateStdDev(values []float64, mean float64) float64 {
	sumSquares := 0.0
	for _, v := range values {
		diff := v - mean
		sumSquares += diff * diff
	}
	variance := sumSquares / float64(len(values))
	return math.Sqrt(variance)
}

// PatternAnomalyDetector detects anomalies using pattern matching.
type PatternAnomalyDetector struct {
	logger *log.Logger
}

// NewPatternAnomalyDetector creates a new pattern detector.
func NewPatternAnomalyDetector(logger *log.Logger) *PatternAnomalyDetector {
	return &PatternAnomalyDetector{
		logger: logger,
	}
}

// Name returns the detector name.
func (pd *PatternAnomalyDetector) Name() string {
	return "pattern"
}

// Detect detects anomalies using pattern analysis.
func (pd *PatternAnomalyDetector) Detect(ctx context.Context, data []DataPoint) ([]Anomaly, error) {
	// In production, would use ML models for pattern detection
	// For now, detect sudden changes in value
	var anomalies []Anomaly

	if len(data) < 2 {
		return []Anomaly{}, nil
	}

	// Detect sudden changes (more than 50% change)
	for i := 1; i < len(data); i++ {
		prevValue := data[i-1].Value
		currValue := data[i].Value

		if prevValue == 0 {
			continue
		}

		changePercent := math.Abs((currValue - prevValue) / prevValue)

		if changePercent > 0.5 {
			severity := "medium"
			if changePercent > 1.0 {
				severity = "high"
			}

			anomaly := Anomaly{
				ID:          fmt.Sprintf("pattern-%d-%d", data[i].Timestamp.Unix(), i),
				Type:        "pattern",
				Severity:    severity,
				Description: fmt.Sprintf("Sudden value change detected (%.2f%% change)", changePercent*100),
				DetectedAt:  time.Now(),
				DataPoint:   data[i],
				Confidence:  changePercent,
				Recommendation: "Investigate data source for sudden changes",
			}

			anomalies = append(anomalies, anomaly)
		}
	}

	return anomalies, nil
}

