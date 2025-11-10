package ai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/quality"
)

// QualityPredictor provides predictive quality monitoring capabilities.
type QualityPredictor struct {
	extractServiceURL string
	httpClient        *http.Client
	logger            *log.Logger
	historicalData    []QualityHistoryPoint
}

// NewQualityPredictor creates a new quality predictor.
func NewQualityPredictor(extractServiceURL string, logger *log.Logger) *QualityPredictor {
	return &QualityPredictor{
		extractServiceURL: extractServiceURL,
		httpClient:        &http.Client{Timeout: 30 * time.Second},
		logger:            logger,
		historicalData:    []QualityHistoryPoint{},
	}
}

// QualityHistoryPoint represents a historical quality measurement.
type QualityHistoryPoint struct {
	Timestamp     time.Time              `json:"timestamp"`
	ElementID     string                 `json:"element_id"`
	Metrics       *quality.QualityMetrics `json:"metrics"`
	QualityScore  float64                `json:"quality_score"`
}

// QualityPrediction represents a quality prediction.
type QualityPrediction struct {
	ElementID          string                 `json:"element_id"`
	CurrentQuality     *quality.QualityMetrics `json:"current_quality"`
	PredictedQuality   *quality.QualityMetrics `json:"predicted_quality"`
	Trend              string                 `json:"trend"`              // "improving", "degrading", "stable"
	AnomalyDetected    bool                   `json:"anomaly_detected"`
	AnomalyConfidence  float64                `json:"anomaly_confidence"`
	AnomalyReason      string                 `json:"anomaly_reason,omitempty"`
	Forecast           []QualityForecastPoint `json:"forecast,omitempty"`
	RiskLevel          string                 `json:"risk_level"` // "low", "medium", "high", "critical"
	Recommendations    []string               `json:"recommendations,omitempty"`
}

// QualityForecastPoint represents a forecast point.
type QualityForecastPoint struct {
	Timestamp    time.Time              `json:"timestamp"`
	PredictedMetrics *quality.QualityMetrics `json:"predicted_metrics"`
	Confidence   float64                `json:"confidence"`
}

// PredictQuality predicts future quality for a data element.
func (qp *QualityPredictor) PredictQuality(
	ctx context.Context,
	elementID string,
	forecastDays int,
) (*QualityPrediction, error) {
	qp.logger.Printf("Predicting quality for element: %s (forecast: %d days)", elementID, forecastDays)

	// Step 1: Fetch current quality metrics
	currentMetrics, err := qp.fetchCurrentQuality(ctx, elementID)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch current quality: %w", err)
	}

	// Step 2: Fetch historical quality data
	history, err := qp.fetchHistoricalQuality(ctx, elementID, 30) // Last 30 days
	if err != nil {
		qp.logger.Printf("Warning: Failed to fetch historical data: %v", err)
		history = []QualityHistoryPoint{}
	}

	// Step 3: Detect anomalies
	anomalyDetected, anomalyConfidence, anomalyReason := qp.detectAnomalies(currentMetrics, history)

	// Step 4: Predict trend
	trend := qp.predictTrend(history)

	// Step 5: Forecast future quality
	forecast := qp.forecastQuality(history, currentMetrics, forecastDays)

	// Step 6: Calculate risk level
	riskLevel := qp.calculateRiskLevel(currentMetrics, anomalyDetected, trend)

	// Step 7: Generate recommendations
	recommendations := qp.generateRecommendations(currentMetrics, trend, anomalyDetected)

	// Step 8: Predict future metrics
	predictedMetrics := qp.predictFutureMetrics(currentMetrics, history, trend)

	prediction := &QualityPrediction{
		ElementID:         elementID,
		CurrentQuality:    currentMetrics,
		PredictedQuality: predictedMetrics,
		Trend:             trend,
		AnomalyDetected:   anomalyDetected,
		AnomalyConfidence: anomalyConfidence,
		AnomalyReason:     anomalyReason,
		Forecast:          forecast,
		RiskLevel:         riskLevel,
		Recommendations:   recommendations,
	}

	return prediction, nil
}

// fetchCurrentQuality fetches current quality metrics.
func (qp *QualityPredictor) fetchCurrentQuality(ctx context.Context, elementID string) (*quality.QualityMetrics, error) {
	url := fmt.Sprintf("%s/metrics/quality?element_id=%s", qp.extractServiceURL, elementID)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := qp.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read response body for better error messages
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
	}

	var metrics quality.QualityMetrics
	if err := json.NewDecoder(resp.Body).Decode(&metrics); err != nil {
		return nil, err
	}

	return &metrics, nil
}

// fetchHistoricalQuality fetches historical quality data.
func (qp *QualityPredictor) fetchHistoricalQuality(
	ctx context.Context,
	elementID string,
	days int,
) ([]QualityHistoryPoint, error) {
	url := fmt.Sprintf("%s/metrics/quality/history?element_id=%s&days=%d", qp.extractServiceURL, elementID, days)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := qp.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		// Read response body for better error messages
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
	}

	var history []QualityHistoryPoint
	if err := json.NewDecoder(resp.Body).Decode(&history); err != nil {
		return nil, err
	}

	return history, nil
}

// detectAnomalies detects anomalies in quality metrics.
func (qp *QualityPredictor) detectAnomalies(
	current *quality.QualityMetrics,
	history []QualityHistoryPoint,
) (bool, float64, string) {
	if len(history) < 3 {
		return false, 0.0, ""
	}

	// Calculate average quality scores
	var avgScore float64
	for _, point := range history {
		avgScore += point.QualityScore
	}
	avgScore /= float64(len(history))

	// Calculate standard deviation
	var variance float64
	for _, point := range history {
		variance += math.Pow(point.QualityScore-avgScore, 2)
	}
	stdDev := math.Sqrt(variance / float64(len(history)))

	// Current quality score
	currentScore := qp.calculateQualityScore(current)

	// Anomaly detection: if current score deviates by more than 2 standard deviations
	threshold := 2.0 * stdDev
	deviation := math.Abs(currentScore - avgScore)

	if deviation > threshold {
		confidence := math.Min(deviation/threshold, 1.0)
		reason := fmt.Sprintf("Quality score %.2f deviates %.2f from average %.2f (threshold: %.2f)",
			currentScore, deviation, avgScore, threshold)
		return true, confidence, reason
	}

	return false, 0.0, ""
}

// calculateQualityScore calculates an overall quality score from metrics.
func (qp *QualityPredictor) calculateQualityScore(metrics *quality.QualityMetrics) float64 {
	// Weighted average of all quality dimensions
	weights := map[string]float64{
		"freshness":   0.2,
		"completeness": 0.25,
		"accuracy":    0.25,
		"consistency": 0.15,
		"validity":    0.15,
	}

	score := metrics.FreshnessScore*weights["freshness"] +
		metrics.AccuracyScore*weights["accuracy"] +
		metrics.ConsistencyScore*weights["consistency"] +
		metrics.ValidityScore*weights["validity"]
	// CompletenessRate field doesn't exist, using AccuracyScore as proxy
	score += metrics.AccuracyScore * weights["completeness"]

	return score
}

// predictTrend predicts the quality trend.
func (qp *QualityPredictor) predictTrend(history []QualityHistoryPoint) string {
	if len(history) < 2 {
		return "stable"
	}

	// Simple linear regression to determine trend
	var sumX, sumY, sumXY, sumX2 float64
	n := float64(len(history))

	for i, point := range history {
		x := float64(i)
		y := point.QualityScore
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)

	if slope > 0.01 {
		return "improving"
	} else if slope < -0.01 {
		return "degrading"
	}
	return "stable"
}

// forecastQuality forecasts future quality.
func (qp *QualityPredictor) forecastQuality(
	history []QualityHistoryPoint,
	current *quality.QualityMetrics,
	days int,
) []QualityForecastPoint {
	if len(history) < 2 {
		// Not enough data, return current metrics
		return []QualityForecastPoint{
			{
				Timestamp:       time.Now().Add(24 * time.Hour),
				PredictedMetrics: current,
				Confidence:      0.5,
			},
		}
	}

	var forecast []QualityForecastPoint
	currentScore := qp.calculateQualityScore(current)
	trend := qp.predictTrend(history)

	// Simple forecast based on trend
	var dailyChange float64
	switch trend {
	case "improving":
		dailyChange = 0.01
	case "degrading":
		dailyChange = -0.01
	default:
		dailyChange = 0.0
	}

	for i := 1; i <= days; i++ {
		predictedScore := currentScore + (dailyChange * float64(i))
		predictedScore = math.Max(0.0, math.Min(1.0, predictedScore)) // Clamp to [0, 1]

		// Estimate metrics based on score (simplified)
		predictedMetrics := qp.estimateMetricsFromScore(predictedScore, current)

		confidence := math.Max(0.5, 1.0 - (float64(i) * 0.05)) // Decrease confidence over time

		forecast = append(forecast, QualityForecastPoint{
			Timestamp:       time.Now().Add(time.Duration(i) * 24 * time.Hour),
			PredictedMetrics: predictedMetrics,
			Confidence:       confidence,
		})
	}

	return forecast
}

// estimateMetricsFromScore estimates quality metrics from an overall score.
func (qp *QualityPredictor) estimateMetricsFromScore(
	score float64,
	baseline *quality.QualityMetrics,
) *quality.QualityMetrics {
	// Adjust all metrics proportionally
	baselineScore := qp.calculateQualityScore(baseline)
	if baselineScore == 0 {
		baselineScore = 0.01 // Avoid division by zero
	}
	ratio := score / baselineScore

	return &quality.QualityMetrics{
		FreshnessScore:   math.Min(1.0, baseline.FreshnessScore*ratio),
		AccuracyScore:    math.Min(1.0, baseline.AccuracyScore*ratio),
		ConsistencyScore: math.Min(1.0, baseline.ConsistencyScore*ratio),
		ValidityScore:    math.Min(1.0, baseline.ValidityScore*ratio),
		ValidationStatus: baseline.ValidationStatus,
	}
}

// predictFutureMetrics predicts future quality metrics.
func (qp *QualityPredictor) predictFutureMetrics(
	current *quality.QualityMetrics,
	history []QualityHistoryPoint,
	trend string,
) *quality.QualityMetrics {
	if len(history) < 2 {
		return current
	}

	// Simple prediction based on trend
	futureScore := qp.calculateQualityScore(current)

	switch trend {
	case "improving":
		futureScore = math.Min(1.0, futureScore+0.05)
	case "degrading":
		futureScore = math.Max(0.0, futureScore-0.05)
	}

	return qp.estimateMetricsFromScore(futureScore, current)
}

// calculateRiskLevel calculates the risk level.
func (qp *QualityPredictor) calculateRiskLevel(
	metrics *quality.QualityMetrics,
	anomalyDetected bool,
	trend string,
) string {
	score := qp.calculateQualityScore(metrics)

	if anomalyDetected && score < 0.5 {
		return "critical"
	}
	if anomalyDetected || (trend == "degrading" && score < 0.6) {
		return "high"
	}
	if trend == "degrading" || score < 0.7 {
		return "medium"
	}
	return "low"
}

// generateRecommendations generates recommendations based on quality analysis.
func (qp *QualityPredictor) generateRecommendations(
	metrics *quality.QualityMetrics,
	trend string,
	anomalyDetected bool,
) []string {
	var recommendations []string

	if metrics.FreshnessScore < 0.8 {
		recommendations = append(recommendations, "Consider refreshing data more frequently")
	}
	if metrics.CompletenessScore < 0.9 {
		recommendations = append(recommendations, "Investigate missing data sources or ETL failures")
	}
	if metrics.AccuracyScore < 0.85 {
		recommendations = append(recommendations, "Review data validation rules and transformation logic")
	}
	if metrics.ConsistencyScore < 0.8 {
		recommendations = append(recommendations, "Check for data synchronization issues across systems")
	}
	if trend == "degrading" {
		recommendations = append(recommendations, "Quality is degrading - investigate root cause")
	}
	if anomalyDetected {
		recommendations = append(recommendations, "Anomaly detected - review recent changes to data pipeline")
	}
	// Check SLOs if available
	if len(metrics.SLOs) > 0 {
		for _, slo := range metrics.SLOs {
			if slo.Status == "violated" {
				recommendations = append(recommendations, fmt.Sprintf("SLO '%s' violated - take immediate action to restore quality", slo.Name))
			}
		}
	}

	if len(recommendations) == 0 {
		recommendations = append(recommendations, "Quality metrics are healthy - continue monitoring")
	}

	return recommendations
}

