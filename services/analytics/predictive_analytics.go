package analytics

import (
	"context"
	"fmt"
	"log"
	"time"
)

// PredictiveAnalytics provides predictive analytics and forecasting.
type PredictiveAnalytics struct {
	logger *log.Logger
}

// NewPredictiveAnalytics creates a new predictive analytics service.
func NewPredictiveAnalytics(logger *log.Logger) *PredictiveAnalytics {
	return &PredictiveAnalytics{logger: logger}
}

// PredictDataQualityIssues predicts data quality issues before they occur.
func (pa *PredictiveAnalytics) PredictDataQualityIssues(
	ctx context.Context,
	currentMetrics map[string]any,
	historicalTrends []map[string]any,
) (*QualityPrediction, error) {
	pa.logger.Println("Predicting data quality issues...")
	
	prediction := &QualityPrediction{
		PredictedIssues: []string{},
		RiskLevel:       "low",
		Confidence:      0.0,
		TimeHorizon:     "7 days",
	}
	
	// Analyze trends
	if len(historicalTrends) > 0 {
		// Detect declining trends
		trendAnalysis := pa.analyzeTrends(historicalTrends)
		
		if trendAnalysis.EntropyDeclining {
			prediction.PredictedIssues = append(prediction.PredictedIssues, 
				"Metadata entropy declining - potential schema quality degradation")
			prediction.RiskLevel = "medium"
		}
		
		if trendAnalysis.KLDivergenceIncreasing {
			prediction.PredictedIssues = append(prediction.PredictedIssues,
				"KL divergence increasing - data type distribution deviating")
			prediction.RiskLevel = "high"
		}
	}
	
	// Check current metrics against thresholds
	if entropy, ok := currentMetrics["metadata_entropy"].(float64); ok {
		if entropy < 2.0 {
			prediction.PredictedIssues = append(prediction.PredictedIssues,
				"Low metadata entropy - limited schema diversity")
			if prediction.RiskLevel == "low" {
				prediction.RiskLevel = "medium"
			}
		}
	}
	
	prediction.Confidence = 0.75
	prediction.PredictedAt = time.Now()
	
	return prediction, nil
}

// RecommendExtractionStrategy recommends optimal extraction strategies.
func (pa *PredictiveAnalytics) RecommendExtractionStrategy(
	ctx context.Context,
	projectID string,
	systemID string,
	historicalData map[string]any,
) (*ExtractionRecommendation, error) {
	pa.logger.Printf("Recommending extraction strategy for project=%s, system=%s", projectID, systemID)
	
	recommendation := &ExtractionRecommendation{
		Strategy:        "standard",
		Priority:        "medium",
		Recommendations: []string{},
		Confidence:      0.7,
	}
	
	// Analyze historical data patterns
	if dataVolume, ok := historicalData["data_volume"].(float64); ok {
		if dataVolume > 1000000 {
			recommendation.Strategy = "batch"
			recommendation.Recommendations = append(recommendation.Recommendations,
				"Use batch processing for large data volumes")
		} else {
			recommendation.Strategy = "streaming"
			recommendation.Recommendations = append(recommendation.Recommendations,
				"Use streaming processing for real-time updates")
		}
	}
	
	// Check schema complexity
	if schemaComplexity, ok := historicalData["schema_complexity"].(float64); ok {
		if schemaComplexity > 0.8 {
			recommendation.Priority = "high"
			recommendation.Recommendations = append(recommendation.Recommendations,
				"High schema complexity - use advanced extraction features")
		}
	}
	
	return recommendation, nil
}

// ForecastTrainingDataNeeds forecasts training data requirements.
func (pa *PredictiveAnalytics) ForecastTrainingDataNeeds(
	ctx context.Context,
	currentCoverage map[string]int,
	growthRate float64,
	timeHorizonDays int,
) (*TrainingDataForecast, error) {
	pa.logger.Println("Forecasting training data needs...")
	
	forecast := &TrainingDataForecast{
		CurrentCoverage: currentCoverage,
		ForecastedCoverage: make(map[string]int),
		RecommendedActions: []string{},
	}
	
	// Project coverage based on growth rate
	for pattern, count := range currentCoverage {
		projectedCount := int(float64(count) * (1.0 + growthRate*float64(timeHorizonDays)/30.0))
		forecast.ForecastedCoverage[pattern] = projectedCount
		
		// Recommend actions if coverage is insufficient
		if count < 100 {
			forecast.RecommendedActions = append(forecast.RecommendedActions,
				fmt.Sprintf("Collect more data for pattern: %s (current: %d)", pattern, count))
		}
	}
	
	return forecast, nil
}

// DetectAnomalies detects anomalies in patterns.
func (pa *PredictiveAnalytics) DetectAnomalies(
	ctx context.Context,
	currentPatterns []map[string]any,
	historicalPatterns []map[string]any,
) (*AnomalyDetection, error) {
	pa.logger.Println("Detecting anomalies in patterns...")
	
	detection := &AnomalyDetection{
		Anomalies: []Anomaly{},
		Severity:  "low",
	}
	
	// Simple anomaly detection: compare current vs historical
	if len(historicalPatterns) > 0 {
		// Calculate baseline statistics
		baseline := pa.calculateBaseline(historicalPatterns)
		
		// Check for deviations
		for _, pattern := range currentPatterns {
			if pa.isAnomalous(pattern, baseline) {
				anomaly := Anomaly{
					Pattern:   pattern,
					Type:      "deviation",
					Severity:  "medium",
					DetectedAt: time.Now(),
				}
				detection.Anomalies = append(detection.Anomalies, anomaly)
			}
		}
	}
	
	if len(detection.Anomalies) > 0 {
		detection.Severity = "medium"
	}
	
	return detection, nil
}

// QualityPrediction represents a prediction about data quality.
type QualityPrediction struct {
	PredictedIssues []string    `json:"predicted_issues"`
	RiskLevel       string      `json:"risk_level"`
	Confidence      float64     `json:"confidence"`
	TimeHorizon     string      `json:"time_horizon"`
	PredictedAt     time.Time   `json:"predicted_at"`
}

// ExtractionRecommendation represents a recommendation for extraction strategy.
type ExtractionRecommendation struct {
	Strategy        string   `json:"strategy"`
	Priority        string   `json:"priority"`
	Recommendations []string `json:"recommendations"`
	Confidence      float64  `json:"confidence"`
}

// TrainingDataForecast represents a forecast of training data needs.
type TrainingDataForecast struct {
	CurrentCoverage    map[string]int `json:"current_coverage"`
	ForecastedCoverage map[string]int `json:"forecasted_coverage"`
	RecommendedActions []string       `json:"recommended_actions"`
}

// AnomalyDetection represents detected anomalies.
type AnomalyDetection struct {
	Anomalies []Anomaly `json:"anomalies"`
	Severity  string    `json:"severity"`
}

// Anomaly represents a single anomaly.
type Anomaly struct {
	Pattern    map[string]any `json:"pattern"`
	Type       string         `json:"type"`
	Severity   string         `json:"severity"`
	DetectedAt time.Time      `json:"detected_at"`
}

// Helper methods
func (pa *PredictiveAnalytics) analyzeTrends(trends []map[string]any) *TrendAnalysis {
	analysis := &TrendAnalysis{
		EntropyDeclining:       false,
		KLDivergenceIncreasing: false,
	}
	
	if len(trends) < 2 {
		return analysis
	}
	
	// Simple trend detection
	first := trends[0]
	last := trends[len(trends)-1]
	
	if firstEntropy, ok1 := first["metadata_entropy"].(float64); ok1 {
		if lastEntropy, ok2 := last["metadata_entropy"].(float64); ok2 {
			if lastEntropy < firstEntropy {
				analysis.EntropyDeclining = true
			}
		}
	}
	
	if firstKL, ok1 := first["kl_divergence"].(float64); ok1 {
		if lastKL, ok2 := last["kl_divergence"].(float64); ok2 {
			if lastKL > firstKL {
				analysis.KLDivergenceIncreasing = true
			}
		}
	}
	
	return analysis
}

type TrendAnalysis struct {
	EntropyDeclining       bool
	KLDivergenceIncreasing bool
}

func (pa *PredictiveAnalytics) calculateBaseline(patterns []map[string]any) map[string]float64 {
	baseline := make(map[string]float64)
	
	// Calculate average values
	for _, pattern := range patterns {
		for key, value := range pattern {
			if num, ok := value.(float64); ok {
				baseline[key] = (baseline[key] + num) / 2.0
			}
		}
	}
	
	return baseline
}

func (pa *PredictiveAnalytics) isAnomalous(pattern map[string]any, baseline map[string]float64) bool {
	threshold := 0.3 // 30% deviation
	
	for key, value := range pattern {
		if num, ok := value.(float64); ok {
			if baselineValue, exists := baseline[key]; exists {
				deviation := abs(num - baselineValue) / baselineValue
				if deviation > threshold {
					return true
				}
			}
		}
	}
	
	return false
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

