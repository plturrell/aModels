package analytics

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// PredictiveAnalytics provides predictive analytics and forecasting.
// Phase 9.4: Enhanced with domain-aware predictions for domain-specific forecasting.
type PredictiveAnalytics struct {
	logger        *log.Logger
	domainDetector *DomainDetector // Phase 9.4: Domain detector for domain predictions
	localaiURL    string           // Phase 9.4: LocalAI URL for domain configs
}

// DomainDetector is a placeholder - would import from extract service
type DomainDetector struct {
	localaiURL string
	logger     *log.Logger
}

// NewDomainDetector creates a domain detector (placeholder).
func NewDomainDetector(localaiURL string, logger *log.Logger) *DomainDetector {
	return &DomainDetector{
		localaiURL: localaiURL,
		logger:     logger,
	}
}

// NewPredictiveAnalytics creates a new predictive analytics service.
func NewPredictiveAnalytics(logger *log.Logger) *PredictiveAnalytics {
	localaiURL := os.Getenv("LOCALAI_URL")
	var domainDetector *DomainDetector
	if localaiURL != "" {
		domainDetector = NewDomainDetector(localaiURL, logger)
	}
	
	return &PredictiveAnalytics{
		logger:        logger,
		domainDetector: domainDetector, // Phase 9.4: Domain detector
		localaiURL:    localaiURL,
	}
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

// PredictDomainPerformance predicts performance metrics for a domain.
// Phase 9.4: Domain-specific performance prediction.
func (pa *PredictiveAnalytics) PredictDomainPerformance(
	ctx context.Context,
	domainID string,
	historicalMetrics []map[string]any,
	timeHorizonDays int,
) (*DomainPerformancePrediction, error) {
	pa.logger.Printf("Predicting performance for domain %s (horizon: %d days)", domainID, timeHorizonDays)
	
	prediction := &DomainPerformancePrediction{
		DomainID:      domainID,
		TimeHorizon:   timeHorizonDays,
		PredictedAt:   time.Now(),
		Metrics:       make(map[string]float64),
		Trends:        []string{},
		Recommendations: []string{},
	}
	
	// Get domain config for context
	domainConfig := pa.getDomainConfig(domainID)
	
	// Analyze historical trends
	if len(historicalMetrics) >= 2 {
		first := historicalMetrics[0]
		last := historicalMetrics[len(historicalMetrics)-1]
		
		// Predict accuracy trend
		if firstAcc, ok1 := first["accuracy"].(float64); ok1 {
			if lastAcc, ok2 := last["accuracy"].(float64); ok2 {
				trend := (lastAcc - firstAcc) / float64(len(historicalMetrics)-1)
				predictedAcc := lastAcc + trend*float64(timeHorizonDays)
				prediction.Metrics["accuracy"] = predictedAcc
				
				if trend > 0 {
					prediction.Trends = append(prediction.Trends, "Accuracy improving")
				} else if trend < 0 {
					prediction.Trends = append(prediction.Trends, "Accuracy declining")
					prediction.Recommendations = append(prediction.Recommendations,
						"Consider retraining model or adjusting domain configuration")
				}
			}
		}
		
		// Predict latency trend
		if firstLat, ok1 := first["latency_ms"].(float64); ok1 {
			if lastLat, ok2 := last["latency_ms"].(float64); ok2 {
				trend := (lastLat - firstLat) / float64(len(historicalMetrics)-1)
				predictedLat := lastLat + trend*float64(timeHorizonDays)
				prediction.Metrics["latency_ms"] = predictedLat
				
				if predictedLat > 2000 {
					prediction.Recommendations = append(prediction.Recommendations,
						"High latency predicted - consider model optimization")
				}
			}
		}
	}
	
	// Adjust predictions based on domain characteristics
	if domainConfig != nil {
		layer := domainConfig["layer"].(string)
		if layer == "data" {
			// Data layer: prioritize latency
			if lat, ok := prediction.Metrics["latency_ms"]; ok && lat > 1000 {
				prediction.Recommendations = append(prediction.Recommendations,
					"Data layer requires low latency - consider smaller model")
			}
		} else if layer == "business" {
			// Business layer: prioritize accuracy
			if acc, ok := prediction.Metrics["accuracy"]; ok && acc < 0.8 {
				prediction.Recommendations = append(prediction.Recommendations,
					"Business layer requires high accuracy - consider model enhancement")
			}
		}
	}
	
	return prediction, nil
}

// PredictDomainDataQuality predicts data quality for a domain.
// Phase 9.4: Domain-specific data quality prediction.
func (pa *PredictiveAnalytics) PredictDomainDataQuality(
	ctx context.Context,
	domainID string,
	currentQuality map[string]any,
	historicalQuality []map[string]any,
) (*DomainQualityPrediction, error) {
	pa.logger.Printf("Predicting data quality for domain %s", domainID)
	
	prediction := &DomainQualityPrediction{
		DomainID:      domainID,
		PredictedAt:   time.Now(),
		RiskLevel:    "low",
		Confidence:    0.7,
		Issues:        []string{},
		Recommendations: []string{},
	}
	
	// Analyze trends
	if len(historicalQuality) >= 2 {
		first := historicalQuality[0]
		last := historicalQuality[len(historicalQuality)-1]
		
		// Check completeness trend
		if firstComp, ok1 := first["completeness"].(float64); ok1 {
			if lastComp, ok2 := last["completeness"].(float64); ok2 {
				if lastComp < firstComp {
					prediction.RiskLevel = "medium"
					prediction.Issues = append(prediction.Issues,
						"Data completeness declining")
					prediction.Recommendations = append(prediction.Recommendations,
						"Increase data collection or validation")
				}
			}
		}
		
		// Check consistency trend
		if firstCons, ok1 := first["consistency"].(float64); ok1 {
			if lastCons, ok2 := last["consistency"].(float64); ok2 {
				if lastCons < firstCons {
					prediction.RiskLevel = "high"
					prediction.Issues = append(prediction.Issues,
						"Data consistency declining")
					prediction.Recommendations = append(prediction.Recommendations,
						"Review data quality rules and validation")
				}
			}
		}
	}
	
	return prediction, nil
}

// PredictDomainTrainingNeeds predicts training data needs for a domain.
// Phase 9.4: Domain-specific training needs prediction.
func (pa *PredictiveAnalytics) PredictDomainTrainingNeeds(
	ctx context.Context,
	domainID string,
	currentCoverage map[string]int,
	growthRate float64,
) (*DomainTrainingNeeds, error) {
	pa.logger.Printf("Predicting training needs for domain %s", domainID)
	
	needs := &DomainTrainingNeeds{
		DomainID:           domainID,
		PredictedAt:        time.Now(),
		CurrentCoverage:    currentCoverage,
		ProjectedCoverage:  make(map[string]int),
		PriorityAreas:      []string{},
		Recommendations:    []string{},
	}
	
	// Get domain config for context
	domainConfig := pa.getDomainConfig(domainID)
	
	// Project coverage needs
	for pattern, count := range currentCoverage {
		projected := int(float64(count) * (1.0 + growthRate))
		needs.ProjectedCoverage[pattern] = projected
		
		// Identify priority areas
		if count < 50 {
			needs.PriorityAreas = append(needs.PriorityAreas, pattern)
			needs.Recommendations = append(needs.Recommendations,
				fmt.Sprintf("Collect more training data for pattern: %s", pattern))
		}
	}
	
	// Adjust based on domain characteristics
	if domainConfig != nil {
		keywords := domainConfig["keywords"].([]string)
		if len(keywords) > 10 {
			// Semantic-rich domain: needs more diverse training data
			needs.Recommendations = append(needs.Recommendations,
				"Semantic-rich domain - focus on diverse training examples")
		}
	}
	
	return needs, nil
}

// getDomainConfig fetches domain configuration from LocalAI.
func (pa *PredictiveAnalytics) getDomainConfig(domainID string) map[string]any {
	if pa.localaiURL == "" {
		return nil
	}
	
	url := strings.TrimSuffix(pa.localaiURL, "/") + "/v1/domains"
	resp, err := http.Get(url)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	
	body, _ := io.ReadAll(resp.Body)
	var domainsData map[string]any
	if json.Unmarshal(body, &domainsData) != nil {
		return nil
	}
	
	domains := domainsData["domains"].(map[string]any)
	if domainInfo, ok := domains[domainID].(map[string]any); ok {
		config := domainInfo["config"].(map[string]any)
		return config
	}
	
	return nil
}

// DomainPerformancePrediction represents domain performance prediction.
type DomainPerformancePrediction struct {
	DomainID        string             `json:"domain_id"`
	TimeHorizon     int                `json:"time_horizon_days"`
	PredictedAt     time.Time          `json:"predicted_at"`
	Metrics         map[string]float64 `json:"predicted_metrics"`
	Trends          []string           `json:"trends"`
	Recommendations []string           `json:"recommendations"`
}

// DomainQualityPrediction represents domain quality prediction.
type DomainQualityPrediction struct {
	DomainID        string    `json:"domain_id"`
	PredictedAt     time.Time `json:"predicted_at"`
	RiskLevel       string    `json:"risk_level"`
	Confidence      float64   `json:"confidence"`
	Issues          []string  `json:"predicted_issues"`
	Recommendations []string  `json:"recommendations"`
}

// DomainTrainingNeeds represents domain training needs prediction.
type DomainTrainingNeeds struct {
	DomainID          string         `json:"domain_id"`
	PredictedAt       time.Time      `json:"predicted_at"`
	CurrentCoverage   map[string]int `json:"current_coverage"`
	ProjectedCoverage map[string]int `json:"projected_coverage"`
	PriorityAreas     []string       `json:"priority_areas"`
	Recommendations   []string       `json:"recommendations"`
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

