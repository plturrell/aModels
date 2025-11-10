package analytics

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"ai_benchmarks/pkg/localai"
	"ai_benchmarks/services/shared/pkg/domain"
)

// PredictiveAnalytics provides predictive analytics and forecasting.
// Phase 9.4: Enhanced with domain-aware predictions for domain-specific forecasting.
type PredictiveAnalytics struct {
	logger         *log.Logger
	domainDetector *domain.Detector // Phase 9.4: Domain detector for domain predictions
	localaiURL     string           // Phase 9.4: LocalAI URL for domain configs
	llmClient      *localai.Client
	llmModel       string
	clock          func() time.Time
}

func (pa *PredictiveAnalytics) now() time.Time {
	if pa != nil && pa.clock != nil {
		return pa.clock()
	}
	return time.Now()
}

func (pa *PredictiveAnalytics) logInfo(msg string, kv ...any) {
	pa.logWithLevel("INFO", msg, kv...)
}

func (pa *PredictiveAnalytics) logWarn(msg string, kv ...any) {
	pa.logWithLevel("WARN", msg, kv...)
}

func (pa *PredictiveAnalytics) logDebug(msg string, kv ...any) {
	pa.logWithLevel("DEBUG", msg, kv...)
}

func (pa *PredictiveAnalytics) logWithLevel(level string, msg string, kv ...any) {
	if pa == nil || pa.logger == nil {
		return
	}
	pa.logger.Printf("PredictiveAnalytics %s: %s%s", level, msg, formatKeyValues(kv))
}

func formatKeyValues(kv []any) string {
	if len(kv) == 0 {
		return ""
	}
	var builder strings.Builder
	for i := 0; i < len(kv); i += 2 {
		key, ok := kv[i].(string)
		if !ok {
			continue
		}
		builder.WriteRune(' ')
		builder.WriteString(key)
		builder.WriteRune('=')
		var value any = "<missing>"
		if i+1 < len(kv) {
			value = kv[i+1]
		}
		builder.WriteString(fmt.Sprintf("%v", value))
	}
	return builder.String()
}

func keysFromMap(values map[string]any) string {
	if len(values) == 0 {
		return ""
	}
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return strings.Join(keys, ",")
}

// NewPredictiveAnalytics creates a new predictive analytics service.
func NewPredictiveAnalytics(logger *log.Logger) *PredictiveAnalytics {
	localaiURL := os.Getenv("LOCALAI_URL")
	llmModel := os.Getenv("PREDICTIVE_ANALYTICS_LLM_MODEL")
	if llmModel == "" {
		llmModel = "phi-3.5-mini"
	}
	pa := &PredictiveAnalytics{
		logger:     logger,
		localaiURL: localaiURL,
		llmModel:   llmModel,
		clock:      time.Now,
	}

	if localaiURL != "" {
		detector := domain.NewDetector(localaiURL, logger)
		if err := detector.LoadDomains(context.Background()); err != nil && !errors.Is(err, domain.ErrNoDomains) {
			pa.logWarn("failed initial domain sync", "error", err)
		} else {
			pa.domainDetector = detector
			pa.logInfo("initialized domain detector", "localai_url", localaiURL)
		}
		pa.llmClient = localai.NewClient(localaiURL)
	}

	return pa
}

// PredictDataQualityIssues predicts data quality issues before they occur.
func (pa *PredictiveAnalytics) PredictDataQualityIssues(
	ctx context.Context,
	currentMetrics map[string]any,
	historicalTrends []map[string]any,
) (*QualityPrediction, error) {
	pa.logInfo(
		"predicting data quality issues",
		"historical_count", len(historicalTrends),
		"current_metric_keys", keysFromMap(currentMetrics),
	)

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
			pa.logDebug("detected entropy decline", "risk_level", prediction.RiskLevel)
		}

		if trendAnalysis.KLDivergenceIncreasing {
			prediction.PredictedIssues = append(prediction.PredictedIssues,
				"KL divergence increasing - data type distribution deviating")
			prediction.RiskLevel = "high"
			pa.logDebug("detected kl divergence increase", "risk_level", prediction.RiskLevel)
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
			pa.logDebug("metadata entropy below threshold", "entropy", entropy, "risk_level", prediction.RiskLevel)
		}
	}

	prediction.Confidence = 0.75
	prediction.PredictedAt = pa.now()

	if pa.llmClient != nil {
		pa.logDebug("enhancing quality prediction with llm", "model", pa.llmModel)
		if err := pa.enhanceQualityPredictionWithLLM(ctx, prediction, currentMetrics, historicalTrends); err != nil {
			pa.logWarn("localai quality refinement failed", "error", err)
		}
	}

	return prediction, nil
}

// RecommendExtractionStrategy recommends optimal extraction strategies.
func (pa *PredictiveAnalytics) RecommendExtractionStrategy(
	ctx context.Context,
	projectID string,
	systemID string,
	historicalData map[string]any,
) (*ExtractionRecommendation, error) {
	pa.logInfo(
		"recommending extraction strategy",
		"project_id", projectID,
		"system_id", systemID,
	)

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
			pa.logDebug("data volume exceeds batch threshold", "data_volume", dataVolume)
		} else {
			recommendation.Strategy = "streaming"
			recommendation.Recommendations = append(recommendation.Recommendations,
				"Use streaming processing for real-time updates")
			pa.logDebug("data volume favors streaming", "data_volume", dataVolume)
		}
	}

	// Check schema complexity
	if schemaComplexity, ok := historicalData["schema_complexity"].(float64); ok {
		if schemaComplexity > 0.8 {
			recommendation.Priority = "high"
			recommendation.Recommendations = append(recommendation.Recommendations,
				"High schema complexity - use advanced extraction features")
			pa.logDebug("schema complexity high", "schema_complexity", schemaComplexity)
		}
	}

	if pa.llmClient != nil {
		pa.logDebug("enhancing extraction recommendation with llm", "model", pa.llmModel)
		if err := pa.enhanceExtractionRecommendationWithLLM(ctx, recommendation, historicalData); err != nil {
			pa.logWarn("localai extraction refinement failed", "error", err)
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
	pa.logInfo(
		"forecasting training data needs",
		"pattern_count", len(currentCoverage),
		"growth_rate", growthRate,
		"time_horizon_days", timeHorizonDays,
	)

	forecast := &TrainingDataForecast{
		CurrentCoverage:    currentCoverage,
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
			pa.logDebug("identified low coverage pattern", "pattern", pattern, "count", count)
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
	pa.logInfo(
		"detecting anomalies",
		"current_count", len(currentPatterns),
		"historical_count", len(historicalPatterns),
	)

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
					Pattern:    pattern,
					Type:       "deviation",
					Severity:   "medium",
					DetectedAt: pa.now(),
				}
				detection.Anomalies = append(detection.Anomalies, anomaly)
			}
		}
	}

	if len(detection.Anomalies) > 0 {
		detection.Severity = "medium"
		pa.logDebug("anomalies detected", "count", len(detection.Anomalies))
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
	pa.logInfo(
		"predicting domain performance",
		"domain_id", domainID,
		"time_horizon_days", timeHorizonDays,
	)

	prediction := &DomainPerformancePrediction{
		DomainID:        domainID,
		TimeHorizon:     timeHorizonDays,
		PredictedAt:     pa.now(),
		Metrics:         make(map[string]float64),
		Trends:          []string{},
		Recommendations: []string{},
	}

	// Get domain config for context
	domainConfig, hasConfig := pa.getDomainConfig(domainID)

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
					pa.logDebug("accuracy improving", "trend", trend)
				} else if trend < 0 {
					prediction.Trends = append(prediction.Trends, "Accuracy declining")
					prediction.Recommendations = append(prediction.Recommendations,
						"Consider retraining model or adjusting domain configuration")
					pa.logDebug("accuracy declining", "trend", trend)
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
					pa.logDebug("latency exceeds threshold", "predicted_latency_ms", predictedLat)
				}
			}
		}
	}

	// Adjust predictions based on domain characteristics
	if hasConfig {
		switch domainConfig.Layer {
		case "data":
			// Data layer: prioritize latency
			if lat, ok := prediction.Metrics["latency_ms"]; ok && lat > 1000 {
				prediction.Recommendations = append(prediction.Recommendations,
					"Data layer requires low latency - consider smaller model")
				pa.logDebug("data layer latency recommendation", "latency_ms", prediction.Metrics["latency_ms"])
			}
		case "business":
			// Business layer: prioritize accuracy
			if acc, ok := prediction.Metrics["accuracy"]; ok && acc < 0.8 {
				prediction.Recommendations = append(prediction.Recommendations,
					"Business layer requires high accuracy - consider model enhancement")
				pa.logDebug("business layer accuracy recommendation", "accuracy", prediction.Metrics["accuracy"])
			}
		}
	}

	if pa.llmClient != nil {
		pa.logDebug("enhancing domain performance with llm", "model", pa.llmModel, "domain_id", domainID)
		if err := pa.enhanceDomainPerformanceWithLLM(ctx, prediction, domainID, historicalMetrics, hasConfig); err != nil {
			pa.logWarn("localai domain performance refinement failed", "error", err, "domain_id", domainID)
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
	pa.logInfo("predicting domain data quality", "domain_id", domainID, "historical_count", len(historicalQuality))

	prediction := &DomainQualityPrediction{
		DomainID:        domainID,
		PredictedAt:     pa.now(),
		RiskLevel:       "low",
		Confidence:      0.7,
		Issues:          []string{},
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
					pa.logDebug("completeness declining", "domain_id", domainID, "risk_level", prediction.RiskLevel)
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
					pa.logDebug("consistency declining", "domain_id", domainID, "risk_level", prediction.RiskLevel)
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
	pa.logInfo("predicting domain training needs", "domain_id", domainID, "pattern_count", len(currentCoverage))

	needs := &DomainTrainingNeeds{
		DomainID:          domainID,
		PredictedAt:       pa.now(),
		CurrentCoverage:   currentCoverage,
		ProjectedCoverage: make(map[string]int),
		PriorityAreas:     []string{},
		Recommendations:   []string{},
	}

	// Get domain config for context
	domainConfig, hasConfig := pa.getDomainConfig(domainID)

	// Project coverage needs
	for pattern, count := range currentCoverage {
		projected := int(float64(count) * (1.0 + growthRate))
		needs.ProjectedCoverage[pattern] = projected

		// Identify priority areas
		if count < 50 {
			needs.PriorityAreas = append(needs.PriorityAreas, pattern)
			needs.Recommendations = append(needs.Recommendations,
				fmt.Sprintf("Collect more training data for pattern: %s", pattern))
			pa.logDebug("low coverage domain pattern", "pattern", pattern, "count", count)
		}
	}

	// Adjust based on domain characteristics
	if hasConfig {
		if len(domainConfig.Keywords) > 10 {
			// Semantic-rich domain: needs more diverse training data
			needs.Recommendations = append(needs.Recommendations,
				"Semantic-rich domain - focus on diverse training examples")
			pa.logDebug("semantic rich domain detected", "keyword_count", len(domainConfig.Keywords))
		}
	}

	return needs, nil
}

// getDomainConfig fetches domain configuration from LocalAI.
func (pa *PredictiveAnalytics) getDomainConfig(domainID string) (domain.DomainConfig, bool) {
	if pa.domainDetector == nil {
		return domain.DomainConfig{}, false
	}
	cfg, ok := pa.domainDetector.Config(domainID)
	return cfg, ok
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
	PredictedIssues []string  `json:"predicted_issues"`
	RiskLevel       string    `json:"risk_level"`
	Confidence      float64   `json:"confidence"`
	TimeHorizon     string    `json:"time_horizon"`
	PredictedAt     time.Time `json:"predicted_at"`
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
	sums := make(map[string]float64)
	counts := make(map[string]int)

	for _, pattern := range patterns {
		for key, value := range pattern {
			if num, ok := value.(float64); ok {
				sums[key] += num
				counts[key]++
			}
		}
	}

	baseline := make(map[string]float64)
	for key, sum := range sums {
		if count := counts[key]; count > 0 {
			baseline[key] = sum / float64(count)
		}
	}

	return baseline
}

func (pa *PredictiveAnalytics) isAnomalous(pattern map[string]any, baseline map[string]float64) bool {
	threshold := 0.3 // 30% deviation

	for key, value := range pattern {
		num, ok := value.(float64)
		if !ok {
			continue
		}

		baselineValue, exists := baseline[key]
		if !exists {
			continue
		}

		var deviation float64
		if baselineValue == 0 {
			deviation = math.Abs(num)
		} else {
			deviation = math.Abs((num - baselineValue) / baselineValue)
		}

		if deviation > threshold {
			return true
		}
	}

	return false
}

func (pa *PredictiveAnalytics) enhanceQualityPredictionWithLLM(
	ctx context.Context,
	prediction *QualityPrediction,
	currentMetrics map[string]any,
	historicalTrends []map[string]any,
) error {
	if prediction == nil {
		return errors.New("nil quality prediction")
	}

	payload := map[string]any{
		"task":                "quality_prediction",
		"baseline_prediction": prediction,
		"current_metrics":     currentMetrics,
		"historical_trends":   historicalTrends,
	}

	var response struct {
		RiskLevel       string   `json:"risk_level"`
		Confidence      float64  `json:"confidence"`
		TimeHorizon     string   `json:"time_horizon"`
		PredictedIssues []string `json:"predicted_issues"`
		Notes           []string `json:"notes"`
	}

	if err := pa.invokeLLM(ctx, "You are a data quality risk analyst. Return JSON.", payload, &response); err != nil {
		return err
	}

	if response.RiskLevel != "" {
		prediction.RiskLevel = response.RiskLevel
	}
	if response.Confidence > 0 {
		prediction.Confidence = response.Confidence
	}
	if response.TimeHorizon != "" {
		prediction.TimeHorizon = response.TimeHorizon
	}
	if len(response.PredictedIssues) > 0 {
		prediction.PredictedIssues = mergeStringSlices(prediction.PredictedIssues, response.PredictedIssues)
	}
	if len(response.Notes) > 0 {
		prediction.PredictedIssues = mergeStringSlices(prediction.PredictedIssues, response.Notes)
	}

	return nil
}

func (pa *PredictiveAnalytics) enhanceExtractionRecommendationWithLLM(
	ctx context.Context,
	recommendation *ExtractionRecommendation,
	historicalData map[string]any,
) error {
	if recommendation == nil {
		return errors.New("nil extraction recommendation")
	}

	payload := map[string]any{
		"task":                    "extraction_strategy",
		"baseline_recommendation": recommendation,
		"historical_data":         historicalData,
	}

	var response struct {
		Strategy        string   `json:"strategy"`
		Priority        string   `json:"priority"`
		Recommendations []string `json:"recommendations"`
		Confidence      float64  `json:"confidence"`
	}

	if err := pa.invokeLLM(ctx, "You help choose data extraction strategies. Return JSON only.", payload, &response); err != nil {
		return err
	}

	if response.Strategy != "" {
		recommendation.Strategy = response.Strategy
	}
	if response.Priority != "" {
		recommendation.Priority = response.Priority
	}
	if response.Confidence > 0 {
		recommendation.Confidence = response.Confidence
	}
	if len(response.Recommendations) > 0 {
		recommendation.Recommendations = mergeStringSlices(recommendation.Recommendations, response.Recommendations)
	}

	return nil
}

func (pa *PredictiveAnalytics) enhanceDomainPerformanceWithLLM(
	ctx context.Context,
	prediction *DomainPerformancePrediction,
	domainID string,
	historicalMetrics []map[string]any,
	hasConfig bool,
) error {
	if prediction == nil {
		return errors.New("nil domain performance prediction")
	}

	payload := map[string]any{
		"task":                "domain_performance_forecast",
		"domain_id":           domainID,
		"baseline_prediction": prediction,
		"historical_metrics":  historicalMetrics,
		"has_domain_config":   hasConfig,
	}

	var response struct {
		Metrics         map[string]float64 `json:"metrics"`
		Trends          []string           `json:"trends"`
		Recommendations []string           `json:"recommendations"`
	}

	if err := pa.invokeLLM(ctx, "You forecast model performance for domains. Respond with JSON.", payload, &response); err != nil {
		return err
	}

	if len(response.Metrics) > 0 {
		if prediction.Metrics == nil {
			prediction.Metrics = make(map[string]float64)
		}
		for k, v := range response.Metrics {
			prediction.Metrics[k] = v
		}
	}
	if len(response.Trends) > 0 {
		prediction.Trends = mergeStringSlices(prediction.Trends, response.Trends)
	}
	if len(response.Recommendations) > 0 {
		prediction.Recommendations = mergeStringSlices(prediction.Recommendations, response.Recommendations)
	}

	return nil
}

func (pa *PredictiveAnalytics) invokeLLM(
	ctx context.Context,
	systemPrompt string,
	payload any,
	out any,
) error {
	if pa.llmClient == nil {
		return errors.New("llm client not configured")
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}

	request := &localai.ChatRequest{
		Model: pa.llmModel,
		Messages: []localai.Message{
			{Role: "system", Content: systemPrompt},
			{Role: "user", Content: string(body)},
		},
		Temperature: 0.2,
		MaxTokens:   512,
	}

	timeoutCtx, cancel := context.WithTimeout(ctx, domain.LLMRequestTimeout())
	defer cancel()

	response, err := pa.llmClient.ChatCompletion(timeoutCtx, request)
	if err != nil {
		return fmt.Errorf("localai chat completion: %w", err)
	}

	content := strings.TrimSpace(response.GetContent())
	if content == "" {
		return errors.New("empty response from LocalAI")
	}

	if err := json.Unmarshal([]byte(content), out); err != nil {
		return fmt.Errorf("decode LocalAI response: %w", err)
	}

	return nil
}

func mergeStringSlices(base []string, additions []string) []string {
	existing := make(map[string]struct{}, len(base))
	for _, item := range base {
		existing[item] = struct{}{}
	}
	for _, item := range additions {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := existing[item]; ok {
			continue
		}
		base = append(base, item)
		existing[item] = struct{}{}
	}
	return base
}
