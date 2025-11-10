package analytics

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"testing"

	"ai_benchmarks/pkg/localai"
)

func TestPredictDataQualityIssuesWithLLMRefinement(t *testing.T) {
	llmResponse := map[string]any{
		"risk_level":       "high",
		"confidence":       0.9,
		"time_horizon":     "48 hours",
		"predicted_issues": []string{"LLM-detected drift"},
		"notes":            []string{"Escalate to data stewardship"},
	}

	server := newLocalAITestServer(t, llmResponse)
	defer server.Close()

	pa := &PredictiveAnalytics{
		logger:    log.New(log.Writer(), "", 0),
		llmClient: localai.NewClient(server.URL),
		llmModel:  "test-model",
	}

	prediction, err := pa.PredictDataQualityIssues(context.Background(), map[string]any{}, nil)
	if err != nil {
		t.Fatalf("PredictDataQualityIssues returned error: %v", err)
	}

	if prediction.RiskLevel != "high" {
		t.Fatalf("expected risk level high, got %s", prediction.RiskLevel)
	}
	if prediction.Confidence != 0.9 {
		t.Fatalf("expected confidence 0.9, got %f", prediction.Confidence)
	}
	if prediction.TimeHorizon != "48 hours" {
		t.Fatalf("expected time horizon 48 hours, got %s", prediction.TimeHorizon)
	}
	if !contains(prediction.PredictedIssues, "LLM-detected drift") {
		t.Fatalf("expected predicted issues to include LLM-detected drift, got %#v", prediction.PredictedIssues)
	}
	if !contains(prediction.PredictedIssues, "Escalate to data stewardship") {
		t.Fatalf("expected predicted issues to include escalation note, got %#v", prediction.PredictedIssues)
	}
}

func TestRecommendExtractionStrategyWithLLMRefinement(t *testing.T) {
	llmResponse := map[string]any{
		"strategy":        "batch",
		"priority":        "high",
		"confidence":      0.95,
		"recommendations": []string{"Enable partition pruning"},
	}

	server := newLocalAITestServer(t, llmResponse)
	defer server.Close()

	pa := &PredictiveAnalytics{
		logger:    log.New(log.Writer(), "", 0),
		llmClient: localai.NewClient(server.URL),
		llmModel:  "test-model",
	}

	recommendation, err := pa.RecommendExtractionStrategy(context.Background(), "project", "system", map[string]any{})
	if err != nil {
		t.Fatalf("RecommendExtractionStrategy returned error: %v", err)
	}

	if recommendation.Strategy != "batch" {
		t.Fatalf("expected strategy batch, got %s", recommendation.Strategy)
	}
	if recommendation.Priority != "high" {
		t.Fatalf("expected priority high, got %s", recommendation.Priority)
	}
	if recommendation.Confidence != 0.95 {
		t.Fatalf("expected confidence 0.95, got %f", recommendation.Confidence)
	}
	if !contains(recommendation.Recommendations, "Enable partition pruning") {
		t.Fatalf("expected recommendation from LLM, got %#v", recommendation.Recommendations)
	}
}

func TestPredictDomainPerformanceWithLLMRefinement(t *testing.T) {
	llmResponse := map[string]any{
		"metrics": map[string]float64{
			"accuracy":   0.93,
			"latency_ms": 1500,
		},
		"trends":          []string{"Accuracy stabilizing"},
		"recommendations": []string{"Review feature refresh cadence"},
	}

	server := newLocalAITestServer(t, llmResponse)
	defer server.Close()

	pa := &PredictiveAnalytics{
		logger:    log.New(log.Writer(), "", 0),
		llmClient: localai.NewClient(server.URL),
		llmModel:  "test-model",
	}

	historical := []map[string]any{
		{"accuracy": 0.9, "latency_ms": 1800.0},
		{"accuracy": 0.91, "latency_ms": 1700.0},
	}

	prediction, err := pa.PredictDomainPerformance(context.Background(), "domain-1", historical, 7)
	if err != nil {
		t.Fatalf("PredictDomainPerformance returned error: %v", err)
	}

	if val := prediction.Metrics["accuracy"]; val != 0.93 {
		t.Fatalf("expected accuracy override 0.93, got %f", val)
	}
	if val := prediction.Metrics["latency_ms"]; val != 1500 {
		t.Fatalf("expected latency override 1500, got %f", val)
	}
	if !contains(prediction.Trends, "Accuracy stabilizing") {
		t.Fatalf("expected LLM trend, got %#v", prediction.Trends)
	}
	if !contains(prediction.Recommendations, "Review feature refresh cadence") {
		t.Fatalf("expected LLM recommendation, got %#v", prediction.Recommendations)
	}
}

func TestPredictDomainPerformanceLLMFallback(t *testing.T) {
	server := newLocalAIInvalidJSONServer(t)
	defer server.Close()

	pa := &PredictiveAnalytics{
		logger:    log.New(log.Writer(), "", 0),
		llmClient: localai.NewClient(server.URL),
		llmModel:  "test-model",
	}

	historical := []map[string]any{
		{"accuracy": 0.8, "latency_ms": 1800.0},
		{"accuracy": 0.85, "latency_ms": 2000.0},
	}

	prediction, err := pa.PredictDomainPerformance(context.Background(), "domain-1", historical, 2)
	if err != nil {
		t.Fatalf("PredictDomainPerformance returned error: %v", err)
	}

	if val := prediction.Metrics["accuracy"]; !closeTo(val, 0.95) {
		t.Fatalf("expected heuristic accuracy ~0.95, got %f", val)
	}
	if val := prediction.Metrics["latency_ms"]; !closeTo(val, 2400) {
		t.Fatalf("expected heuristic latency ~2400, got %f", val)
	}
	if !contains(prediction.Recommendations, "High latency predicted - consider model optimization") {
		t.Fatalf("expected heuristic latency recommendation, got %#v", prediction.Recommendations)
	}
}

func TestForecastTrainingDataNeeds(t *testing.T) {
	pa := &PredictiveAnalytics{logger: log.New(log.Writer(), "", 0)}

	current := map[string]int{"patternA": 50, "patternB": 200}
	forecast, err := pa.ForecastTrainingDataNeeds(context.Background(), current, 0.1, 30)
	if err != nil {
		t.Fatalf("ForecastTrainingDataNeeds returned error: %v", err)
	}

	if val := forecast.ForecastedCoverage["patternA"]; val != 55 {
		t.Fatalf("expected projected coverage 55 for patternA, got %d", val)
	}
	if val := forecast.ForecastedCoverage["patternB"]; val != 220 {
		t.Fatalf("expected projected coverage 220 for patternB, got %d", val)
	}
	if !contains(forecast.RecommendedActions, "Collect more data for pattern: patternA (current: 50)") {
		t.Fatalf("expected recommendation for patternA, got %#v", forecast.RecommendedActions)
	}
}

func TestDetectAnomalies(t *testing.T) {
	pa := &PredictiveAnalytics{logger: log.New(log.Writer(), "", 0)}

	historical := []map[string]any{
		{"volume": 100.0},
		{"volume": 110.0},
	}
	current := []map[string]any{
		{"volume": 200.0},
	}

	detection, err := pa.DetectAnomalies(context.Background(), current, historical)
	if err != nil {
		t.Fatalf("DetectAnomalies returned error: %v", err)
	}

	if len(detection.Anomalies) != 1 {
		t.Fatalf("expected one anomaly, got %d", len(detection.Anomalies))
	}
	if detection.Severity != "medium" {
		t.Fatalf("expected severity medium, got %s", detection.Severity)
	}
}

func TestPredictDomainDataQuality(t *testing.T) {
	pa := &PredictiveAnalytics{logger: log.New(log.Writer(), "", 0)}

	historical := []map[string]any{
		{"completeness": 0.95, "consistency": 0.98},
		{"completeness": 0.9, "consistency": 0.85},
	}

	prediction, err := pa.PredictDomainDataQuality(context.Background(), "domain-1", map[string]any{}, historical)
	if err != nil {
		t.Fatalf("PredictDomainDataQuality returned error: %v", err)
	}

	if prediction.RiskLevel != "high" {
		t.Fatalf("expected risk level high due to consistency decline, got %s", prediction.RiskLevel)
	}
	if !contains(prediction.Issues, "Data consistency declining") {
		t.Fatalf("expected consistency issue, got %#v", prediction.Issues)
	}
	if !contains(prediction.Recommendations, "Review data quality rules and validation") {
		t.Fatalf("expected recommendation to review rules, got %#v", prediction.Recommendations)
	}
}

func TestPredictDomainTrainingNeeds(t *testing.T) {
	pa := &PredictiveAnalytics{logger: log.New(log.Writer(), "", 0)}

	current := map[string]int{"patternA": 20, "patternB": 80}
	needs, err := pa.PredictDomainTrainingNeeds(context.Background(), "domain-1", current, 0.5)
	if err != nil {
		t.Fatalf("PredictDomainTrainingNeeds returned error: %v", err)
	}

	if val := needs.ProjectedCoverage["patternA"]; val != 30 {
		t.Fatalf("expected projected coverage 30 for patternA, got %d", val)
	}
	if !contains(needs.PriorityAreas, "patternA") {
		t.Fatalf("expected patternA as priority, got %#v", needs.PriorityAreas)
	}
	if !contains(needs.Recommendations, "Collect more training data for pattern: patternA") {
		t.Fatalf("expected recommendation for patternA, got %#v", needs.Recommendations)
	}
}

func newLocalAITestServer(t *testing.T, response map[string]any) *httptest.Server {
	t.Helper()

	payload, err := json.Marshal(response)
	if err != nil {
		t.Fatalf("failed to marshal response: %v", err)
	}

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			jsonResponse := map[string]any{
				"choices": []map[string]any{
					{
						"message": map[string]any{
							"content": string(payload),
						},
					},
				},
			}
			if err := json.NewEncoder(w).Encode(jsonResponse); err != nil {
				t.Fatalf("failed to encode response: %v", err)
			}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/domains":
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(`{"data":[]}`)); err != nil {
				t.Fatalf("failed to write domains response: %v", err)
			}
		default:
			http.NotFound(w, r)
		}
	})

	return httptest.NewServer(handler)
}

func newLocalAIInvalidJSONServer(t *testing.T) *httptest.Server {
	t.Helper()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/chat/completions":
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(`{"choices":[{"message":{"content":"not-json"}}]}`)); err != nil {
				t.Fatalf("failed to write invalid JSON response: %v", err)
			}
		case r.Method == http.MethodGet && r.URL.Path == "/v1/domains":
			w.Header().Set("Content-Type", "application/json")
			if _, err := w.Write([]byte(`{"data":[]}`)); err != nil {
				t.Fatalf("failed to write domains response: %v", err)
			}
		default:
			http.NotFound(w, r)
		}
	})

	return httptest.NewServer(handler)
}

func contains(slice []string, value string) bool {
	for _, item := range slice {
		if item == value {
			return true
		}
	}
	return false
}

func closeTo(actual, expected float64) bool {
	const eps = 1e-6
	return actual > expected-eps && actual < expected+eps
}
