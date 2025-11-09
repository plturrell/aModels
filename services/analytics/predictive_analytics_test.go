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

func contains(slice []string, value string) bool {
	for _, item := range slice {
		if item == value {
			return true
		}
	}
	return false
}
