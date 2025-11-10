package quality

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/httpclient"
)

// QualityMonitor monitors data quality by integrating with Extract service metrics.
type QualityMonitor struct {
	extractServiceURL string
	httpClient        *httpclient.Client
	logger            *log.Logger
}

// NewQualityMonitor creates a new quality monitor.
func NewQualityMonitor(extractServiceURL string, logger *log.Logger) *QualityMonitor {
	var client *httpclient.Client
	if extractServiceURL != "" {
		client = httpclient.NewClient(httpclient.ClientConfig{
			Timeout:         30 * time.Second,
			MaxRetries:      3,
			InitialBackoff:  1 * time.Second,
			MaxBackoff:      5 * time.Second,
			BaseURL:         extractServiceURL,
			HealthCheckPath: "/healthz",
			Logger:          logger,
		})
	}
	
	return &QualityMonitor{
		extractServiceURL: extractServiceURL,
		httpClient:        client,
		logger:            logger,
	}
}

// ExtractMetrics represents quality metrics from the Extract service.
type ExtractMetrics struct {
	MetadataEntropy   float64            `json:"metadata_entropy"`
	KLDivergence      float64            `json:"kl_divergence"`
	ColumnCount       int                `json:"column_count"`
	QualityScore      float64            `json:"quality_score"`
	QualityLevel      string             `json:"quality_level"`
	ActualDistribution map[string]float64 `json:"actual_distribution"`
	IdealDistribution  map[string]float64 `json:"ideal_distribution"`
	LastUpdated       time.Time          `json:"last_updated"`
}

// FetchQualityMetrics fetches quality metrics from the Extract service for a data element.
func (qm *QualityMonitor) FetchQualityMetrics(ctx context.Context, dataElementID string) (*ExtractMetrics, error) {
	if qm.extractServiceURL == "" {
		return nil, fmt.Errorf("extract service URL not configured")
	}
	
	query := map[string]any{
		"query": `
			MATCH (n:Node {id: $element_id})
			WHERE n.properties_json CONTAINS $element_id
			RETURN n.metadata_entropy AS metadata_entropy,
			       n.kl_divergence AS kl_divergence,
			       n.column_count AS column_count,
			       n.properties_json AS properties
		`,
		"params": map[string]any{
			"element_id": dataElementID,
		},
	}
	
	var result struct {
		Data []map[string]any `json:"data"`
	}
	
	if qm.httpClient != nil {
		// Use enhanced HTTP client
		validator := func(data map[string]interface{}) error {
			if resultData, ok := data["data"].([]interface{}); !ok || len(resultData) == 0 {
				return fmt.Errorf("response missing or empty 'data' field")
			}
			return nil
		}
		
		var responseData map[string]interface{}
		err := qm.httpClient.PostJSON(ctx, "/knowledge-graph/query", query, &responseData, validator)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch metrics: %w", err)
		}
		
		// Convert response to result struct
		if data, ok := responseData["data"].([]interface{}); ok {
			for _, item := range data {
				if itemMap, ok := item.(map[string]interface{}); ok {
					result.Data = append(result.Data, itemMap)
				}
			}
		}
	} else {
		// Fallback to basic HTTP client
		jsonData, err := json.Marshal(query)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal query: %w", err)
		}
		
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		
		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch metrics: %w", err)
		}
		defer resp.Body.Close()
		
		if resp.StatusCode != http.StatusOK {
			// Read response body for better error messages
			body, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("extract service returned status %d: %s", resp.StatusCode, string(body))
		}
		
		if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}
	}
	
	if len(result.Data) == 0 {
		return nil, fmt.Errorf("no metrics found for data element %s", dataElementID)
	}
	
	// Parse metrics from result
	record := result.Data[0]
	metrics := &ExtractMetrics{
		LastUpdated: time.Now(),
	}
	
	if val, ok := record["metadata_entropy"].(float64); ok {
		metrics.MetadataEntropy = val
	}
	if val, ok := record["kl_divergence"].(float64); ok {
		metrics.KLDivergence = val
	}
	if val, ok := record["column_count"].(float64); ok {
		metrics.ColumnCount = int(val)
	}
	
	// Calculate quality score from metrics
	// Use similar logic to metrics_interpreter.go
	entropyScore := 0.5
	if metrics.MetadataEntropy >= 1.0 && metrics.MetadataEntropy <= 4.0 {
		entropyScore = 1.0
	} else if metrics.MetadataEntropy < 1.0 {
		entropyScore = metrics.MetadataEntropy / 1.0 * 0.5
	}
	
	klScore := 1.0
	if metrics.KLDivergence > 1.0 {
		klScore = 0.0
	} else if metrics.KLDivergence > 0.5 {
		klScore = 1.0 - (metrics.KLDivergence-0.5)/(1.0-0.5)*0.5
	}
	
	columnScore := 1.0
	if metrics.ColumnCount < 5 {
		columnScore = float64(metrics.ColumnCount) / 5.0
	}
	
	metrics.QualityScore = entropyScore*0.4 + klScore*0.4 + columnScore*0.2
	
	// Determine quality level
	if metrics.QualityScore >= 0.9 {
		metrics.QualityLevel = "excellent"
	} else if metrics.QualityScore >= 0.7 {
		metrics.QualityLevel = "good"
	} else if metrics.QualityScore >= 0.5 {
		metrics.QualityLevel = "fair"
	} else if metrics.QualityScore >= 0.3 {
		metrics.QualityLevel = "poor"
	} else {
		metrics.QualityLevel = "critical"
	}
	
	return metrics, nil
}

// UpdateQualityMetrics updates quality metrics for a data element.
func (qm *QualityMonitor) UpdateQualityMetrics(ctx context.Context, elementID string, metrics *QualityMetrics) error {
	// Fetch real metrics from Extract service
	extractMetrics, err := qm.FetchQualityMetrics(ctx, elementID)
	if err != nil {
		// Log but don't fail - metrics might not be available yet
		if qm.logger != nil {
			qm.logger.Printf("Warning: Failed to fetch metrics for %s: %v", elementID, err)
		}
		return nil // Non-fatal
	}
	
	// Update quality metrics
	metrics.UpdateMetric("freshness", extractMetrics.QualityScore)
	metrics.UpdateMetric("completeness", extractMetrics.QualityScore)
	metrics.UpdateMetric("accuracy", extractMetrics.QualityScore)
	metrics.UpdateMetric("consistency", extractMetrics.QualityScore)
	metrics.UpdateMetric("validity", extractMetrics.QualityScore)
	
	// Update SLOs
	for i := range metrics.SLOs {
		switch metrics.SLOs[i].Name {
		case "freshness":
			metrics.SLOs[i].Current = extractMetrics.QualityScore
		case "completeness":
			metrics.SLOs[i].Current = extractMetrics.QualityScore
		case "quality":
			metrics.SLOs[i].Current = extractMetrics.QualityScore
		}
	}
	
	metrics.LastValidated = extractMetrics.LastUpdated
	metrics.CheckSLOs()
	
	if qm.logger != nil {
		qm.logger.Printf("Updated quality metrics for %s: score=%.2f, level=%s", 
			elementID, extractMetrics.QualityScore, extractMetrics.QualityLevel)
	}
	
	return nil
}

// MonitorQuality continuously monitors quality for all data elements.
func (qm *QualityMonitor) MonitorQuality(ctx context.Context, interval time.Duration, callback func(elementID string, metrics *ExtractMetrics)) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			// In production, would query catalog for all data elements
			// For now, this is a placeholder for the monitoring loop
			if qm.logger != nil {
				qm.logger.Println("Quality monitoring cycle started")
			}
		}
	}
}

