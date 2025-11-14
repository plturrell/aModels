package breakdetection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/httpclient"
)

// BreakSearchService provides semantic search for similar breaks
type BreakSearchService struct {
	searchServiceURL string
	httpClient       *httpclient.Client
	logger           *log.Logger
}

// NewBreakSearchService creates a new break search service
func NewBreakSearchService(searchServiceURL string, logger *log.Logger) *BreakSearchService {
	if searchServiceURL == "" {
		// Should fail explicitly instead of defaulting
		return &BreakSearchService{
			searchServiceURL: "",
			httpClient:       nil,
			logger:           logger,
		}
	}
	
	client := httpclient.NewClient(httpclient.ClientConfig{
		Timeout:         30 * time.Second,
		MaxRetries:      3,
		InitialBackoff:  1 * time.Second,
		MaxBackoff:      5 * time.Second,
		BaseURL:         searchServiceURL,
		HealthCheckPath: "/healthz",
		Logger:          logger,
	})
	
	return &BreakSearchService{
		searchServiceURL: searchServiceURL,
		httpClient:       client,
		logger:           logger,
	}
}

// SearchSimilarBreaks searches for similar historical breaks
func (bss *BreakSearchService) SearchSimilarBreaks(ctx context.Context, breakRecord *Break, limit int, threshold float64) ([]SimilarBreak, error) {
	if bss.searchServiceURL == "" {
		return nil, fmt.Errorf("search service URL not configured")
	}

	// Build search query from break details
	query := bss.buildBreakSearchQuery(breakRecord)

	// Prepare search request
	searchRequest := map[string]interface{}{
		"query":           query,
		"artifact_type":   "break",
		"limit":           limit,
		"threshold":       threshold,
		"use_semantic":    true,
		"use_hybrid_search": true,
		"filters": map[string]interface{}{
			"system_name":    string(breakRecord.SystemName),
			"detection_type": string(breakRecord.DetectionType),
		},
	}

	if bss.logger != nil {
		bss.logger.Printf("Searching for similar breaks: %s", breakRecord.BreakID)
	}

	// Call search service
	var searchResponse struct {
		Results []struct {
			ID       string                 `json:"id"`
			Content  string                 `json:"content"`
			Score    float64                `json:"score"`
			Metadata map[string]interface{} `json:"metadata"`
		} `json:"results"`
	}
	
	if bss.httpClient != nil {
		// Use enhanced HTTP client
		validator := func(data map[string]interface{}) error {
			if _, ok := data["results"]; !ok {
				return fmt.Errorf("response missing 'results' field")
			}
			return nil
		}
		
		var responseData map[string]interface{}
		err := bss.httpClient.PostJSON(ctx, "/knowledge-graph/search", searchRequest, &responseData, validator)
		if err != nil {
			return nil, fmt.Errorf("failed to execute search request: %w", err)
		}
		
		// Convert response to searchResponse struct
		if results, ok := responseData["results"].([]interface{}); ok {
			for _, result := range results {
				if resultMap, ok := result.(map[string]interface{}); ok {
					var id, content string
					var score float64
					var metadata map[string]interface{}
					
					if val, ok := resultMap["id"].(string); ok {
						id = val
					}
					if val, ok := resultMap["content"].(string); ok {
						content = val
					}
					if val, ok := resultMap["score"].(float64); ok {
						score = val
					}
					if val, ok := resultMap["metadata"].(map[string]interface{}); ok {
						metadata = val
					} else {
						metadata = make(map[string]interface{})
					}
					
					searchResponse.Results = append(searchResponse.Results, struct {
						ID       string                 `json:"id"`
						Content  string                 `json:"content"`
						Score    float64                `json:"score"`
						Metadata map[string]interface{} `json:"metadata"`
					}{
						ID:       id,
						Content:  content,
						Score:    score,
						Metadata: metadata,
					})
				}
			}
		}
	} else {
		// Fallback to basic HTTP client
		searchURL := fmt.Sprintf("%s/knowledge-graph/search", bss.searchServiceURL)
		jsonData, err := json.Marshal(searchRequest)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal search request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(jsonData))
		if err != nil {
			return nil, fmt.Errorf("failed to create search request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return nil, fmt.Errorf("failed to execute search request: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("search service returned status %d", resp.StatusCode)
		}

		if err := json.NewDecoder(resp.Body).Decode(&searchResponse); err != nil {
			return nil, fmt.Errorf("failed to decode search response: %w", err)
		}
	}

	// Convert search results to SimilarBreak
	similarBreaks := make([]SimilarBreak, 0, len(searchResponse.Results))
	for _, result := range searchResponse.Results {
		similarBreak := SimilarBreak{
			BreakID:     result.ID,
			Similarity:  result.Score,
			Description: result.Content,
		}

		// Extract metadata
		if metadata := result.Metadata; metadata != nil {
			if breakID, ok := metadata["break_id"].(string); ok {
				similarBreak.BreakID = breakID
			}
			if resolution, ok := metadata["resolution"].(string); ok {
				similarBreak.Resolution = resolution
			}
			if detectedAtStr, ok := metadata["detected_at"].(string); ok {
				if detectedAt, err := time.Parse(time.RFC3339, detectedAtStr); err == nil {
					similarBreak.DetectedAt = detectedAt
				}
			}
		}

		similarBreaks = append(similarBreaks, similarBreak)
	}

	if bss.logger != nil {
		bss.logger.Printf("Found %d similar breaks for: %s", len(similarBreaks), breakRecord.BreakID)
	}

	return similarBreaks, nil
}

// buildBreakSearchQuery builds a semantic search query from break details
func (bss *BreakSearchService) buildBreakSearchQuery(breakRecord *Break) string {
	var queryBuilder bytes.Buffer

	queryBuilder.WriteString(fmt.Sprintf("%s break in %s system", breakRecord.BreakType, breakRecord.SystemName))
	
	if breakRecord.BreakType != "" {
		queryBuilder.WriteString(fmt.Sprintf(" type: %s", breakRecord.BreakType))
	}
	
	if len(breakRecord.AffectedEntities) > 0 {
		queryBuilder.WriteString(fmt.Sprintf(" affecting: %s", breakRecord.AffectedEntities[0]))
	}
	
	if breakRecord.Difference != nil {
		if diffStr, ok := breakRecord.Difference["field"].(string); ok {
			queryBuilder.WriteString(fmt.Sprintf(" field: %s", diffStr))
		}
	}

	return queryBuilder.String()
}

// SearchBreakPatterns searches for break patterns in historical data
func (bss *BreakSearchService) SearchBreakPatterns(ctx context.Context, systemName SystemName, detectionType DetectionType, limit int) ([]map[string]interface{}, error) {
	if bss.searchServiceURL == "" {
		return nil, fmt.Errorf("search service URL not configured")
	}

	query := fmt.Sprintf("historical break patterns %s %s", systemName, detectionType)

	searchRequest := map[string]interface{}{
		"query":           query,
		"artifact_type":   "break_pattern",
		"limit":           limit,
		"use_semantic":    true,
		"use_hybrid_search": false,
		"filters": map[string]interface{}{
			"system_name":    string(systemName),
			"detection_type": string(detectionType),
		},
	}

	searchURL := fmt.Sprintf("%s/knowledge-graph/search", bss.searchServiceURL)
	jsonData, err := json.Marshal(searchRequest)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, searchURL, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create search request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := bss.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute search request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search service returned status %d", resp.StatusCode)
	}

	var searchResponse struct {
		Results []map[string]interface{} `json:"results"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResponse); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}

	return searchResponse.Results, nil
}

// IndexBreak indexes a break for future search
func (bss *BreakSearchService) IndexBreak(ctx context.Context, breakRecord *Break) error {
	if bss.searchServiceURL == "" {
		return fmt.Errorf("search service URL not configured")
	}

	// Build indexable content
	content := bss.buildBreakIndexContent(breakRecord)

	// Prepare index request
	indexRequest := map[string]interface{}{
		"id":       breakRecord.BreakID,
		"content":  content,
		"metadata": map[string]interface{}{
			"break_id":        breakRecord.BreakID,
			"system_name":     string(breakRecord.SystemName),
			"detection_type":  string(breakRecord.DetectionType),
			"break_type":      string(breakRecord.BreakType),
			"severity":        string(breakRecord.Severity),
			"status":          string(breakRecord.Status),
			"detected_at":     breakRecord.DetectedAt.Format(time.RFC3339),
			"affected_entities": breakRecord.AffectedEntities,
		},
		"artifact_type": "break",
	}

	if bss.httpClient != nil {
		// Use enhanced HTTP client
		var responseData map[string]interface{}
		err := bss.httpClient.PostJSON(ctx, "/knowledge-graph/index", indexRequest, &responseData)
		if err != nil {
			return fmt.Errorf("failed to execute index request: %w", err)
		}
	} else {
		// Fallback to basic HTTP client
		indexURL := fmt.Sprintf("%s/knowledge-graph/index", bss.searchServiceURL)
		jsonData, err := json.Marshal(indexRequest)
		if err != nil {
			return fmt.Errorf("failed to marshal index request: %w", err)
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, indexURL, bytes.NewReader(jsonData))
		if err != nil {
			return fmt.Errorf("failed to create index request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: 30 * time.Second}
		resp, err := client.Do(req)
		if err != nil {
			return fmt.Errorf("failed to execute index request: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
			return fmt.Errorf("index service returned status %d", resp.StatusCode)
		}
	}

	if bss.logger != nil {
		bss.logger.Printf("Indexed break for search: %s", breakRecord.BreakID)
	}

	return nil
}

// buildBreakIndexContent builds indexable content from break
func (bss *BreakSearchService) buildBreakIndexContent(breakRecord *Break) string {
	var contentBuilder bytes.Buffer

	contentBuilder.WriteString(fmt.Sprintf("Break: %s\n", breakRecord.BreakType))
	contentBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	contentBuilder.WriteString(fmt.Sprintf("Detection Type: %s\n", breakRecord.DetectionType))
	contentBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))

	if breakRecord.RootCauseAnalysis != "" {
		contentBuilder.WriteString(fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis))
	}

	if breakRecord.AIDescription != "" {
		contentBuilder.WriteString(fmt.Sprintf("Description: %s\n", breakRecord.AIDescription))
	}

	if len(breakRecord.AffectedEntities) > 0 {
		contentBuilder.WriteString(fmt.Sprintf("Affected: %v\n", breakRecord.AffectedEntities))
	}

	if breakRecord.Difference != nil {
		contentBuilder.WriteString(fmt.Sprintf("Difference: %v\n", breakRecord.Difference))
	}

	return contentBuilder.String()
}

