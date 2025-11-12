package integrations

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
)

// OrchestrationChainMatcher uses semantic embeddings to match chains to tasks
type OrchestrationChainMatcher struct {
	logger           *log.Logger
	extractServiceURL string
	useSemanticMatch bool
}

// NewOrchestrationChainMatcher creates a new chain matcher
func NewOrchestrationChainMatcher(logger *log.Logger) *OrchestrationChainMatcher {
	return &OrchestrationChainMatcher{
		logger:           logger,
		useSemanticMatch: os.Getenv("USE_SAP_RPT_EMBEDDINGS") == "true",
	}
}

// SetExtractServiceURL sets the Extract service URL for semantic search
func (ocm *OrchestrationChainMatcher) SetExtractServiceURL(url string) {
	ocm.extractServiceURL = url
}

// MatchChainToTask matches an orchestration chain to a task using semantic search
func (ocm *OrchestrationChainMatcher) MatchChainToTask(
	taskDescription string,
	tableName string,
	classification string,
) (string, float64, error) {
	// Use classification to route to appropriate chain
	if classification != "" {
		chainName := ocm.routeByClassification(classification)
		if chainName != "" {
			return chainName, 0.9, nil // High confidence for classification-based routing
		}
	}

	// Use semantic search if enabled
	if ocm.useSemanticMatch && ocm.extractServiceURL != "" {
		chainName, score, err := ocm.matchViaSemantic(taskDescription, tableName)
		if err == nil && chainName != "" {
			return chainName, score, nil
		}
	}

	// Default fallback
	return "default_chain", 0.5, nil
}

// routeByClassification routes to chain based on table classification
func (ocm *OrchestrationChainMatcher) routeByClassification(classification string) string {
	switch classification {
	case "transaction":
		return "transaction_processing_chain"
	case "reference":
		return "reference_lookup_chain"
	case "staging":
		return "staging_etl_chain"
	case "test":
		return "test_processing_chain"
	default:
		return ""
	}
}

// matchViaSemantic uses semantic search to find matching chain
func (ocm *OrchestrationChainMatcher) matchViaSemantic(
	taskDescription string,
	tableName string,
) (string, float64, error) {
	// Search for relevant tables/workflows using semantic search
	query := fmt.Sprintf("%s %s", taskDescription, tableName)

	searchPayload := map[string]any{
		"query":           query,
		"artifact_type":    "table",
		"limit":           5,
		"use_semantic":     true,
		"use_hybrid_search": true,
	}

	payloadJSON, err := json.Marshal(searchPayload)
	if err != nil {
		return "", 0.0, fmt.Errorf("marshal search payload: %w", err)
	}

	searchURL := fmt.Sprintf("%s/knowledge-graph/search", ocm.extractServiceURL)
	resp, err := http.Post(searchURL, "application/json", bytes.NewReader(payloadJSON))
	if err != nil {
		return "", 0.0, fmt.Errorf("semantic search failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", 0.0, fmt.Errorf("search returned status %d", resp.StatusCode)
	}

	var searchResult struct {
		Results []struct {
			Metadata map[string]any `json:"metadata"`
			Score    float64        `json:"score"`
		} `json:"results"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&searchResult); err != nil {
		return "", 0.0, fmt.Errorf("decode search result: %w", err)
	}

	// Analyze results to determine chain
	if len(searchResult.Results) > 0 {
		bestResult := searchResult.Results[0]
		if metadata := bestResult.Metadata; metadata != nil {
			if classification, ok := metadata["table_classification"].(string); ok {
				chainName := ocm.routeByClassification(classification)
				if chainName != "" {
					return chainName, bestResult.Score, nil
				}
			}
		}
	}

	return "", 0.0, nil
}

// SelectChainForTable selects an orchestration chain for a table based on classification
func (ocm *OrchestrationChainMatcher) SelectChainForTable(
	tableName string,
	projectID string,
	systemID string,
) (string, error) {
	// Query classification for this table
	queryPayload := map[string]any{
		"query": `
			MATCH (n)
			WHERE n.type = 'table' AND n.label = $table_name
			RETURN n.props.table_classification AS classification
		`,
		"params": map[string]any{
			"table_name": tableName,
		},
	}

	payloadJSON, err := json.Marshal(queryPayload)
	if err != nil {
		return "default_chain", err
	}

	queryURL := fmt.Sprintf("%s/knowledge-graph/query", ocm.extractServiceURL)
	resp, err := http.Post(queryURL, "application/json", bytes.NewReader(payloadJSON))
	if err != nil {
		return "default_chain", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "default_chain", fmt.Errorf("query returned status %d", resp.StatusCode)
	}

	var queryResult struct {
		Data []map[string]any `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
		return "default_chain", err
	}

	if len(queryResult.Data) > 0 {
		if classification, ok := queryResult.Data[0]["classification"].(string); ok {
			chainName := ocm.routeByClassification(classification)
			if chainName != "" {
				return chainName, nil
			}
		}
	}

	return "default_chain", nil
}

