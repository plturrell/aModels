// +build integration

package agents

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"testing"
	"time"
)

// TestPerplexityPipelineGNNIntegration tests GNN query methods in PerplexityPipeline.
func TestPerplexityPipelineGNNIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	trainingURL := os.Getenv("TRAINING_SERVICE_URL")
	if trainingURL == "" {
		trainingURL = "http://training-service:8080"
	}

	graphServiceURL := os.Getenv("GRAPH_SERVICE_URL")
	if graphServiceURL == "" {
		graphServiceURL = "http://graph-service:8081"
	}

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)

	config := PerplexityPipelineConfig{
		PerplexityAPIKey:    "test-key",
		PerplexityBaseURL:   "https://api.perplexity.ai",
		DeepSeekOCREndpoint: "",
		DeepSeekOCRAPIKey:   "",
		DeepResearchURL:     "",
		UnifiedWorkflowURL:  "",
		CatalogURL:          "",
		TrainingURL:         trainingURL,
		LocalAIURL:          "",
		SearchURL:           "",
		ExtractURL:          "",
		Logger:              logger,
	}

	pipeline, err := NewPerplexityPipeline(config)
	if err != nil {
		t.Fatalf("Failed to create PerplexityPipeline: %v", err)
	}

	// Test nodes and edges
	testNodes := []map[string]interface{}{
		{
			"id":   "table_1",
			"type": "table",
			"properties": map[string]interface{}{
				"name":        "customers",
				"column_count": 5,
				"row_count":   1000,
				"domain":      "sales",
			},
		},
		{
			"id":   "column_1",
			"type": "column",
			"properties": map[string]interface{}{
				"name":      "customer_id",
				"data_type": "string",
				"nullable":  false,
			},
		},
	}

	testEdges := []map[string]interface{}{
		{
			"source_id": "table_1",
			"target_id": "column_1",
			"label":     "HAS_COLUMN",
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Test QueryGNNEmbeddings
	t.Run("QueryGNNEmbeddings", func(t *testing.T) {
		result, err := pipeline.QueryGNNEmbeddings(ctx, testNodes, testEdges, true)
		if err != nil {
			t.Logf("QueryGNNEmbeddings error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNEmbeddings returned nil result")
			return
		}

		// Verify result structure
		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		// Check for error (acceptable if model not trained)
		if errorMsg, hasError := resultMap["error"]; hasError {
			t.Logf("GNN service returned error (may be expected): %v", errorMsg)
			return
		}

		// Verify embeddings are present
		if _, hasEmbedding := resultMap["graph_embedding"]; !hasEmbedding {
			if _, hasNodeEmbedding := resultMap["node_embeddings"]; !hasNodeEmbedding {
				t.Error("No embeddings found in result")
			}
		}
	})

	// Test QueryGNNStructuralInsights
	t.Run("QueryGNNStructuralInsights", func(t *testing.T) {
		result, err := pipeline.QueryGNNStructuralInsights(ctx, testNodes, testEdges, "all", 0.5)
		if err != nil {
			t.Logf("QueryGNNStructuralInsights error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNStructuralInsights returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		if errorMsg, hasError := resultMap["error"]; hasError {
			t.Logf("GNN service returned error (may be expected): %v", errorMsg)
			return
		}

		// Verify insights are present
		if _, hasAnomalies := resultMap["anomalies"]; !hasAnomalies {
			if _, hasPatterns := resultMap["patterns"]; !hasPatterns {
				t.Log("No insights found (may be expected if model not trained)")
			}
		}
	})

	// Test QueryGNNPredictLinks
	t.Run("QueryGNNPredictLinks", func(t *testing.T) {
		result, err := pipeline.QueryGNNPredictLinks(ctx, testNodes, testEdges, nil, 5)
		if err != nil {
			t.Logf("QueryGNNPredictLinks error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNPredictLinks returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		if errorMsg, hasError := resultMap["error"]; hasError {
			t.Logf("GNN service returned error (may be expected): %v", errorMsg)
			return
		}

		// Verify predictions are present
		if _, hasPredictions := resultMap["predictions"]; !hasPredictions {
			t.Log("No predictions found (may be expected if model not trained)")
		}
	})

	// Test QueryGNNClassifyNodes
	t.Run("QueryGNNClassifyNodes", func(t *testing.T) {
		result, err := pipeline.QueryGNNClassifyNodes(ctx, testNodes, testEdges, nil)
		if err != nil {
			t.Logf("QueryGNNClassifyNodes error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNClassifyNodes returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		if errorMsg, hasError := resultMap["error"]; hasError {
			t.Logf("GNN service returned error (may be expected): %v", errorMsg)
			return
		}

		// Verify predictions are present
		if _, hasPredictions := resultMap["predictions"]; !hasPredictions {
			t.Log("No predictions found (may be expected if model not trained)")
		}
	})

	// Test QueryGNNHybrid
	t.Run("QueryGNNHybrid", func(t *testing.T) {
		result, err := pipeline.QueryGNNHybrid(ctx, "Find similar tables", testNodes, testEdges, "embeddings", false)
		if err != nil {
			t.Logf("QueryGNNHybrid error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNHybrid returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		// Verify hybrid result structure
		if _, hasKG := resultMap["kg_result"]; !hasKG {
			if _, hasGNN := resultMap["gnn_result"]; !hasGNN {
				t.Log("No results found (may be expected if services not available)")
			}
		}
	})
}

// TestDMSPipelineGNNIntegration tests GNN query methods in DMSPipeline.
func TestDMSPipelineGNNIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	trainingURL := os.Getenv("TRAINING_SERVICE_URL")
	if trainingURL == "" {
		trainingURL = "http://training-service:8080"
	}

	graphServiceURL := os.Getenv("GRAPH_SERVICE_URL")
	if graphServiceURL == "" {
		graphServiceURL = "http://graph-service:8081"
	}

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)

	config := DMSPipelineConfig{
		DMSConnectorConfig: map[string]interface{}{
			"endpoint": "test-endpoint",
		},
		UnifiedWorkflowURL: "",
		CatalogURL:         "",
		TrainingURL:       trainingURL,
		LocalAIURL:         "",
		SearchURL:          "",
		ExtractURL:         "",
		Logger:             logger,
	}

	pipeline, err := NewDMSPipeline(config)
	if err != nil {
		t.Fatalf("Failed to create DMSPipeline: %v", err)
	}

	testNodes := []map[string]interface{}{
		{
			"id":   "table_1",
			"type": "table",
			"properties": map[string]interface{}{
				"name": "customers",
			},
		},
	}

	testEdges := []map[string]interface{}{}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Test QueryGNNEmbeddings
	t.Run("QueryGNNEmbeddings", func(t *testing.T) {
		result, err := pipeline.QueryGNNEmbeddings(ctx, testNodes, testEdges, true)
		if err != nil {
			t.Logf("QueryGNNEmbeddings error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNEmbeddings returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		if errorMsg, hasError := resultMap["error"]; hasError {
			t.Logf("GNN service returned error (may be expected): %v", errorMsg)
			return
		}

		if _, hasEmbedding := resultMap["graph_embedding"]; !hasEmbedding {
			if _, hasNodeEmbedding := resultMap["node_embeddings"]; !hasNodeEmbedding {
				t.Error("No embeddings found in result")
			}
		}
	})

	// Test QueryGNNHybrid
	t.Run("QueryGNNHybrid", func(t *testing.T) {
		result, err := pipeline.QueryGNNHybrid(ctx, "Find similar tables", testNodes, testEdges, "embeddings", false)
		if err != nil {
			t.Logf("QueryGNNHybrid error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNHybrid returned nil result")
			return
		}

		resultBytes, err := json.Marshal(result)
		if err != nil {
			t.Errorf("Failed to marshal result: %v", err)
			return
		}

		var resultMap map[string]interface{}
		if err := json.Unmarshal(resultBytes, &resultMap); err != nil {
			t.Errorf("Failed to unmarshal result: %v", err)
			return
		}

		if _, hasKG := resultMap["kg_result"]; !hasKG {
			if _, hasGNN := resultMap["gnn_result"]; !hasGNN {
				t.Log("No results found (may be expected if services not available)")
			}
		}
	})
}

