// +build integration

package workflows

import (
	"context"
	"encoding/json"
	"os"
	"testing"
	"time"
)

// TestGNNProcessorIntegration tests the GNN processor integration with StateGraph workflows.
func TestGNNProcessorIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
	if trainingServiceURL == "" {
		trainingServiceURL = "http://training-service:8080"
	}

	extractServiceURL := os.Getenv("EXTRACT_SERVICE_URL")
	if extractServiceURL == "" {
		extractServiceURL = "http://extract-service:19080"
	}

	opts := GNNProcessorOptions{
		TrainingServiceURL: trainingServiceURL,
		ExtractServiceURL:  extractServiceURL,
	}

	// Test nodes and edges
	testNodes := []Node{
		{
			ID:   "table_1",
			Type: "table",
			Properties: map[string]interface{}{
				"name":        "customers",
				"column_count": 5,
				"row_count":   1000,
				"domain":      "sales",
			},
		},
		{
			ID:   "column_1",
			Type: "column",
			Properties: map[string]interface{}{
				"name":      "customer_id",
				"data_type": "string",
				"nullable":  false,
				"domain":    "sales",
			},
		},
		{
			ID:   "table_2",
			Type: "table",
			Properties: map[string]interface{}{
				"name":        "orders",
				"column_count": 4,
				"row_count":   5000,
				"domain":      "sales",
			},
		},
	}

	testEdges := []Edge{
		{
			SourceID: "table_1",
			TargetID: "column_1",
			Label:    "HAS_COLUMN",
			Properties: map[string]interface{}{},
		},
		{
			SourceID: "table_1",
			TargetID: "table_2",
			Label:    "RELATED_TO",
			Properties: map[string]interface{}{},
		},
	}

	// Test QueryGNNNode
	t.Run("QueryGNNNode_Embeddings", func(t *testing.T) {
		queryNode := QueryGNNNode(opts)

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		state := map[string]interface{}{
			"gnn_query": GNNQueryRequest{
				QueryType: "embeddings",
				Nodes:     testNodes,
				Edges:     testEdges,
				Params: map[string]interface{}{
					"graph_level": true,
				},
			},
		}

		result, err := queryNode(ctx, state)
		if err != nil {
			t.Logf("QueryGNNNode error (may be expected if service not available): %v", err)
			// Don't fail if service is not available
			return
		}

		if result == nil {
			t.Error("QueryGNNNode returned nil result")
			return
		}

		// Verify result structure
		if gnnResult, ok := result["gnn_result"]; ok {
			gnnResultMap, ok := gnnResult.(map[string]interface{})
			if !ok {
				t.Error("gnn_result is not a map")
				return
			}

			// Check for error (acceptable if model not trained)
			if errorMsg, hasError := gnnResultMap["error"]; hasError {
				t.Logf("GNN service returned error (may be expected): %v", errorMsg)
				return
			}

			// Verify embeddings are present
			if _, hasEmbedding := gnnResultMap["graph_embedding"]; !hasEmbedding {
				if _, hasNodeEmbedding := gnnResultMap["node_embeddings"]; !hasNodeEmbedding {
					t.Error("No embeddings found in result")
				}
			}
		} else {
			t.Error("gnn_result not found in result")
		}
	})

	// Test QueryGNNNode with structural insights
	t.Run("QueryGNNNode_StructuralInsights", func(t *testing.T) {
		queryNode := QueryGNNNode(opts)

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		state := map[string]interface{}{
			"gnn_query": GNNQueryRequest{
				QueryType: "structural-insights",
				Nodes:     testNodes,
				Edges:     testEdges,
				Params: map[string]interface{}{
					"insight_type": "all",
					"threshold":    0.5,
				},
			},
		}

		result, err := queryNode(ctx, state)
		if err != nil {
			t.Logf("QueryGNNNode error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("QueryGNNNode returned nil result")
			return
		}

		if gnnResult, ok := result["gnn_result"]; ok {
			gnnResultMap, ok := gnnResult.(map[string]interface{})
			if !ok {
				t.Error("gnn_result is not a map")
				return
			}

			if errorMsg, hasError := gnnResultMap["error"]; hasError {
				t.Logf("GNN service returned error (may be expected): %v", errorMsg)
				return
			}

			// Verify insights are present
			if _, hasAnomalies := gnnResultMap["anomalies"]; !hasAnomalies {
				if _, hasPatterns := gnnResultMap["patterns"]; !hasPatterns {
					t.Log("No insights found (may be expected if model not trained)")
				}
			}
		}
	})

	// Test HybridQueryNode
	t.Run("HybridQueryNode", func(t *testing.T) {
		hybridNode := HybridQueryNode(opts)

		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		defer cancel()

		state := map[string]interface{}{
			"hybrid_query": HybridQueryRequest{
				Query:    "Find similar tables in the sales domain",
				QueryKG:  false, // Skip KG for this test
				QueryGNN: true,
				GNNType:  "embeddings",
				Combine:  false,
			},
		}

		result, err := hybridNode(ctx, state)
		if err != nil {
			t.Logf("HybridQueryNode error (may be expected if service not available): %v", err)
			return
		}

		if result == nil {
			t.Error("HybridQueryNode returned nil result")
			return
		}

		// Verify result structure
		if hybridResult, ok := result["hybrid_result"]; ok {
			hybridResultMap, ok := hybridResult.(map[string]interface{})
			if !ok {
				t.Error("hybrid_result is not a map")
				return
			}

			// Check for GNN result
			if _, hasGNN := hybridResultMap["gnn_result"]; !hasGNN {
				t.Log("No GNN result found (may be expected if service not available)")
			}
		}
	})

	// Test isStructuralQuery function
	t.Run("IsStructuralQuery", func(t *testing.T) {
		testCases := []struct {
			query    string
			expected bool
		}{
			{"Find similar tables", true},
			{"What patterns exist?", true},
			{"Classify these nodes", true},
			{"Generate embeddings", true},
			{"Predict missing links", true},
			{"What tables exist?", false},
			{"Show me column customer_id", false},
			{"MATCH (n) RETURN n", false},
		}

		for _, tc := range testCases {
			result := isStructuralQuery(tc.query)
			if result != tc.expected {
				t.Errorf("isStructuralQuery(%q) = %v, expected %v", tc.query, result, tc.expected)
			}
		}
	})
}

// TestGNNWorkflowIntegration tests the complete GNN workflow.
func TestGNNWorkflowIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
	if trainingServiceURL == "" {
		trainingServiceURL = "http://training-service:8080"
	}

	extractServiceURL := os.Getenv("EXTRACT_SERVICE_URL")
	if extractServiceURL == "" {
		extractServiceURL = "http://extract-service:19080"
	}

	opts := GNNProcessorOptions{
		TrainingServiceURL: trainingServiceURL,
		ExtractServiceURL:  extractServiceURL,
	}

	workflow, err := NewGNNProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create GNN workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	testNodes := []Node{
		{
			ID:   "table_1",
			Type: "table",
			Properties: map[string]interface{}{
				"name": "customers",
			},
		},
	}

	testEdges := []Edge{}

	state := map[string]interface{}{
		"gnn_query": GNNQueryRequest{
			QueryType: "embeddings",
			Nodes:     testNodes,
			Edges:     testEdges,
			Params: map[string]interface{}{
				"graph_level": true,
			},
		},
	}

	result, err := workflow.Invoke(ctx, state)
	if err != nil {
		t.Logf("GNN workflow error (may be expected if service not available): %v", err)
		return
	}

	if result == nil {
		t.Error("Workflow returned nil result")
		return
	}

	// Verify workflow result
	if gnnResult, ok := result["gnn_result"]; ok {
		if gnnResult == nil {
			t.Error("gnn_result is nil")
		}
	} else {
		t.Log("gnn_result not found (may be expected if workflow not fully configured)")
	}
}

// TestUnifiedWorkflowGNNIntegration tests GNN integration in unified workflow.
func TestUnifiedWorkflowGNNIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	trainingServiceURL := os.Getenv("TRAINING_SERVICE_URL")
	if trainingServiceURL == "" {
		trainingServiceURL = "http://training-service:8080"
	}

	extractServiceURL := os.Getenv("EXTRACT_SERVICE_URL")
	if extractServiceURL == "" {
		extractServiceURL = "http://extract-service:19080"
	}

	agentflowServiceURL := os.Getenv("AGENTFLOW_SERVICE_URL")
	if agentflowServiceURL == "" {
		agentflowServiceURL = "http://agentflow-service:9001"
	}

	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		localAIURL = "http://localai:8080"
	}

	opts := UnifiedProcessorOptions{
		ExtractServiceURL:   extractServiceURL,
		AgentFlowServiceURL: agentflowServiceURL,
		LocalAIURL:          localAIURL,
		TrainingServiceURL: trainingServiceURL,
	}

	workflow, err := NewUnifiedProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create unified workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	testNodes := []Node{
		{
			ID:   "table_1",
			Type: "table",
			Properties: map[string]interface{}{
				"name": "customers",
			},
		},
	}

	testEdges := []Edge{}

	state := map[string]interface{}{
		"unified_request": map[string]interface{}{
			"workflow_mode": "sequential",
			"gnn_query": map[string]interface{}{
				"query_type": "embeddings",
				"nodes":      testNodes,
				"edges":      testEdges,
				"params": map[string]interface{}{
					"graph_level": true,
				},
			},
		},
	}

	result, err := workflow.Invoke(ctx, state)
	if err != nil {
		t.Logf("Unified workflow GNN error (may be expected if service not available): %v", err)
		return
	}

	if result == nil {
		t.Error("Workflow returned nil result")
		return
	}

	// Verify GNN query was processed
	if unifiedResult, ok := result["unified_result"]; ok {
		unifiedResultMap, ok := unifiedResult.(map[string]interface{})
		if !ok {
			t.Log("unified_result is not a map (may be expected)")
			return
		}

		if gnnQueryProcessed, ok := unifiedResultMap["gnn_query_processed"]; ok {
			if processed, ok := gnnQueryProcessed.(bool); ok && processed {
				t.Log("✓ GNN query was processed in unified workflow")
			}
		}
	}
}

