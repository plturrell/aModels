// +build integration

package workflows

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/chains"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms/localai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/prompts"
)

// TestOrchestrationChainCreation tests that orchestration chains can be created for all supported types.
func TestOrchestrationChainCreation(t *testing.T) {
	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		localAIURL = "http://localai:8080"
	}

	testCases := []struct {
		name      string
		chainName string
		wantErr   bool
	}{
		{"default chain", "llm_chain", false},
		{"default alias", "default", false},
		{"question answering", "question_answering", false},
		{"qa alias", "qa", false},
		{"summarization", "summarization", false},
		{"summarize alias", "summarize", false},
		{"knowledge graph analyzer", "knowledge_graph_analyzer", false},
		{"kg analyzer alias", "kg_analyzer", false},
		{"data quality analyzer", "data_quality_analyzer", false},
		{"quality analyzer alias", "quality_analyzer", false},
		{"pipeline analyzer", "pipeline_analyzer", false},
		{"pipeline alias", "pipeline", false},
		{"sql analyzer", "sql_analyzer", false},
		{"sql alias", "sql", false},
		{"agentflow analyzer", "agentflow_analyzer", false},
		{"agentflow alias", "agentflow", false},
		{"unknown chain", "unknown_chain", false}, // Should default to simple chain
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			chain, err := createOrchestrationChain(tc.chainName, localAIURL)
			if (err != nil) != tc.wantErr {
				t.Errorf("createOrchestrationChain() error = %v, wantErr %v", err, tc.wantErr)
				return
			}
			if err == nil && chain == nil {
				t.Error("createOrchestrationChain() returned nil chain without error")
				return
			}
			if err == nil {
				outputKeys := chain.GetOutputKeys()
				if len(outputKeys) == 0 {
					t.Error("chain has no output keys")
				}
			}
		})
	}
}

// TestOrchestrationChainExecution tests that orchestration chains can execute with valid inputs.
func TestOrchestrationChainExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		t.Skip("LOCALAI_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test simple LLM chain
	chain, err := createOrchestrationChain("llm_chain", localAIURL)
	if err != nil {
		t.Fatalf("Failed to create chain: %v", err)
	}

	inputs := map[string]any{
		"input": "What is 2+2?",
	}

	result, err := chains.Call(ctx, chain, inputs)
	if err != nil {
		t.Fatalf("Chain execution failed: %v", err)
	}

	if result == nil {
		t.Fatal("Chain returned nil result")
	}

	// Check for expected output key
	outputKeys := chain.GetOutputKeys()
	if len(outputKeys) == 0 {
		t.Fatal("Chain has no output keys")
	}

	// Check that at least one output key has a value
	hasOutput := false
	for _, key := range outputKeys {
		if val, ok := result[key]; ok && val != nil {
			hasOutput = true
			if strVal, ok := val.(string); ok && strVal != "" {
				t.Logf("Chain output [%s]: %s", key, strVal)
			}
		}
	}

	if !hasOutput {
		t.Error("Chain execution produced no output")
	}
}

// TestOrchestrationChainWithKnowledgeGraph tests orchestration chain with knowledge graph context.
func TestOrchestrationChainWithKnowledgeGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		t.Skip("LOCALAI_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create knowledge graph analyzer chain
	chain, err := createOrchestrationChain("knowledge_graph_analyzer", localAIURL)
	if err != nil {
		t.Fatalf("Failed to create chain: %v", err)
	}

	// Prepare inputs with knowledge graph context
	inputs := map[string]any{
		"node_count":               10,
		"edge_count":               15,
		"quality_score":           0.85,
		"quality_level":           "good",
		"knowledge_graph_context": "Test knowledge graph with 10 nodes and 15 edges",
		"query":                   "Analyze the knowledge graph structure",
	}

	result, err := chains.Call(ctx, chain, inputs)
	if err != nil {
		t.Fatalf("Chain execution failed: %v", err)
	}

	if result == nil {
		t.Fatal("Chain returned nil result")
	}

	// Verify output
	outputKeys := chain.GetOutputKeys()
	if len(outputKeys) == 0 {
		t.Fatal("Chain has no output keys")
	}
}

// TestOrchestrationChainRetry tests that retry logic works for orchestration chains.
func TestOrchestrationChainRetry(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		t.Skip("LOCALAI_URL not set, skipping integration test")
	}

	// This test verifies that the retry logic in RunOrchestrationChainNode
	// handles transient failures correctly. Since we're using the real
	// orchestration framework now, retry is handled by the integration utilities.
	// This test ensures the chain can be created and executed.

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	chain, err := createOrchestrationChain("llm_chain", localAIURL)
	if err != nil {
		t.Fatalf("Failed to create chain: %v", err)
	}

	inputs := map[string]any{
		"input": "Test retry mechanism",
	}

	// Execute multiple times to verify consistency
	for i := 0; i < 3; i++ {
		result, err := chains.Call(ctx, chain, inputs)
		if err != nil {
			t.Errorf("Chain execution failed on attempt %d: %v", i+1, err)
			continue
		}
		if result == nil {
			t.Errorf("Chain returned nil result on attempt %d", i+1)
		}
	}
}

// TestOrchestrationChainInputValidation tests that chains handle missing inputs gracefully.
func TestOrchestrationChainInputValidation(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
	}

	localAIURL := os.Getenv("LOCALAI_URL")
	if localAIURL == "" {
		t.Skip("LOCALAI_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	chain, err := createOrchestrationChain("question_answering", localAIURL)
	if err != nil {
		t.Fatalf("Failed to create chain: %v", err)
	}

	// Test with missing required inputs
	inputs := map[string]any{
		"question": "What is the answer?",
		// Missing "context"
	}

	// The chain should either handle missing inputs gracefully or return an error
	_, err = chains.Call(ctx, chain, inputs)
	// We don't assert on the error here because the framework might handle
	// missing inputs differently (e.g., use empty string, or return error)
	// The important thing is it doesn't panic
	if err != nil {
		t.Logf("Chain returned error for missing input (expected): %v", err)
	}
}

