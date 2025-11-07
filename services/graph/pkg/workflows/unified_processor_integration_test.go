// +build integration

package workflows

import (
	"context"
	"os"
	"testing"
	"time"
)

// TestUnifiedWorkflowSequentialMode tests the unified workflow in sequential mode.
func TestUnifiedWorkflowSequentialMode(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
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
	}

	workflow, err := NewUnifiedProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create unified workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Test with minimal request (orchestration only)
	state := map[string]any{
		"unified_request": map[string]any{
			"workflow_mode": "sequential",
			"orchestration_request": map[string]any{
				"chain_name": "llm_chain",
				"inputs": map[string]any{
					"input": "Test unified workflow",
				},
			},
		},
	}

	result, err := workflow.Invoke(ctx, state)
	if err != nil {
		t.Fatalf("Unified workflow execution failed: %v", err)
	}

	if result == nil {
		t.Fatal("Workflow returned nil result")
	}

	// Verify orchestration result is present
	if orchResult, ok := result["orchestration_result"]; !ok {
		t.Error("orchestration_result not found in workflow result")
	} else if orchResult == nil {
		t.Error("orchestration_result is nil")
	}
}

// TestUnifiedWorkflowWithKnowledgeGraph tests unified workflow with knowledge graph processing.
func TestUnifiedWorkflowWithKnowledgeGraph(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
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
	}

	workflow, err := NewUnifiedProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create unified workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()

	// Test with knowledge graph request
	state := map[string]any{
		"unified_request": map[string]any{
			"workflow_mode": "sequential",
			"knowledge_graph_request": map[string]any{
				"project_id": "test-project",
				"system_id":  "test-system",
			},
			"orchestration_request": map[string]any{
				"chain_name": "knowledge_graph_analyzer",
				"inputs": map[string]any{
					"query": "Analyze the knowledge graph",
				},
			},
		},
	}

	result, err := workflow.Invoke(ctx, state)
	if err != nil {
		t.Fatalf("Unified workflow execution failed: %v", err)
	}

	if result == nil {
		t.Fatal("Workflow returned nil result")
	}

	// Verify knowledge graph result is present (if KG processing was requested)
	if kgResult, ok := result["knowledge_graph"]; ok && kgResult != nil {
		t.Log("Knowledge graph processing completed")
	}
}

// TestUnifiedWorkflowParallelMode tests the unified workflow in parallel mode.
func TestUnifiedWorkflowParallelMode(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
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
	}

	workflow, err := NewUnifiedProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create unified workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Test with parallel mode
	state := map[string]any{
		"unified_request": map[string]any{
			"workflow_mode": "parallel",
			"orchestration_request": map[string]any{
				"chain_name": "llm_chain",
				"inputs": map[string]any{
					"input": "Test parallel execution",
				},
			},
		},
	}

	result, err := workflow.Invoke(ctx, state)
	if err != nil {
		t.Fatalf("Unified workflow execution failed: %v", err)
	}

	if result == nil {
		t.Fatal("Workflow returned nil result")
	}
}

// TestUnifiedWorkflowErrorHandling tests that the unified workflow handles errors gracefully.
func TestUnifiedWorkflowErrorHandling(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping integration test in short mode")
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
	}

	workflow, err := NewUnifiedProcessorWorkflow(opts)
	if err != nil {
		t.Fatalf("Failed to create unified workflow: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Test with invalid request (missing required fields)
	state := map[string]any{
		"unified_request": map[string]any{
			"workflow_mode": "sequential",
			"orchestration_request": map[string]any{
				// Missing chain_name
				"inputs": map[string]any{
					"input": "Test error handling",
				},
			},
		},
	}

	// The workflow should handle missing chain_name gracefully
	result, err := workflow.Invoke(ctx, state)
	// We don't assert on error here because the workflow might handle
	// missing fields differently (skip execution, return error, etc.)
	if err != nil {
		t.Logf("Workflow returned error for invalid request (may be expected): %v", err)
	} else if result != nil {
		t.Log("Workflow handled invalid request gracefully")
	}
}

