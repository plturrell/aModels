package boolq

import (
	"context"
	"os"
	"testing"
)

func TestSAPAIClient_GenerateCompletion_Integration(t *testing.T) {
	deploymentID := os.Getenv("AICORE_DEPLOYMENT_ID")
	if deploymentID == "" {
		t.Skip("Skipping integration test: AICORE_DEPLOYMENT_ID not set")
	}

	client := NewSAPAIClient(deploymentID, "gemmavault")
	if client.APIEndpoint == "" || client.AuthURL == "" || client.ClientID == "" || client.ClientSecret == "" {
		t.Fatal("SAP AI Core client not configured; check environment variables")
	}

	prompt := "What is the capital of France?"
	maxTokens := 50

	ctx := context.Background()
	completion, err := client.GenerateCompletion(ctx, prompt, maxTokens)

	if err != nil {
		t.Fatalf("GenerateCompletion failed: %v", err)
	}

	if completion == "" {
		t.Error("Expected a non-empty completion, but got an empty string")
	}

	t.Logf("Received completion: %s", completion)
}
