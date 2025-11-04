package langchainservice

import (
	"context"
	"os"
	"testing"
	"time"
)

func TestClient_RunChain(t *testing.T) {
	baseURL := os.Getenv("LANGCHAIN_SERVICE_URL")
	if baseURL == "" {
		t.Skip("LANGCHAIN_SERVICE_URL not set; skipping integration test")
	}

	client := NewClient(baseURL)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	response, err := client.RunChain(ctx, "What persistence layers are supported?")
	if err != nil {
		t.Fatalf("RunChain error: %v", err)
	}
	if response.Result == "" {
		t.Fatalf("expected non-empty result")
	}
}
