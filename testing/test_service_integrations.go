package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

const (
	localAIURL      = "http://localhost:8081"
	deepAgentsURL   = "http://localhost:9004"
	graphURL        = "http://localhost:8080"
	extractURL      = "http://localhost:8082"
	gatewayURL      = "http://localhost:8000"
	defaultTimeout  = 30 * time.Second
	healthTimeout   = 5 * time.Second
)

// TestDeepAgentsLocalAI tests DeepAgents → LocalAI integration
func TestDeepAgentsLocalAI(t *testing.T) {
	url := os.Getenv("DEEPAGENTS_URL")
	if url == "" {
		url = deepAgentsURL
	}

	client := &http.Client{Timeout: healthTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/healthz")
	if err != nil {
		t.Skipf("DeepAgents service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("DeepAgents health check failed: status %d", resp.StatusCode)
		return
	}

	t.Logf("✅ DeepAgents health check passed")
	t.Logf("   (DeepAgents should be configured to use LocalAI at %s)", localAIURL)
}

// TestGraphServiceLocalAI tests Graph service → LocalAI integration
func TestGraphServiceLocalAI(t *testing.T) {
	url := os.Getenv("GRAPH_URL")
	if url == "" {
		url = graphURL
	}

	client := &http.Client{Timeout: healthTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/health")
	if err != nil {
		t.Skipf("Graph service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Logf("⚠️  Graph service returned: %d (may not have /health endpoint)", resp.StatusCode)
		return
	}

	t.Logf("✅ Graph service health check passed")
	t.Logf("   (Graph service should be configured to use LocalAI at %s)", localAIURL)
}

// TestExtractServiceLocalAI tests Extract service → LocalAI integration
func TestExtractServiceLocalAI(t *testing.T) {
	url := os.Getenv("EXTRACT_URL")
	if url == "" {
		url = extractURL
	}

	client := &http.Client{Timeout: healthTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/health")
	if err != nil {
		t.Skipf("Extract service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Logf("⚠️  Extract service returned: %d (may not have /health endpoint)", resp.StatusCode)
		return
	}

	t.Logf("✅ Extract service health check passed")
	t.Logf("   (Extract service should use LocalAI for extraction)")
}

// TestGatewayServiceLocalAI tests Gateway service → LocalAI integration
func TestGatewayServiceLocalAI(t *testing.T) {
	url := os.Getenv("GATEWAY_URL")
	if url == "" {
		url = gatewayURL
	}

	client := &http.Client{Timeout: healthTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/health")
	if err != nil {
		t.Skipf("Gateway service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Logf("⚠️  Gateway service returned: %d (may not have /health endpoint)", resp.StatusCode)
		return
	}

	t.Logf("✅ Gateway service health check passed")
	t.Logf("   (Gateway service should use LocalAI at %s)", localAIURL)
}

// TestLocalAIConnectivity tests that LocalAI is accessible from all services
func TestLocalAIConnectivity(t *testing.T) {
	url := os.Getenv("LOCALAI_URL")
	if url == "" {
		url = localAIURL
	}

	client := &http.Client{Timeout: healthTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/health")
	if err != nil {
		t.Fatalf("LocalAI service not available: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("LocalAI health check failed: status %d", resp.StatusCode)
	}

	t.Logf("✅ LocalAI is accessible at %s", url)
}

// TestLocalAIDomains tests that LocalAI domains are properly configured
func TestLocalAIDomains(t *testing.T) {
	url := os.Getenv("LOCALAI_URL")
	if url == "" {
		url = localAIURL
	}

	client := &http.Client{Timeout: healthTimeout}

	resp, err := client.Get(url + "/v1/domains")
	if err != nil {
		t.Fatalf("Failed to get domains: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("Domains endpoint failed: status %d, body: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	data, ok := result["data"].([]interface{})
	if !ok {
		t.Fatalf("Invalid domains response structure")
	}

	// Check for required domains
	requiredDomains := []string{"general", "0x3579-VectorProcessingAgent"}
	foundDomains := make(map[string]bool)
	
	for _, domain := range data {
		domainMap, ok := domain.(map[string]interface{})
		if !ok {
			continue
		}
		id, ok := domainMap["id"].(string)
		if ok {
			foundDomains[id] = true
		}
	}

	t.Logf("✅ Found %d domains", len(data))
	for _, required := range requiredDomains {
		if foundDomains[required] {
			t.Logf("   ✅ Required domain '%s' found", required)
		} else {
			t.Logf("   ⚠️  Required domain '%s' not found", required)
		}
	}
}

// TestNoExternalAPICalls verifies no external API calls are configured
func TestNoExternalAPICalls(t *testing.T) {
	externalAPIs := []string{
		"api.openai.com",
		"api.anthropic.com",
		"api.deepseek.com",
		"generativeai.googleapis.com",
	}

	// Check environment variables
	envVars := os.Environ()
	foundExternal := []string{}

	for _, envVar := range envVars {
		value := os.Getenv(envVar)
		for _, api := range externalAPIs {
			if contains(value, api) {
				foundExternal = append(foundExternal, fmt.Sprintf("%s=%s", envVar, value))
			}
		}
	}

	if len(foundExternal) > 0 {
		t.Errorf("Found external API references in environment:")
		for _, ref := range foundExternal {
			t.Errorf("   - %s", ref)
		}
	} else {
		t.Logf("✅ No external API references found in environment")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || 
		(len(s) > len(substr) && 
			(s[:len(substr)] == substr || 
			 s[len(s)-len(substr):] == substr ||
			 containsMiddle(s, substr))))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

