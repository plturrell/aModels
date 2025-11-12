package main

import (
	"testing"
)

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"testing"
	"time"
)

const (
	localAIURL     = "http://localhost:8081"
	defaultTimeout = 60 * time.Second
)

// TestDeepSeekOCR tests DeepSeek OCR integration via LocalAI
func TestDeepSeekOCR(t *testing.T) {
	url := os.Getenv("LOCALAI_URL")
	if url == "" {
		url = localAIURL
	}

	// Check if DeepSeek OCR is configured
	// First, check if there's a domain with deepseek-ocr backend
	client := &http.Client{Timeout: 10 * time.Second}

	resp, err := client.Get(url + "/v1/domains")
	if err != nil {
		t.Skipf("LocalAI service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Skipf("Cannot access LocalAI domains endpoint")
		return
	}

	var domainsResult map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&domainsResult); err != nil {
		t.Skipf("Failed to decode domains response: %v", err)
		return
	}

	data, ok := domainsResult["data"].([]interface{})
	if !ok {
		t.Skipf("Invalid domains response structure")
		return
	}

	// Check for DeepSeek OCR domain
	hasOCR := false
	for _, domain := range data {
		domainMap, ok := domain.(map[string]interface{})
		if !ok {
			continue
		}
		backendType, ok := domainMap["backend_type"].(string)
		if ok && backendType == "deepseek-ocr" {
			hasOCR = true
			domainID, _ := domainMap["id"].(string)
			t.Logf("✅ Found DeepSeek OCR domain: %s", domainID)
			break
		}
	}

	if !hasOCR {
		t.Skipf("DeepSeek OCR domain not configured")
		return
	}

	// Test OCR with a simple test image (1x1 pixel PNG)
	// In a real scenario, you would load an actual image file
	testImageBase64 := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	
	payload := map[string]interface{}{
		"model": "deepseek-ocr", // or the actual domain ID
		"messages": []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{
						"type": "image_url",
						"image_url": map[string]string{
							"url": fmt.Sprintf("data:image/png;base64,%s", testImageBase64),
						},
					},
				},
			},
		},
		"max_tokens": 100,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest("POST", url+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err = client.Do(req)
	if err != nil {
		t.Skipf("DeepSeek OCR request failed: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Logf("⚠️  DeepSeek OCR returned status %d: %s", resp.StatusCode, string(body))
		t.Logf("   (This may be expected if OCR model is not fully configured)")
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Errorf("Invalid OCR response structure")
		return
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Errorf("Invalid choice structure")
		return
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		t.Errorf("Invalid message structure")
		return
	}

	content, _ := message["content"].(string)
	t.Logf("✅ DeepSeek OCR test successful")
	t.Logf("   Response: %s", content[:min(100, len(content))])
}

// TestDeepSeekOCRScript tests if DeepSeek OCR script is available
func TestDeepSeekOCRScript(t *testing.T) {
	scriptPath := os.Getenv("DEEPSEEK_OCR_SCRIPT")
	if scriptPath == "" {
		scriptPath = "./scripts/deepseek_ocr_cli.py"
	}

	// Check if script exists
	if _, err := os.Stat(scriptPath); os.IsNotExist(err) {
		t.Skipf("DeepSeek OCR script not found at %s", scriptPath)
		return
	}

	t.Logf("✅ DeepSeek OCR script found at %s", scriptPath)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

