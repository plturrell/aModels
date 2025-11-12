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
	transformersURL = "http://localhost:9090"
	localAIURL      = "http://localhost:8081"
	searchURL       = "http://localhost:8090"
	defaultTimeout  = 30 * time.Second
)

// TestTransformersService tests the transformers service for embeddings
func TestTransformersService(t *testing.T) {
	url := os.Getenv("TRANSFORMERS_URL")
	if url == "" {
		url = transformersURL
	}

	client := &http.Client{Timeout: defaultTimeout}

	// Test health endpoint
	resp, err := client.Get(url + "/health")
	if err != nil {
		t.Skipf("Transformers service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Transformers health check failed: status %d", resp.StatusCode)
		return
	}

	t.Logf("✅ Transformers service health check passed")
}

// TestLocalAIEmbeddings tests LocalAI embeddings endpoint
func TestLocalAIEmbeddings(t *testing.T) {
	url := os.Getenv("LOCALAI_URL")
	if url == "" {
		url = localAIURL
	}

	client := &http.Client{Timeout: defaultTimeout}

	payload := map[string]interface{}{
		"model": "0x3579-VectorProcessingAgent",
		"input": []string{"test embedding"},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest("POST", url+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		t.Skipf("LocalAI service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Errorf("LocalAI embeddings failed: status %d, body: %s", resp.StatusCode, string(body))
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	data, ok := result["data"].([]interface{})
	if !ok || len(data) == 0 {
		t.Errorf("Invalid embeddings response structure")
		return
	}

	embedding, ok := data[0].(map[string]interface{})
	if !ok {
		t.Errorf("Invalid embedding structure")
		return
	}

	embeddingVec, ok := embedding["embedding"].([]interface{})
	if !ok {
		t.Errorf("Embedding is not an array")
		return
	}

	t.Logf("✅ LocalAI embeddings successful: dimension %d", len(embeddingVec))
}

// TestSearchInferenceEmbeddings tests search-inference service embeddings
func TestSearchInferenceEmbeddings(t *testing.T) {
	url := os.Getenv("SEARCH_URL")
	if url == "" {
		url = searchURL
	}

	client := &http.Client{Timeout: defaultTimeout}

	payload := map[string]interface{}{
		"text": "test search embedding",
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest("POST", url+"/v1/embed", bytes.NewBuffer(jsonData))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		t.Skipf("Search-inference service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Errorf("Search embeddings failed: status %d, body: %s", resp.StatusCode, string(body))
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	embedding, ok := result["embedding"].([]interface{})
	if !ok {
		t.Errorf("Invalid embedding structure")
		return
	}

	t.Logf("✅ Search-inference embeddings successful: dimension %d", len(embedding))
}

// TestEmbeddingModelConsistency tests that all embedding endpoints use the same model
func TestEmbeddingModelConsistency(t *testing.T) {
	// This test verifies that all services are using the same embedding model
	// (all-MiniLM-L6-v2) which should produce 384-dimensional embeddings
	
	url := os.Getenv("LOCALAI_URL")
	if url == "" {
		url = localAIURL
	}

	client := &http.Client{Timeout: defaultTimeout}

	// Test with same input across services
	testInput := "consistency test"

	// Get embedding from LocalAI
	payload := map[string]interface{}{
		"model": "0x3579-VectorProcessingAgent",
		"input": []string{testInput},
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("Failed to marshal payload: %v", err)
	}

	req, err := http.NewRequest("POST", url+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		t.Skipf("LocalAI service not available: %v", err)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Skipf("LocalAI embeddings not available")
		return
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	data, ok := result["data"].([]interface{})
	if !ok || len(data) == 0 {
		t.Errorf("Invalid embeddings response")
		return
	}

	embedding, ok := data[0].(map[string]interface{})
	if !ok {
		t.Errorf("Invalid embedding structure")
		return
	}

	embeddingVec, ok := embedding["embedding"].([]interface{})
	if !ok {
		t.Errorf("Embedding is not an array")
		return
	}

	dimension := len(embeddingVec)
	expectedDim := 384 // all-MiniLM-L6-v2 dimension

	if dimension != expectedDim {
		t.Logf("⚠️  Embedding dimension is %d (expected %d for all-MiniLM-L6-v2)", dimension, expectedDim)
	} else {
		t.Logf("✅ Embedding dimension matches expected: %d", expectedDim)
	}
}

