package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/domain"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/inference"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/models/ai"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Models/maths/util"
	"golang.org/x/time/rate"
	"time"
)

// TestHandleChat_ValidRequest tests a valid chat completion request
func TestHandleChat_ValidRequest(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello, how are you?"},
		},
		"max_tokens": 32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d, body: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp["object"] != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got %v", resp["object"])
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("expected choices array with at least one choice")
	}
}

// TestHandleChat_InvalidMethod tests that non-POST methods are rejected
func TestHandleChat_InvalidMethod(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", w.Code)
	}
}

// TestHandleChat_InvalidContentType tests that non-JSON content types are rejected
func TestHandleChat_InvalidContentType(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("test")))
	req.Header.Set(HeaderContentType, "text/plain")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusUnsupportedMediaType {
		t.Errorf("expected status 415, got %d", w.Code)
	}
}

// TestHandleChat_InvalidJSON tests that invalid JSON is rejected
func TestHandleChat_InvalidJSON(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("{invalid json")))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}

	var errorResp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&errorResp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	if errorResp["error"] == nil {
		t.Error("expected error in response")
	}
}

// TestHandleChat_EmptyMessages tests that empty messages array is rejected
func TestHandleChat_EmptyMessages(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model":    "general",
		"messages": []map[string]string{},
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleChat_AutoDomainDetection tests automatic domain detection
func TestHandleChat_AutoDomainDetection(t *testing.T) {
	srv := newTestServer()

	// Add a domain with specific keywords
	srv.domainManager.AddDomain("sql", &domain.DomainConfig{
		Name:        "SQL",
		ModelPath:   "models/sql",
		MaxTokens:   64,
		Temperature: 0.6,
		Keywords:    []string{"sql", "database", "query"},
	})

	reqBody := map[string]interface{}{
		"model": DomainAuto,
		"messages": []map[string]string{
			{"role": "user", "content": "How do I write a SQL query?"},
		},
		"max_tokens": 32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	// Should still work, may use general domain as fallback
	if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError {
		t.Errorf("expected status 200 or 500, got %d", w.Code)
	}
}

// TestHandleChat_WithUserID tests request with user ID header
func TestHandleChat_WithUserID(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test message"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	req.Header.Set(HeaderUserID, "test-user-123")
	req.Header.Set(HeaderSessionID, "test-session-456")
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleChat_ModelNotFound tests handling when model is not found
func TestHandleChat_ModelNotFound(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "nonexistent-model",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	// Should return error or use default domain
	if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError {
		t.Errorf("expected status 200 or 500, got %d", w.Code)
	}
}

// TestHandleChat_WithTemperature tests request with custom temperature
func TestHandleChat_WithTemperature(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"temperature": 0.9,
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	metadata, ok := resp["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("expected metadata in response")
	}

	if metadata["top_p"] == nil {
		t.Error("expected top_p in metadata")
	}
}

// TestHandleChat_WithTopPAndTopK tests request with custom top_p and top_k
func TestHandleChat_WithTopPAndTopK(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"top_p":     0.95,
		"top_k":     40,
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleChat_MultipleMessages tests request with multiple messages
func TestHandleChat_MultipleMessages(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "What is AI?"},
			{"role": "assistant", "content": "AI is artificial intelligence."},
			{"role": "user", "content": "Tell me more."},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("expected choices array")
	}
}

// TestHandleChat_ResponseStructure tests the response structure
func TestHandleChat_ResponseStructure(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Check required fields
	requiredFields := []string{"id", "object", "created", "model", "choices", "usage"}
	for _, field := range requiredFields {
		if resp[field] == nil {
			t.Errorf("expected field %s in response", field)
		}
	}

	// Check usage structure
	usage, ok := resp["usage"].(map[string]interface{})
	if !ok {
		t.Fatal("expected usage to be a map")
	}

	usageFields := []string{"prompt_tokens", "completion_tokens", "total_tokens"}
	for _, field := range usageFields {
		if usage[field] == nil {
			t.Errorf("expected field %s in usage", field)
		}
	}
}

// TestHandleChat_BackendRouting tests routing to different backends
func TestHandleChat_BackendRouting(t *testing.T) {
	srv := newTestServer()

	// Test with safetensors backend (default)
	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	metadata, ok := resp["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("expected metadata in response")
	}

	backendType, ok := metadata["backend_type"].(string)
	if !ok {
		t.Error("expected backend_type in metadata")
	}

	if backendType == "" {
		t.Error("backend_type should not be empty")
	}
}

// TestHandleChat_ErrorHandling tests error handling scenarios
func TestHandleChat_ErrorHandling(t *testing.T) {
	t.Run("InvalidJSON", func(t *testing.T) {
		srv := newTestServer()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("{invalid")))
		req.Header.Set(HeaderContentType, ContentTypeJSON)
		w := httptest.NewRecorder()

		srv.HandleChat(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})

	t.Run("EmptyBody", func(t *testing.T) {
		srv := newTestServer()
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("")))
		req.Header.Set(HeaderContentType, ContentTypeJSON)
		w := httptest.NewRecorder()

		srv.HandleChat(w, req)

		if w.Code != http.StatusBadRequest {
			t.Errorf("expected status 400, got %d", w.Code)
		}
	})
}

// TestHandleChat_WithDomains tests request with available domains
func TestHandleChat_WithDomains(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"domains":    []string{"general", "sql"},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleChat_ResponseMetadata tests response metadata structure
func TestHandleChat_ResponseMetadata(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	metadata, ok := resp["metadata"].(map[string]interface{})
	if !ok {
		t.Fatal("expected metadata in response")
	}

	// Check metadata fields
	expectedFields := []string{"model_key", "cache_hit", "top_p", "top_k", "backend_type", "fallback_used"}
	for _, field := range expectedFields {
		if metadata[field] == nil {
			t.Errorf("expected field %s in metadata", field)
		}
	}
}

// TestHandleChat_ContentGeneration tests that content is actually generated
func TestHandleChat_ContentGeneration(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Say hello"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("expected choices array")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatal("expected choice to be a map")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		t.Fatal("expected message in choice")
	}

	content, ok := message["content"].(string)
	if !ok {
		t.Fatal("expected content in message")
	}

	if content == "" {
		t.Error("expected non-empty content")
	}
}

