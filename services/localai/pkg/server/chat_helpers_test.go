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

// TestValidateChatRequest_ValidRequest tests validation of valid requests
func TestValidateChatRequest_ValidRequest(t *testing.T) {
	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello"},
		},
		"max_tokens": 32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)

	chatReq, err := validateChatRequest(req)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if chatReq.Model != "general" {
		t.Errorf("expected model 'general', got %s", chatReq.Model)
	}

	if len(chatReq.Messages) != 1 {
		t.Errorf("expected 1 message, got %d", len(chatReq.Messages))
	}
}

// TestValidateChatRequest_InvalidMethod tests validation rejects non-POST methods
func TestValidateChatRequest_InvalidMethod(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)

	_, err := validateChatRequest(req)
	if err == nil {
		t.Fatal("expected error for non-POST method")
	}

	if !strings.Contains(err.Error(), "POST") {
		t.Errorf("expected error to mention POST, got %v", err)
	}
}

// TestValidateChatRequest_InvalidContentType tests validation rejects non-JSON content types
func TestValidateChatRequest_InvalidContentType(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("test")))
	req.Header.Set(HeaderContentType, "text/plain")

	_, err := validateChatRequest(req)
	if err == nil {
		t.Fatal("expected error for non-JSON content type")
	}
}

// TestValidateChatRequest_InvalidJSON tests validation rejects invalid JSON
func TestValidateChatRequest_InvalidJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte("{invalid")))
	req.Header.Set(HeaderContentType, ContentTypeJSON)

	_, err := validateChatRequest(req)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

// TestValidateChatRequest_EmptyMessages tests validation rejects empty messages
func TestValidateChatRequest_EmptyMessages(t *testing.T) {
	reqBody := map[string]interface{}{
		"model":    "general",
		"messages": []map[string]string{},
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)

	_, err := validateChatRequest(req)
	if err == nil {
		t.Fatal("expected error for empty messages")
	}
}

// TestBuildPromptFromMessages tests building prompt from messages
func TestBuildPromptFromMessages(t *testing.T) {
	messages := []ChatMessageInternal{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
		{Role: "user", Content: "How are you?"},
	}

	prompt := buildPromptFromMessages(messages)

	expected := "Hello\nHi there\nHow are you?\n"
	if prompt != expected {
		t.Errorf("expected prompt %q, got %q", expected, prompt)
	}
}

// TestBuildPromptFromMessages_Empty tests building prompt from empty messages
func TestBuildPromptFromMessages_Empty(t *testing.T) {
	messages := []ChatMessageInternal{}

	prompt := buildPromptFromMessages(messages)

	if prompt != "" {
		t.Errorf("expected empty prompt, got %q", prompt)
	}
}

// TestResolveModelForDomain tests model resolution with fallback
func TestResolveModelForDomain(t *testing.T) {
	srv := newTestServer()

	// Test with existing model
	model, modelKey, fallbackUsed, fallbackKey, err := srv.resolveModelForDomain("general", nil, "safetensors")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if model == nil {
		t.Fatal("expected model to be resolved")
	}

	if modelKey != "general" {
		t.Errorf("expected modelKey 'general', got %s", modelKey)
	}

	if fallbackUsed {
		t.Error("expected no fallback for existing model")
	}

	if fallbackKey != "" {
		t.Errorf("expected empty fallbackKey, got %s", fallbackKey)
	}
}

// TestResolveModelForDomain_WithFallback tests model resolution with fallback
func TestResolveModelForDomain_WithFallback(t *testing.T) {
	srv := newTestServer()

	// Add domain with fallback
	domainConfig := &domain.DomainConfig{
		Name:          "SQL",
		ModelPath:     "models/sql",
		FallbackModel: "general",
	}

	model, modelKey, fallbackUsed, fallbackKey, err := srv.resolveModelForDomain("sql", domainConfig, "safetensors")
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	if model == nil {
		t.Fatal("expected model to be resolved via fallback")
	}

	if !fallbackUsed {
		t.Error("expected fallback to be used")
	}

	if fallbackKey != "general" {
		t.Errorf("expected fallbackKey 'general', got %s", fallbackKey)
	}

	if modelKey != "general" {
		t.Errorf("expected modelKey 'general', got %s", modelKey)
	}
}

// TestResolveModelForDomain_TransformersBackend tests model resolution for transformers backend
func TestResolveModelForDomain_TransformersBackend(t *testing.T) {
	srv := newTestServer()

	domainConfig := &domain.DomainConfig{
		Name:       "Phi",
		ModelPath:  "models/phi",
		BackendType: BackendTypeTransformers,
	}

	model, modelKey, fallbackUsed, fallbackKey, err := srv.resolveModelForDomain("phi", domainConfig, BackendTypeTransformers)
	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}

	// Transformers backend doesn't require safetensors model
	if model != nil {
		t.Error("expected nil model for transformers backend")
	}

	if modelKey != "phi" {
		t.Errorf("expected modelKey 'phi', got %s", modelKey)
	}

	if fallbackUsed {
		t.Error("expected no fallback for transformers backend")
	}

	if fallbackKey != "" {
		t.Errorf("expected empty fallbackKey, got %s", fallbackKey)
	}
}

// TestResolveModelForDomain_NoModelAvailable tests error when no model is available
func TestResolveModelForDomain_NoModelAvailable(t *testing.T) {
	srv := newTestServer()
	// Remove all models
	srv.models = make(map[string]*ai.VaultGemma)

	domainConfig := &domain.DomainConfig{
		Name:     "Nonexistent",
		ModelPath: "models/nonexistent",
	}

	_, _, _, _, err := srv.resolveModelForDomain("nonexistent", domainConfig, "safetensors")
	if err == nil {
		t.Fatal("expected error when no model is available")
	}

	if !strings.Contains(err.Error(), "no models available") {
		t.Errorf("expected error about no models, got %v", err)
	}
}

// TestBuildChatResponse tests building chat response
func TestBuildChatResponse(t *testing.T) {
	modelKey := "general"
	content := "Hello, this is a test response."
	tokensUsed := 10
	prompt := "Hello"
	metadata := map[string]interface{}{
		"backend_type": "safetensors",
		"cache_hit":    false,
	}

	resp := buildChatResponse(modelKey, content, tokensUsed, prompt, metadata)

	if resp["object"] != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got %v", resp["object"])
	}

	if resp["model"] != modelKey {
		t.Errorf("expected model %s, got %v", modelKey, resp["model"])
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatal("expected choices array")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatal("expected choice to be a map")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		t.Fatal("expected message in choice")
	}

	if message["content"] != content {
		t.Errorf("expected content %q, got %v", content, message["content"])
	}

	usage, ok := resp["usage"].(map[string]interface{})
	if !ok {
		t.Fatal("expected usage in response")
	}

	if usage["completion_tokens"] != tokensUsed {
		t.Errorf("expected completion_tokens %d, got %v", tokensUsed, usage["completion_tokens"])
	}
}

// TestHandleChatError tests error handling function
func TestHandleChatError(t *testing.T) {
	w := httptest.NewRecorder()

	handleChatError(w, ErrInvalidRequest, http.StatusBadRequest)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}

	var errorResp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&errorResp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}

	errorObj, ok := errorResp["error"].(map[string]interface{})
	if !ok {
		t.Fatal("expected error object in response")
	}

	if errorObj["code"] != ErrorCodeInvalidRequest {
		t.Errorf("expected error code %s, got %v", ErrorCodeInvalidRequest, errorObj["code"])
	}
}

// TestHandleChatError_DifferentErrorTypes tests different error types
func TestHandleChatError_DifferentErrorTypes(t *testing.T) {
	testCases := []struct {
		err        error
		statusCode int
		errorCode  string
	}{
		{ErrInvalidRequest, http.StatusBadRequest, ErrorCodeInvalidRequest},
		{ErrModelNotFound, http.StatusInternalServerError, ErrorCodeModelNotFound},
		{ErrBackendUnavailable, http.StatusBadGateway, ErrorCodeBackendUnavailable},
		{ErrTimeout, http.StatusRequestTimeout, ErrorCodeTimeout},
		{ErrInternalError, http.StatusInternalServerError, ErrorCodeInternalError},
	}

	for _, tc := range testCases {
		t.Run(tc.errorCode, func(t *testing.T) {
			w := httptest.NewRecorder()
			handleChatError(w, tc.err, tc.statusCode)

			if w.Code != tc.statusCode {
				t.Errorf("expected status %d, got %d", tc.statusCode, w.Code)
			}

			var errorResp map[string]interface{}
			if err := json.NewDecoder(w.Body).Decode(&errorResp); err != nil {
				t.Fatalf("failed to decode error response: %v", err)
			}

			errorObj, ok := errorResp["error"].(map[string]interface{})
			if !ok {
				t.Fatal("expected error object")
			}

			if errorObj["code"] != tc.errorCode {
				t.Errorf("expected error code %s, got %v", tc.errorCode, errorObj["code"])
			}
		})
	}
}

