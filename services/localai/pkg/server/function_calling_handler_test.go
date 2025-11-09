package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestHandleFunctionCalling_ValidRequest tests a valid function calling request
func TestHandleFunctionCalling_ValidRequest(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "What is the weather?"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get the current weather",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"location": map[string]interface{}{
								"type":        "string",
								"description": "The city and state",
							},
						},
					},
				},
			},
		},
		"max_tokens": 32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

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
		t.Fatalf("expected choices array")
	}
}

// TestHandleFunctionCalling_InvalidMethod tests that non-POST methods are rejected
func TestHandleFunctionCalling_InvalidMethod(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions/function-calling", nil)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_InvalidContentType tests that non-JSON content types are rejected
func TestHandleFunctionCalling_InvalidContentType(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader([]byte("test")))
	req.Header.Set(HeaderContentType, "text/plain")
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusUnsupportedMediaType {
		t.Errorf("expected status 415, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_InvalidJSON tests that invalid JSON is rejected
func TestHandleFunctionCalling_InvalidJSON(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader([]byte("{invalid json")))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_EmptyMessages tests that empty messages array is rejected
func TestHandleFunctionCalling_EmptyMessages(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model":    "general",
		"messages": []map[string]string{},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name": "test_function",
				},
			},
		},
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_NoTools tests request without tools
func TestHandleFunctionCalling_NoTools(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	// Should still work, just without function calling
	if w.Code != http.StatusOK && w.Code != http.StatusBadRequest {
		t.Errorf("expected status 200 or 400, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_MultipleTools tests request with multiple tools
func TestHandleFunctionCalling_MultipleTools(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Get weather and time"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather",
				},
			},
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_time",
					"description": "Get current time",
				},
			},
		},
		"max_tokens": 32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_ResponseStructure tests the response structure
func TestHandleFunctionCalling_ResponseStructure(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name": "test_function",
				},
			},
		},
		"max_tokens": 16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Check required fields
	requiredFields := []string{"id", "object", "created", "model", "choices"}
	for _, field := range requiredFields {
		if resp[field] == nil {
			t.Errorf("expected field %s in response", field)
		}
	}
}

// TestHandleFunctionCalling_ToolChoice tests request with tool_choice parameter
func TestHandleFunctionCalling_ToolChoice(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Get weather"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "get_weather",
					"description": "Get weather",
				},
			},
		},
		"tool_choice": "auto",
		"max_tokens":  32,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleFunctionCalling_WithTemperature tests function calling with custom temperature
func TestHandleFunctionCalling_WithTemperature(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name": "test_function",
				},
			},
		},
		"temperature": 0.8,
		"max_tokens":  16,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/function-calling", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleFunctionCalling(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

