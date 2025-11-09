package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestHandleStreamingChat_ValidRequest tests a valid streaming chat request
func TestHandleStreamingChat_ValidRequest(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Hello, how are you?"},
		},
		"max_tokens": 32,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d, body: %s", w.Code, w.Body.String())
	}

	// Check content type
	if w.Header().Get(HeaderContentType) != ContentTypeSSE {
		t.Errorf("expected content type %s, got %s", ContentTypeSSE, w.Header().Get(HeaderContentType))
	}

	// Check that response contains SSE format
	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Error("expected SSE format with 'data:' prefix")
	}
}

// TestHandleStreamingChat_InvalidMethod tests that non-POST methods are rejected
func TestHandleStreamingChat_InvalidMethod(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions/stream", nil)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", w.Code)
	}
}

// TestHandleStreamingChat_InvalidContentType tests that non-JSON content types are rejected
func TestHandleStreamingChat_InvalidContentType(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader([]byte("test")))
	req.Header.Set(HeaderContentType, "text/plain")
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusUnsupportedMediaType {
		t.Errorf("expected status 415, got %d", w.Code)
	}
}

// TestHandleStreamingChat_InvalidJSON tests that invalid JSON is rejected
func TestHandleStreamingChat_InvalidJSON(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader([]byte("{invalid json")))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleStreamingChat_EmptyMessages tests that empty messages array is rejected
func TestHandleStreamingChat_EmptyMessages(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model":    "general",
		"messages": []map[string]string{},
		"stream":   true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleStreamingChat_SSEFormat tests that response follows SSE format
func TestHandleStreamingChat_SSEFormat(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Parse SSE format
	scanner := bufio.NewScanner(w.Body)
	chunkCount := 0
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			chunkCount++
			// Extract JSON from data: prefix
			jsonStr := strings.TrimPrefix(line, "data: ")
			if jsonStr != "" && jsonStr != "[DONE]" {
				var chunk map[string]interface{}
				if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
					t.Errorf("failed to parse SSE chunk JSON: %v", err)
				}
			}
		}
	}

	if chunkCount == 0 {
		t.Error("expected at least one SSE chunk")
	}
}

// TestHandleStreamingChat_WithTemperature tests streaming with custom temperature
func TestHandleStreamingChat_WithTemperature(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"temperature": 0.9,
		"max_tokens": 16,
		"stream":      true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleStreamingChat_MultipleMessages tests streaming with multiple messages
func TestHandleStreamingChat_MultipleMessages(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "system", "content": "You are helpful."},
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

// TestHandleStreamingChat_ChunkStructure tests the structure of streaming chunks
func TestHandleStreamingChat_ChunkStructure(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Parse and validate chunks
	scanner := bufio.NewScanner(w.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data:") {
			jsonStr := strings.TrimPrefix(line, "data: ")
			if jsonStr == "" || jsonStr == "[DONE]" {
				continue
			}

			var chunk map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &chunk); err != nil {
				t.Errorf("failed to parse chunk: %v", err)
				continue
			}

			// Check required fields
			if chunk["id"] == nil {
				t.Error("expected 'id' in chunk")
			}
			if chunk["object"] == nil {
				t.Error("expected 'object' in chunk")
			}
			if chunk["model"] == nil {
				t.Error("expected 'model' in chunk")
			}
		}
	}
}

// TestHandleStreamingChat_Headers tests that correct headers are set
func TestHandleStreamingChat_Headers(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Check headers
	if w.Header().Get(HeaderContentType) != ContentTypeSSE {
		t.Errorf("expected Content-Type %s, got %s", ContentTypeSSE, w.Header().Get(HeaderContentType))
	}

	if w.Header().Get(HeaderCacheControl) != "no-cache" {
		t.Error("expected Cache-Control: no-cache")
	}

	if w.Header().Get(HeaderConnection) != "keep-alive" {
		t.Error("expected Connection: keep-alive")
	}
}

// TestHandleStreamingChat_DoneMarker tests that [DONE] marker is sent at the end
func TestHandleStreamingChat_DoneMarker(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "general",
		"messages": []map[string]string{
			{"role": "user", "content": "Test"},
		},
		"max_tokens": 16,
		"stream":     true,
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions/stream", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleStreamingChat(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	// Check for [DONE] marker
	body := w.Body.String()
	if !strings.Contains(body, "[DONE]") {
		t.Error("expected [DONE] marker in streaming response")
	}
}

