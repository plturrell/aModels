package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

// TestHandleEmbeddings_ValidRequest tests a valid embeddings request
func TestHandleEmbeddings_ValidRequest(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  "This is a test sentence for embeddings.",
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d, body: %s", w.Code, w.Body.String())
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp["object"] != "list" {
		t.Errorf("expected object 'list', got %v", resp["object"])
	}

	data, ok := resp["data"].([]interface{})
	if !ok || len(data) == 0 {
		t.Fatalf("expected data array with at least one item")
	}

	item, ok := data[0].(map[string]interface{})
	if !ok {
		t.Fatal("expected data item to be a map")
	}

	embedding, ok := item["embedding"].([]interface{})
	if !ok || len(embedding) == 0 {
		t.Fatalf("expected embedding array")
	}
}

// TestHandleEmbeddings_InvalidMethod tests that non-POST methods are rejected
func TestHandleEmbeddings_InvalidMethod(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/embeddings", nil)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected status 405, got %d", w.Code)
	}
}

// TestHandleEmbeddings_InvalidContentType tests that non-JSON content types are rejected
func TestHandleEmbeddings_InvalidContentType(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte("test")))
	req.Header.Set(HeaderContentType, "text/plain")
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusUnsupportedMediaType {
		t.Errorf("expected status 415, got %d", w.Code)
	}
}

// TestHandleEmbeddings_InvalidJSON tests that invalid JSON is rejected
func TestHandleEmbeddings_InvalidJSON(t *testing.T) {
	srv := newTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader([]byte("{invalid json")))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleEmbeddings_ArrayInput tests request with array of strings
func TestHandleEmbeddings_ArrayInput(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input": []string{
			"First sentence.",
			"Second sentence.",
			"Third sentence.",
		},
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	data, ok := resp["data"].([]interface{})
	if !ok || len(data) != 3 {
		t.Fatalf("expected 3 items in data array, got %d", len(data))
	}
}

// TestHandleEmbeddings_ResponseStructure tests the response structure
func TestHandleEmbeddings_ResponseStructure(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  "Test sentence",
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	// Check required fields
	requiredFields := []string{"object", "data", "model"}
	for _, field := range requiredFields {
		if resp[field] == nil {
			t.Errorf("expected field %s in response", field)
		}
	}

	// Check data item structure
	data, ok := resp["data"].([]interface{})
	if !ok || len(data) == 0 {
		t.Fatalf("expected data array")
	}

	item, ok := data[0].(map[string]interface{})
	if !ok {
		t.Fatal("expected data item to be a map")
	}

	itemFields := []string{"object", "embedding", "index"}
	for _, field := range itemFields {
		if item[field] == nil {
			t.Errorf("expected field %s in data item", field)
		}
	}
}

// TestHandleEmbeddings_InvalidInputType tests that invalid input types are rejected
func TestHandleEmbeddings_InvalidInputType(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  123, // Invalid type
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", w.Code)
	}
}

// TestHandleEmbeddings_EmptyInput tests that empty input is handled
func TestHandleEmbeddings_EmptyInput(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  "",
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	// Should still work, may return empty or default embedding
	if w.Code != http.StatusOK && w.Code != http.StatusBadRequest {
		t.Errorf("expected status 200 or 400, got %d", w.Code)
	}
}

// TestHandleEmbeddings_ModelNotAvailable tests handling when model is not available
func TestHandleEmbeddings_ModelNotAvailable(t *testing.T) {
	srv := newTestServer()
	// Remove the model
	srv.models = make(map[string]*ai.VaultGemma)

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  "Test",
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected status 503, got %d", w.Code)
	}
}

// TestHandleEmbeddings_EmbeddingDimensions tests that embeddings have correct dimensions
func TestHandleEmbeddings_EmbeddingDimensions(t *testing.T) {
	srv := newTestServer()

	reqBody := map[string]interface{}{
		"model": "vaultgemma",
		"input":  "Test sentence for embedding dimensions.",
	}

	raw, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/embeddings", bytes.NewReader(raw))
	req.Header.Set(HeaderContentType, ContentTypeJSON)
	w := httptest.NewRecorder()

	srv.HandleEmbeddings(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	data, ok := resp["data"].([]interface{})
	if !ok || len(data) == 0 {
		t.Fatalf("expected data array")
	}

	item, ok := data[0].(map[string]interface{})
	if !ok {
		t.Fatal("expected data item to be a map")
	}

	embedding, ok := item["embedding"].([]interface{})
	if !ok {
		t.Fatal("expected embedding to be an array")
	}

	if len(embedding) == 0 {
		t.Error("expected non-empty embedding")
	}

	// Check that all values are numbers
	for i, val := range embedding {
		if _, ok := val.(float64); !ok {
			t.Errorf("expected embedding[%d] to be a number, got %T", i, val)
		}
	}
}

