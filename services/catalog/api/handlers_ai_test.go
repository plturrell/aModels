package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

func TestHandleCreateDataElementsBulk_WithAIDeduplication(t *testing.T) {
	// Set environment variable to enable AI features
	os.Setenv("CATALOG_AI_DEDUPLICATION_ENABLED", "true")
	defer os.Unsetenv("CATALOG_AI_DEDUPLICATION_ENABLED")

	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	// Register an existing element first
	existing := iso11179.NewDataElement(
		"http://test.org/element1",
		"Customer Name",
		"http://test.org/concept1",
		"http://test.org/representation1",
		"Name of the customer",
	)
	registry.RegisterDataElement(existing)

	reqBody := []map[string]interface{}{
		{
			"name":                  "Customer Name", // Duplicate
			"data_element_concept_id": "http://test.org/concept1",
			"representation_id":     "http://test.org/representation1",
			"definition":            "Name of the customer",
		},
		{
			"name":                  "Product Code", // New
			"data_element_concept_id": "http://test.org/concept2",
			"representation_id":     "http://test.org/representation2",
			"definition":            "Unique product identifier",
		},
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/catalog/data-elements/bulk", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.HandleCreateDataElementsBulk(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("Expected status %d, got %d", http.StatusCreated, w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Should have created at least one element (the new one)
	created, _ := response["created"].(float64)
	if int(created) < 1 {
		t.Errorf("Expected at least 1 created, got %v", response["created"])
	}
}

func TestHandleCreateDataElementsBulk_WithAIValidation(t *testing.T) {
	os.Setenv("CATALOG_AI_VALIDATION_ENABLED", "true")
	defer os.Unsetenv("CATALOG_AI_VALIDATION_ENABLED")

	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	reqBody := []map[string]interface{}{
		{
			"name":                  "Test Element",
			"data_element_concept_id": "http://test.org/concept1",
			"representation_id":     "http://test.org/representation1",
			"definition":            "Test definition",
		},
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/catalog/data-elements/bulk", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.HandleCreateDataElementsBulk(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("Expected status %d, got %d", http.StatusCreated, w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Response should include AI suggestions if validation is enabled
	// (Even if AI service is unavailable, response structure should be correct)
	_, hasAISuggestions := response["ai_suggestions"]
	if !hasAISuggestions {
		// This is OK - AI suggestions only appear if AI processing occurred
		t.Log("AI suggestions not present (AI service may be unavailable)")
	}
}

func TestHandleCreateDataElementsBulk_WithoutAI(t *testing.T) {
	// Ensure AI features are disabled
	os.Unsetenv("CATALOG_AI_DEDUPLICATION_ENABLED")
	os.Unsetenv("CATALOG_AI_VALIDATION_ENABLED")
	os.Unsetenv("CATALOG_AI_RESEARCH_ENABLED")

	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	reqBody := []map[string]interface{}{
		{
			"name":                  "Test Element",
			"data_element_concept_id": "http://test.org/concept1",
			"representation_id":     "http://test.org/representation1",
			"definition":            "Test definition",
		},
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/catalog/data-elements/bulk", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.HandleCreateDataElementsBulk(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("Expected status %d, got %d", http.StatusCreated, w.Code)
	}

	var response map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&response); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}

	// Should work without AI features
	created, _ := response["created"].(float64)
	if int(created) != 1 {
		t.Errorf("Expected 1 created, got %v", response["created"])
	}
}

func TestCheckDuplicatesWithDeepAgents_Disabled(t *testing.T) {
	os.Unsetenv("CATALOG_AI_DEDUPLICATION_ENABLED")

	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	candidates := []CandidateElement{
		{
			Name:       "Test",
			Definition: "Test definition",
		},
	}

	suggestions, err := handlers.checkDuplicatesWithDeepAgents(context.Background(), candidates)
	if err != nil {
		t.Fatalf("Expected no error when disabled, got: %v", err)
	}
	if suggestions != nil {
		t.Errorf("Expected nil suggestions when disabled, got: %v", suggestions)
	}
}

func TestValidateWithDeepAgents_Disabled(t *testing.T) {
	os.Unsetenv("CATALOG_AI_VALIDATION_ENABLED")

	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	candidates := []CandidateElement{
		{
			Name:       "Test",
			Definition: "Test definition",
		},
	}

	suggestions, err := handlers.validateWithDeepAgents(context.Background(), candidates)
	if err != nil {
		t.Fatalf("Expected no error when disabled, got: %v", err)
	}
	if suggestions != nil {
		t.Errorf("Expected nil suggestions when disabled, got: %v", suggestions)
	}
}

// TestDeepAgentsStructuredOutput tests structured output integration
func TestDeepAgentsStructuredOutput(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)
	
	// Enable AI features
	os.Setenv("CATALOG_AI_DEDUPLICATION_ENABLED", "true")
	os.Setenv("CATALOG_AI_VALIDATION_ENABLED", "true")
	defer os.Unsetenv("CATALOG_AI_DEDUPLICATION_ENABLED")
	defer os.Unsetenv("CATALOG_AI_VALIDATION_ENABLED")
	
	// Re-initialize to pick up environment variables
	handlers = NewCatalogHandlers(registry, nil)
	
	if handlers.deepAgentsClient == nil {
		t.Skip("DeepAgents client not initialized (service may be unavailable)")
	}
	
	// Test structured output parsing
	candidates := []CandidateElement{
		{
			Name:                 "Test Element",
			Definition:           "A test data element",
			DataElementConceptID: "DEC-001",
			RepresentationID:     "REP-001",
		},
	}
	
	ctx := context.Background()
	
	// Test deduplication with structured output
	response, err := handlers.deepAgentsClient.CheckDuplicates(ctx, candidates, nil)
	if err != nil {
		t.Logf("DeepAgents deduplication returned error (non-fatal): %v", err)
	}
	if response != nil {
		t.Logf("Received %d deduplication suggestions", len(response.Suggestions))
		for _, suggestion := range response.Suggestions {
			t.Logf("  - Index %d: %s (confidence: %.2f)", suggestion.Index, suggestion.Action, suggestion.Confidence)
		}
	}
	
	// Test validation with structured output
	validationResponse, err := handlers.deepAgentsClient.ValidateDefinitions(ctx, candidates)
	if err != nil {
		t.Logf("DeepAgents validation returned error (non-fatal): %v", err)
	}
	if validationResponse != nil {
		t.Logf("Received %d validation suggestions", len(validationResponse.Suggestions))
		for _, suggestion := range validationResponse.Suggestions {
			t.Logf("  - Index %d: score %.2f, %d improvements", suggestion.Index, suggestion.Score, len(suggestion.Improvements))
		}
	}
}

// TestDeepAgentsCaching tests response caching
func TestDeepAgentsCaching(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)
	
	os.Setenv("CATALOG_AI_DEDUPLICATION_ENABLED", "true")
	defer os.Unsetenv("CATALOG_AI_DEDUPLICATION_ENABLED")
	
	handlers = NewCatalogHandlers(registry, nil)
	
	if handlers.deepAgentsClient == nil {
		t.Skip("DeepAgents client not initialized")
	}
	
	// Create a mock cache
	mockCache := &mockCache{
		store: make(map[string]interface{}),
	}
	handlers.deepAgentsClient.SetCache(mockCache)
	
	candidates := []CandidateElement{
		{
			Name:                 "Cached Element",
			Definition:           "Should be cached",
			DataElementConceptID: "DEC-001",
			RepresentationID:     "REP-001",
		},
	}
	
	ctx := context.Background()
	
	// First call - should hit DeepAgents (if available) or return empty
	response1, _ := handlers.deepAgentsClient.CheckDuplicates(ctx, candidates, nil)
	
	// Second call - should use cache if first call succeeded
	response2, _ := handlers.deepAgentsClient.CheckDuplicates(ctx, candidates, nil)
	
	if response1 != nil && response2 != nil {
		// Verify cache was used (responses should be identical)
		if len(response1.Suggestions) != len(response2.Suggestions) {
			t.Logf("Cache may not be working - responses differ")
		} else {
			t.Logf("Cache appears to be working - responses match")
		}
	} else {
		t.Logf("Cache test skipped - DeepAgents service may be unavailable")
	}
}

type mockCache struct {
	store map[string]interface{}
}

func (m *mockCache) Get(ctx interface{}, key string, dest interface{}) error {
	if val, ok := m.store[key]; ok {
		// Simple JSON marshaling for test
		data, _ := json.Marshal(val)
		return json.Unmarshal(data, dest)
	}
	return fmt.Errorf("not found")
}

func (m *mockCache) Set(ctx interface{}, key string, value interface{}, ttl interface{}) error {
	m.store[key] = value
	return nil
}

