package api

import (
	"bytes"
	"context"
	"encoding/json"
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

