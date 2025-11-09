package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

func TestHandleCreateDataElementsBulk_Success(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	reqBody := []map[string]interface{}{
		{
			"name":                  "Test Element 1",
			"data_element_concept_id": "http://test.org/concept1",
			"representation_id":     "http://test.org/representation1",
			"definition":            "Test definition 1",
		},
		{
			"name":                  "Test Element 2",
			"data_element_concept_id": "http://test.org/concept2",
			"representation_id":     "http://test.org/representation2",
			"definition":            "Test definition 2",
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

	created, ok := response["created"].(float64)
	if !ok || int(created) != 2 {
		t.Errorf("Expected 2 created, got %v", response["created"])
	}

	errors, ok := response["errors"].(float64)
	if !ok || int(errors) != 0 {
		t.Errorf("Expected 0 errors, got %v", response["errors"])
	}
}

func TestHandleCreateDataElementsBulk_PartialFailure(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	reqBody := []map[string]interface{}{
		{
			"name":                  "Valid Element",
			"data_element_concept_id": "http://test.org/concept1",
			"representation_id":     "http://test.org/representation1",
			"definition":            "Valid definition",
		},
		{
			// Missing required fields
			"name": "Invalid Element",
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

	created, _ := response["created"].(float64)
	if int(created) != 1 {
		t.Errorf("Expected 1 created, got %v", response["created"])
	}

	errors, _ := response["errors"].(float64)
	if int(errors) != 1 {
		t.Errorf("Expected 1 error, got %v", response["errors"])
	}
}

func TestHandleCreateDataElementsBulk_EmptyRequest(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org")
	handlers := NewCatalogHandlers(registry, nil)

	reqBody := []map[string]interface{}{}
	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/catalog/data-elements/bulk", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	handlers.HandleCreateDataElementsBulk(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("Expected status %d, got %d", http.StatusBadRequest, w.Code)
	}
}

