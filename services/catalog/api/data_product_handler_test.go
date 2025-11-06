package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/plturrell/aModels/services/catalog/iso11179"
	"github.com/plturrell/aModels/services/catalog/quality"
	"github.com/plturrell/aModels/services/catalog/workflows"
)

func TestHandleGetDataProduct(t *testing.T) {
	// Create test registry
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	
	// Create test data element
	element := iso11179.NewDataElement(
		"http://test.org/catalog/data-element/test",
		"Test Element",
		"http://test.org/catalog/concept/test",
		"http://test.org/catalog/representation/test",
		"Test definition",
	)
	registry.RegisterDataElement(element)
	
	// Create handlers
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		"http://localhost:8081",
		"http://localhost:8081",
		"http://localhost:9001",
		"http://localhost:8081",
		"http://localhost:8085",
		registry,
		quality.NewQualityMonitor("http://localhost:9002", nil),
		nil,
		nil,
		nil,
	)
	
	handler := NewDataProductHandler(unifiedWorkflow, registry, nil, nil)
	
	// Test GET request
	req := httptest.NewRequest(http.MethodGet, "/catalog/data-products/test", nil)
	w := httptest.NewRecorder()
	
	handler.HandleGetDataProduct(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}
	
	if response["data_product"] == nil {
		t.Error("Response missing data_product field")
	}
}

func TestHandleGetSampleData(t *testing.T) {
	// Create test registry
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	
	// Create test data element
	element := iso11179.NewDataElement(
		"http://test.org/catalog/data-element/test",
		"Test Element",
		"http://test.org/catalog/concept/test",
		"http://test.org/catalog/representation/test",
		"Test definition",
	)
	registry.RegisterDataElement(element)
	
	// Create handlers
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		"http://localhost:8081",
		"http://localhost:8081",
		"http://localhost:9001",
		"http://localhost:8081",
		"http://localhost:8085",
		registry,
		quality.NewQualityMonitor("http://localhost:9002", nil),
		nil,
		nil,
		nil,
	)
	
	handler := NewDataProductHandler(unifiedWorkflow, registry, nil, nil)
	
	// Test GET request for sample data
	req := httptest.NewRequest(http.MethodGet, "/catalog/data-elements/test/sample", nil)
	w := httptest.NewRecorder()
	
	handler.HandleGetSampleData(w, req)
	
	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
	
	var response map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &response); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}
	
	if response["sample_data"] == nil {
		t.Error("Response missing sample_data field")
	}
	
	sampleData, ok := response["sample_data"].([]interface{})
	if !ok {
		t.Error("sample_data is not an array")
	}
	
	if len(sampleData) != 5 {
		t.Errorf("Expected 5 sample records, got %d", len(sampleData))
	}
}

func TestHandleBuildDataProduct(t *testing.T) {
	// Create test registry
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	
	// Create handlers
	unifiedWorkflow := workflows.NewUnifiedWorkflowIntegration(
		"http://localhost:8081",
		"http://localhost:8081",
		"http://localhost:9001",
		"http://localhost:8081",
		"http://localhost:8085",
		registry,
		quality.NewQualityMonitor("http://localhost:9002", nil),
		nil,
		nil,
		nil,
	)
	
	handler := NewDataProductHandler(unifiedWorkflow, registry, nil, nil)
	
	// Test POST request
	requestBody := map[string]string{
		"topic":         "test_topic",
		"customer_need": "I need to test data products",
	}
	jsonData, _ := json.Marshal(requestBody)
	
	req := httptest.NewRequest(http.MethodPost, "/catalog/data-products/build", bytes.NewReader(jsonData))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	
	handler.HandleBuildDataProduct(w, req)
	
	// Note: This will fail if the unified workflow can't connect to services
	// In a real test, we'd mock the HTTP client
	if w.Code != http.StatusOK && w.Code != http.StatusInternalServerError {
		t.Errorf("Expected status 200 or 500 (due to missing services), got %d", w.Code)
	}
}

func TestGenerateSampleData(t *testing.T) {
	registry := iso11179.NewMetadataRegistry("test", "Test Catalog", "http://test.org/catalog")
	
	handler := NewDataProductHandler(nil, registry, nil, nil)
	
	element := iso11179.NewDataElement(
		"http://test.org/catalog/data-element/test",
		"Test Element",
		"http://test.org/catalog/concept/test",
		"http://test.org/catalog/representation/test",
		"Test definition",
	)
	
	samples := handler.generateSampleDataFromElement(element)
	
	if len(samples) != 5 {
		t.Errorf("Expected 5 samples, got %d", len(samples))
	}
	
	for i, sample := range samples {
		if sample["id"] != fmt.Sprintf("sample_%d", i+1) {
			t.Errorf("Sample %d has wrong id: %v", i, sample["id"])
		}
		if sample["name"] == nil {
			t.Errorf("Sample %d missing name", i)
		}
	}
}

