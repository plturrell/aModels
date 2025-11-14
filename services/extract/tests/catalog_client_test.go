package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

func TestCatalogClient_RegisterDataElement_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/catalog/data-elements" {
			t.Errorf("Expected path /catalog/data-elements, got %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("Expected POST, got %s", r.Method)
		}

		var req DataElementRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("Failed to decode request: %v", err)
		}

		if req.Name != "Test Element" {
			t.Errorf("Expected name 'Test Element', got %s", req.Name)
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"identifier": req.Identifier,
			"name":       req.Name,
		})
	}))
	defer server.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	client := NewCatalogClient(server.URL, logger)
	ctx := context.Background()

	element := DataElementRequest{
		Name:                 "Test Element",
		DataElementConceptID: "http://test/concept",
		RepresentationID:     "http://test/representation",
		Definition:           "Test definition",
		Identifier:           "http://test/element",
	}

	err := client.RegisterDataElement(ctx, element)
	if err != nil {
		t.Fatalf("Expected success, got error: %v", err)
	}
}

func TestCatalogClient_RegisterDataElement_Retry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		if attempts < 2 {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusCreated)
	}))
	defer server.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	client := NewCatalogClient(server.URL, logger)
	client.maxRetries = 3
	client.retryDelay = 10 * time.Millisecond // Fast for testing
	ctx := context.Background()

	element := DataElementRequest{
		Name:                 "Test Element",
		DataElementConceptID: "http://test/concept",
		RepresentationID:     "http://test/representation",
		Definition:           "Test definition",
		Identifier:           "http://test/element",
	}

	err := client.RegisterDataElement(ctx, element)
	if err != nil {
		t.Fatalf("Expected success after retry, got error: %v", err)
	}

	if attempts < 2 {
		t.Errorf("Expected at least 2 attempts, got %d", attempts)
	}
}

func TestCatalogClient_RegisterDataElement_ClientError_NoRetry(t *testing.T) {
	attempts := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempts++
		w.WriteHeader(http.StatusBadRequest)
		json.NewEncoder(w).Encode(map[string]string{"error": "bad request"})
	}))
	defer server.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	client := NewCatalogClient(server.URL, logger)
	client.maxRetries = 3
	client.retryDelay = 10 * time.Millisecond
	ctx := context.Background()

	element := DataElementRequest{
		Name:                 "Test Element",
		DataElementConceptID: "http://test/concept",
		RepresentationID:     "http://test/representation",
		Definition:           "Test definition",
		Identifier:           "http://test/element",
	}

	err := client.RegisterDataElement(ctx, element)
	if err == nil {
		t.Fatal("Expected error for bad request")
	}

	if attempts != 1 {
		t.Errorf("Expected 1 attempt (no retry for 4xx), got %d", attempts)
	}
}

func TestCatalogClient_RegisterDataElementsBulk_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/catalog/data-elements/bulk" {
			t.Errorf("Expected path /catalog/data-elements/bulk, got %s", r.URL.Path)
		}

		var req []DataElementRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("Failed to decode request: %v", err)
		}

		if len(req) != 2 {
			t.Errorf("Expected 2 elements, got %d", len(req))
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"created": 2,
			"errors":  0,
		})
	}))
	defer server.Close()

	logger := log.New(os.Stderr, "[test] ", log.LstdFlags)
	client := NewCatalogClient(server.URL, logger)
	ctx := context.Background()

	elements := []DataElementRequest{
		{
			Name:                 "Element 1",
			DataElementConceptID: "http://test/concept1",
			RepresentationID:     "http://test/representation1",
			Definition:           "Definition 1",
			Identifier:           "http://test/element1",
		},
		{
			Name:                 "Element 2",
			DataElementConceptID: "http://test/concept2",
			RepresentationID:     "http://test/representation2",
			Definition:           "Definition 2",
			Identifier:           "http://test/element2",
		},
	}

	err := client.RegisterDataElementsBulk(ctx, elements)
	if err != nil {
		t.Fatalf("Expected success, got error: %v", err)
	}
}

func TestCatalogClient_Disabled(t *testing.T) {
	client := NewCatalogClient("", nil)
	ctx := context.Background()

	element := DataElementRequest{
		Name:                 "Test Element",
		DataElementConceptID: "http://test/concept",
		RepresentationID:     "http://test/representation",
		Definition:           "Test definition",
		Identifier:           "http://test/element",
	}

	err := client.RegisterDataElement(ctx, element)
	if err != nil {
		t.Fatalf("Expected no error when disabled, got: %v", err)
	}
}

func TestConvertNodeToDataElement(t *testing.T) {
	node := Node{
		ID:    "test-node-123",
		Type:  graph.NodeTypeColumn,
		Label: "Customer Name",
		Props: map[string]interface{}{
			"description": "Name of the customer",
			"data_type":   "string",
			"nullable":    true,
		},
	}

	element := ConvertNodeToDataElement(node, "project-1", "system-1")

	if element.Name != "Customer Name" {
		t.Errorf("Expected name 'Customer Name', got %s", element.Name)
	}

	if element.Metadata["source"] != "extract-service" {
		t.Errorf("Expected source 'extract-service', got %s", element.Metadata["source"])
	}

	if element.Metadata["project_id"] != "project-1" {
		t.Errorf("Expected project_id 'project-1', got %s", element.Metadata["project_id"])
	}

	if element.Metadata["data_type"] != "string" {
		t.Errorf("Expected data_type 'string', got %s", element.Metadata["data_type"])
	}
}

