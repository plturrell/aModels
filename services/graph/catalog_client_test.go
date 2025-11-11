package graph

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

func TestCatalogClient_RegisterDataElement_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/catalog/data-elements" {
			t.Errorf("Expected path /catalog/data-elements, got %s", r.URL.Path)
		}

		var req DataElementRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("Failed to decode request: %v", err)
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"identifier": req.Identifier,
		})
	}))
	defer server.Close()

	client := NewCatalogClient(server.URL, nil)
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

func TestCatalogClient_RegisterDataElementsBulk_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/catalog/data-elements/bulk" {
			t.Errorf("Expected path /catalog/data-elements/bulk, got %s", r.URL.Path)
		}

		var req []DataElementRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("Failed to decode request: %v", err)
		}

		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"created": len(req),
			"errors":  0,
		})
	}))
	defer server.Close()

	client := NewCatalogClient(server.URL, nil)
	ctx := context.Background()

	elements := []DataElementRequest{
		{
			Name:                 "Element 1",
			DataElementConceptID: "http://test/concept1",
			RepresentationID:     "http://test/representation1",
			Definition:           "Definition 1",
		},
	}

	err := client.RegisterDataElementsBulk(ctx, elements)
	if err != nil {
		t.Fatalf("Expected success, got error: %v", err)
	}
}

func TestCatalogClient_CircuitBreaker(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer server.Close()

	client := NewCatalogClient(server.URL, nil)
	client.circuitBreaker.threshold = 3
	client.circuitBreaker.timeout = 100 * time.Millisecond
	ctx := context.Background()

	element := DataElementRequest{
		Name:                 "Test",
		DataElementConceptID: "http://test/concept",
		RepresentationID:     "http://test/representation",
		Definition:           "Test",
	}

	// Trigger failures to open circuit
	for i := 0; i < 3; i++ {
		client.RegisterDataElement(ctx, element)
	}

	// Circuit should be open now
	err := client.RegisterDataElement(ctx, element)
	if err == nil {
		t.Error("Expected error when circuit is open")
	}

	// Wait for timeout
	time.Sleep(150 * time.Millisecond)

	// Should transition to half-open, but will fail again
	err = client.RegisterDataElement(ctx, element)
	if err == nil {
		t.Error("Expected error when circuit is half-open and request fails")
	}
}

func TestConvertISO11179ToRequest(t *testing.T) {
	element := iso11179.NewDataElement(
		"http://test/element",
		"Test Element",
		"http://test/concept",
		"http://test/representation",
		"Test definition",
	)
	element.AddMetadata("source", "murex")
	element.AddMetadata("domain", "finance")

	req := ConvertISO11179ToRequest(element)

	if req.Name != "Test Element" {
		t.Errorf("Expected name 'Test Element', got %s", req.Name)
	}

	if req.Metadata["source"] != "murex" {
		t.Errorf("Expected source 'murex', got %s", req.Metadata["source"])
	}

	if req.Metadata["domain"] != "finance" {
		t.Errorf("Expected domain 'finance', got %s", req.Metadata["domain"])
	}
}

