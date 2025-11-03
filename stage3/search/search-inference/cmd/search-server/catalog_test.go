package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightclient"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/internal/catalog/flightcatalog"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/search"
)

func TestHandleAgentCatalogReturnsCachedSnapshot(t *testing.T) {
	service := &search.SearchService{}
	catalog := convertCatalog(sampleFlightCatalog())
	service.UpdateAgentCatalog(catalog)

	handler := handleAgentCatalog(service, nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/agent-catalog", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	suites, ok := payload["suites"].([]any)
	if !ok || len(suites) != 1 {
		t.Fatalf("expected suites array with one entry, got %v", payload["suites"])
	}
	tools, ok := payload["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("expected tools array with two entries, got %v", payload["tools"])
	}
	if updated, ok := payload["updated_at"].(string); !ok || updated == "" {
		t.Fatalf("expected updated_at timestamp, got %v", payload["updated_at"])
	}
	if summary, ok := payload["agent_catalog_summary"].(string); !ok || summary == "" {
		t.Fatalf("expected agent_catalog_summary string, got %v", payload["agent_catalog_summary"])
	}
	if context, ok := payload["agent_catalog_context"].(string); !ok || context == "" {
		t.Fatalf("expected agent_catalog_context string, got %v", payload["agent_catalog_context"])
	}
	stats, ok := payload["agent_catalog_stats"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent_catalog_stats object, got %v", payload["agent_catalog_stats"])
	}
	if suiteCount, ok := stats["suite_count"].(float64); !ok || suiteCount != 1 {
		t.Fatalf("expected suite_count 1, got %v", stats["suite_count"])
	}
	if unique, ok := payload["agent_catalog_unique_tools"].([]any); !ok || len(unique) == 0 {
		t.Fatalf("expected agent_catalog_unique_tools array, got %v", payload["agent_catalog_unique_tools"])
	}
	if details, ok := payload["agent_catalog_tool_details"].([]any); !ok || len(details) == 0 {
		t.Fatalf("expected agent_catalog_tool_details array, got %v", payload["agent_catalog_tool_details"])
	}
}

func TestHandleAgentCatalogStats(t *testing.T) {
	service := &search.SearchService{}
	catalog := convertCatalog(sampleFlightCatalog())
	service.UpdateAgentCatalog(catalog)

	handler := handleAgentCatalogStats(service)
	req := httptest.NewRequest(http.MethodGet, "/v1/agent-catalog/stats", nil)
	rr := httptest.NewRecorder()

	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", rr.Code)
	}

	var payload map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &payload); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if _, ok := payload["stats"].(map[string]any); !ok {
		t.Fatalf("expected stats object, got %v", payload["stats"])
	}
	if _, ok := payload["summary"].(string); !ok {
		t.Fatalf("expected summary string, got %v", payload["summary"])
	}
	if _, ok := payload["context"].(string); !ok {
		t.Fatalf("expected context string, got %v", payload["context"])
	}
	if unique, ok := payload["unique_tools"].([]any); !ok || len(unique) == 0 {
		t.Fatalf("expected unique_tools array, got %v", payload["unique_tools"])
	}
}

func sampleFlightCatalog() flightcatalog.Catalog {
	now := time.Date(2025, 1, 2, 15, 4, 5, 0, time.UTC)
	return flightcatalog.Catalog{
		Suites: []flightclient.ServiceSuiteInfo{
			{
				Name:           "SearchSuite",
				ToolNames:      []string{"search_documents", "rerank_results"},
				ToolCount:      2,
				Implementation: "search-runtime",
				Version:        "0.9.0",
				AttachedAt:     now,
			},
		},
		Tools: []flightclient.ToolInfo{
			{Name: "search_documents", Description: "Execute semantic search"},
			{Name: "rerank_results", Description: "Rerank search hits"},
		},
	}
}
