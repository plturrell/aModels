package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/search"
)

func TestHandleModelInfoIncludesAgentCatalog(t *testing.T) {
	svc := &search.SearchService{}
	model := &search.SearchModelWithLocalAI{SearchModel: &search.SearchModel{}}
	catalog := search.AgentCatalog{
		Suites: []search.AgentSuite{{Name: "agentic", ToolNames: []string{"tool"}, ToolCount: 1}},
		Tools:  []search.AgentTool{{Name: "tool", Description: "desc"}},
	}
	svc.UpdateAgentCatalog(catalog)

	srv := NewSearchServer(model, svc)

	req := httptest.NewRequest(http.MethodGet, "/v1/model", nil)
	w := httptest.NewRecorder()

	srv.HandleModelInfo(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp struct {
		Info map[string]any `json:"info"`
	}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	catalogVal, ok := resp.Info["agent_catalog"].([]any)
	if !ok || len(catalogVal) == 0 {
		t.Fatalf("expected agent_catalog array, got %v", resp.Info["agent_catalog"])
	}
	if summary, ok := resp.Info["agent_catalog_summary"].(string); !ok || summary == "" {
		t.Fatalf("expected agent_catalog_summary string, got %v", resp.Info["agent_catalog_summary"])
	}
	if context, ok := resp.Info["agent_catalog_context"].(string); !ok || context == "" {
		t.Fatalf("expected agent_catalog_context string, got %v", resp.Info["agent_catalog_context"])
	}
	stats, ok := resp.Info["agent_catalog_stats"].(map[string]any)
	if !ok {
		t.Fatalf("expected agent_catalog_stats object, got %v", resp.Info["agent_catalog_stats"])
	}
	if suiteCount, ok := stats["suite_count"].(float64); !ok || suiteCount != 1 {
		t.Fatalf("expected suite_count 1, got %v", stats["suite_count"])
	}
	if unique, ok := resp.Info["agent_catalog_unique_tools"].([]any); !ok || len(unique) == 0 {
		t.Fatalf("expected agent_catalog_unique_tools array, got %v", resp.Info["agent_catalog_unique_tools"])
	}
	if details, ok := resp.Info["agent_catalog_tool_details"].([]any); !ok || len(details) == 0 {
		t.Fatalf("expected agent_catalog_tool_details array, got %v", resp.Info["agent_catalog_tool_details"])
	}
}
