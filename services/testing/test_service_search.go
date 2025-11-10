package testing

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// handleSearchScenarios searches for similar test scenarios.
func (ts *TestService) handleSearchScenarios(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		Query string `json:"query"`
		Limit int    `json:"limit,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	if req.Limit <= 0 {
		req.Limit = 10
	}

	if ts.searchClient == nil || !ts.searchClient.IsEnabled() {
		http.Error(w, "search is not enabled", http.StatusServiceUnavailable)
		return
	}

	scenarios, err := ts.searchClient.SearchScenarios(r.Context(), req.Query, req.Limit)
	if err != nil {
		http.Error(w, fmt.Sprintf("search failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]any{
		"query":     req.Query,
		"limit":     req.Limit,
		"scenarios": scenarios,
		"count":     len(scenarios),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleSearchPatterns searches for similar data patterns.
func (ts *TestService) handleSearchPatterns(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		TableName  string `json:"table_name"`
		ColumnName string `json:"column_name"`
		Limit      int    `json:"limit,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if req.TableName == "" || req.ColumnName == "" {
		http.Error(w, "table_name and column_name are required", http.StatusBadRequest)
		return
	}

	if req.Limit <= 0 {
		req.Limit = 10
	}

	if ts.searchClient == nil || !ts.searchClient.IsEnabled() {
		http.Error(w, "search is not enabled", http.StatusServiceUnavailable)
		return
	}

	patterns, err := ts.searchClient.SearchPatterns(r.Context(), req.TableName, req.ColumnName, req.Limit)
	if err != nil {
		http.Error(w, fmt.Sprintf("search failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]any{
		"table_name":  req.TableName,
		"column_name": req.ColumnName,
		"limit":       req.Limit,
		"patterns":    patterns,
		"count":       len(patterns),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleSearchKnowledgeGraph performs semantic search on the knowledge graph.
func (ts *TestService) handleSearchKnowledgeGraph(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		Query        string `json:"query"`
		ArtifactType string `json:"artifact_type,omitempty"`
		Limit        int    `json:"limit,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Query == "" {
		http.Error(w, "query is required", http.StatusBadRequest)
		return
	}

	if req.Limit <= 0 {
		req.Limit = 10
	}

	if ts.searchClient == nil || !ts.searchClient.IsEnabled() {
		http.Error(w, "search is not enabled", http.StatusServiceUnavailable)
		return
	}

	results, err := ts.searchClient.SearchKnowledgeGraph(r.Context(), req.Query, req.ArtifactType, req.Limit)
	if err != nil {
		http.Error(w, fmt.Sprintf("search failed: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]any{
		"query":         req.Query,
		"artifact_type": req.ArtifactType,
		"limit":         req.Limit,
		"results":       results,
		"count":         len(results),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

