package testing

import (
	// "context" // Unused
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// TestService provides HTTP API for test execution.
type TestService struct {
	generator    *SampleGenerator
	logger       *log.Logger
	searchClient *SearchClient
}

// NewTestService creates a new test service.
func NewTestService(generator *SampleGenerator, searchClient *SearchClient, logger *log.Logger) *TestService {
	return &TestService{
		generator:    generator,
		logger:       logger,
		searchClient: searchClient,
	}
}

// RegisterRoutes registers HTTP routes for the test service.
func (ts *TestService) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/test/generate-sample", ts.handleGenerateSample)
	mux.HandleFunc("/test/execute-scenario", ts.handleExecuteScenario)
	mux.HandleFunc("/test/load-knowledge-graph", ts.handleLoadKnowledgeGraph)
	mux.HandleFunc("/test/executions", ts.handleListExecutions)
	
	// Signavio endpoints (must be registered before generic /test/executions/ handler)
	mux.HandleFunc("/test/export-signavio-batch", ts.handleExportSignavioBatch)
	mux.HandleFunc("/test/signavio/health", ts.handleSignavioHealth)
	
	// Execution detail endpoints (with path parameters) - route based on suffix
	mux.HandleFunc("/test/executions/", ts.handleExecutionDetail)
	
	mux.HandleFunc("/test/search-scenarios", ts.handleSearchScenarios)
	mux.HandleFunc("/test/search-patterns", ts.handleSearchPatterns)
	mux.HandleFunc("/test/search-knowledge-graph", ts.handleSearchKnowledgeGraph)
}

// handleExecutionDetail routes execution detail requests based on path suffix.
func (ts *TestService) handleExecutionDetail(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	
	// Route based on path suffix
	if strings.HasSuffix(path, "/export-signavio") {
		ts.handleExportToSignavio(w, r)
		return
	}
	
	if strings.HasSuffix(path, "/signavio-metrics") {
		ts.handleGetSignavioMetrics(w, r)
		return
	}
	
	// Default: get execution details
	ts.handleGetExecution(w, r)
}

// handleGenerateSample generates sample data for a table.
func (ts *TestService) handleGenerateSample(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		TableName string                 `json:"table_name"`
		RowCount  int                    `json:"row_count"`
		SeedData  map[string][]any       `json:"seed_data,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	config := &TableTestConfig{
		TableName: req.TableName,
		RowCount:  req.RowCount,
		SeedData:  req.SeedData,
	}

	data, err := ts.generator.GenerateSampleData(r.Context(), config)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to generate sample data: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]any{
		"table_name": req.TableName,
		"row_count":  len(data),
		"data":       data,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleExecuteScenario executes a test scenario.
func (ts *TestService) handleExecuteScenario(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var scenario TestScenario
	if err := json.NewDecoder(r.Body).Decode(&scenario); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	execution, err := ts.generator.ExecuteTestScenario(r.Context(), &scenario)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to execute scenario: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(execution)
}

// handleLoadKnowledgeGraph loads knowledge graph data.
func (ts *TestService) handleLoadKnowledgeGraph(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	defer r.Body.Close()

	var req struct {
		ProjectID string `json:"project_id"`
		SystemID  string `json:"system_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if err := ts.generator.LoadKnowledgeGraph(r.Context(), req.ProjectID, req.SystemID); err != nil {
		http.Error(w, fmt.Sprintf("failed to load knowledge graph: %v", err), http.StatusInternalServerError)
		return
	}

	response := map[string]any{
		"status":      "success",
		"tables":      len(ts.generator.knowledgeGraph.Tables),
		"columns":     len(ts.generator.knowledgeGraph.Columns),
		"last_updated": ts.generator.knowledgeGraph.LastUpdated,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleListExecutions lists test executions.
func (ts *TestService) handleListExecutions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	query := "SELECT id, scenario_id, status, start_time, end_time FROM test_executions ORDER BY start_time DESC LIMIT 100"
	rows, err := ts.generator.db.QueryContext(r.Context(), query)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to query executions: %v", err), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	executions := []map[string]any{}
	for rows.Next() {
		var id, scenarioID, status string
		var startTime, endTime time.Time
		if err := rows.Scan(&id, &scenarioID, &status, &startTime, &endTime); err != nil {
			continue
		}
		executions = append(executions, map[string]any{
			"id":          id,
			"scenario_id": scenarioID,
			"status":      status,
			"start_time":  startTime,
			"end_time":    endTime,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{"executions": executions})
}

// handleGetExecution gets a specific test execution.
func (ts *TestService) handleGetExecution(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract execution ID from path
	executionID := r.URL.Path[len("/test/executions/"):]
	
	var id, scenarioID, status string
	var startTime, endTime time.Time
	var metricsJSON, issuesJSON, resultsJSON string

	query := "SELECT id, scenario_id, status, start_time, end_time, metrics_json, quality_issues_json, results_json FROM test_executions WHERE id = $1"
	err := ts.generator.db.QueryRowContext(r.Context(), query, executionID).Scan(
		&id, &scenarioID, &status, &startTime, &endTime, &metricsJSON, &issuesJSON, &resultsJSON,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			http.Error(w, "execution not found", http.StatusNotFound)
			return
		}
		http.Error(w, fmt.Sprintf("failed to query execution: %v", err), http.StatusInternalServerError)
		return
	}

	var metrics ExecutionMetrics
	json.Unmarshal([]byte(metricsJSON), &metrics)
	
	var issues []QualityIssue
	json.Unmarshal([]byte(issuesJSON), &issues)
	
	var results map[string]any
	json.Unmarshal([]byte(resultsJSON), &results)

	execution := map[string]any{
		"id":            id,
		"scenario_id":   scenarioID,
		"status":        status,
		"start_time":    startTime,
		"end_time":      endTime,
		"metrics":       metrics,
		"quality_issues": issues,
		"results":       results,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(execution)
}

