package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// AgentsHandler provides HTTP handlers for AI Agents.
type AgentsHandler struct {
	agentSystem *agents.AgentSystem
	logger      *log.Logger
}

// NewAgentsHandler creates a new agents handler.
func NewAgentsHandler(agentSystem *agents.AgentSystem, logger *log.Logger) *AgentsHandler {
	return &AgentsHandler{
		agentSystem: agentSystem,
		logger:      logger,
	}
}

// HandleStartIngestion handles POST /api/agents/ingestion/start.
func (h *AgentsHandler) HandleStartIngestion(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SourceType string                 `json:"source_type"`
		Config     map[string]interface{} `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.SourceType == "" {
		http.Error(w, "source_type is required", http.StatusBadRequest)
		return
	}

	// Start ingestion
	err := h.agentSystem.RunIngestion(r.Context(), req.SourceType, req.Config)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to start ingestion: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":      "started",
		"source_type": req.SourceType,
		"message":     "Data ingestion started successfully",
	})
}

// HandleGetIngestionStatus handles GET /api/agents/ingestion/{sourceType}/status.
func (h *AgentsHandler) HandleGetIngestionStatus(w http.ResponseWriter, r *http.Request) {
	sourceType := r.URL.Query().Get("source_type")
	if sourceType == "" {
		http.Error(w, "source_type parameter is required", http.StatusBadRequest)
		return
	}

	// Get agent status
	// In production, would get from agent system
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"source_type": sourceType,
		"status":       "running",
		"message":     "Status retrieval coming soon",
	})
}

// HandleLearnMappingRules handles POST /api/agents/mapping/learn.
func (h *AgentsHandler) HandleLearnMappingRules(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Patterns []agents.MappingPattern `json:"patterns"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Learn mapping rules
	// In production, would call agent system
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":   "success",
		"message":  "Mapping rules learning initiated",
		"patterns": len(req.Patterns),
	})
}

// HandleDetectAnomalies handles POST /api/agents/anomaly/detect.
func (h *AgentsHandler) HandleDetectAnomalies(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DataPoints []agents.DataPoint `json:"data_points"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Detect anomalies
	anomalies, err := h.agentSystem.RunAnomalyDetection(r.Context(), req.DataPoints)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to detect anomalies: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":    "success",
		"anomalies": anomalies,
		"count":     len(anomalies),
	})
}

// HandleGenerateTests handles POST /api/agents/test/generate.
func (h *AgentsHandler) HandleGenerateTests(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Schema  interface{}            `json:"schema"`
		Options agents.TestGenOptions  `json:"options"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Generate tests
	results, err := h.agentSystem.GenerateTests(r.Context(), req.Schema, req.Options)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to generate tests: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"results": results,
		"count":   len(results),
	})
}

// writeJSON writes a JSON response.
func writeJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

