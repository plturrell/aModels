package autonomous

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/research"
)

// AutonomousHandler handles HTTP requests for the autonomous intelligence layer.
type AutonomousHandler struct {
	system *IntegratedAutonomousSystem
	logger *log.Logger
}

// NewAutonomousHandler creates a new autonomous handler.
func NewAutonomousHandler(
	deepResearchClient *research.DeepResearchClient,
	deepAgentsURL string,
	unifiedWorkflowURL string,
	logger *log.Logger,
) *AutonomousHandler {
	system := NewIntegratedAutonomousSystem(
		deepResearchClient,
		deepAgentsURL,
		unifiedWorkflowURL,
		nil, // DB will be injected via SetDB
		logger,
	)

	return &AutonomousHandler{
		system: system,
		logger: logger,
	}
}

// SetDB sets the database connection for the handler.
func (h *AutonomousHandler) SetDB(db interface{}) {
	// Recreate system with DB if needed
	// For now, we'll handle this in the main.go initialization
}

// HandleExecuteTask handles POST /api/autonomous/execute
func (h *AutonomousHandler) HandleExecuteTask(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		TaskID      string                 `json:"task_id"`
		Type        string                 `json:"type"`
		Description string                 `json:"description"`
		Query       string                 `json:"query"`
		Context     map[string]interface{} `json:"context"`
		AgentID     string                 `json:"agent_id,omitempty"`
		Priority    int                    `json:"priority,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	task := &AutonomousTask{
		ID:          req.TaskID,
		Type:        req.Type,
		Description: req.Description,
		Query:       req.Query,
		Context:     req.Context,
		AgentID:     req.AgentID,
		Priority:    req.Priority,
	}

	if task.ID == "" {
		task.ID = generateTaskID()
	}

	result, err := h.system.ExecuteWithGooseMigration(r.Context(), task)
	if err != nil {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error":   err.Error(),
			"success": false,
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

// HandleGetMetrics handles GET /api/autonomous/metrics
func (h *AutonomousHandler) HandleGetMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	metrics := h.system.GetPerformanceMetrics()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(metrics)
}

// HandleGetAgents handles GET /api/autonomous/agents
func (h *AutonomousHandler) HandleGetAgents(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	registry := h.system.GetAgentRegistry()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"agents": registry.agents,
		"total":  len(registry.agents),
	})
}

// HandleGetKnowledgeBase handles GET /api/autonomous/knowledge
func (h *AutonomousHandler) HandleGetKnowledgeBase(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	kb := h.system.GetKnowledgeBase()
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"patterns":      kb.patterns,
		"patterns_count": len(kb.patterns),
		"solutions":     kb.solutions,
		"solutions_count": len(kb.solutions),
		"best_practices": kb.bestPractices,
		"best_practices_count": len(kb.bestPractices),
	})
}

// generateTaskID generates a unique task ID.
func generateTaskID() string {
	return fmt.Sprintf("task_%d", time.Now().UnixNano())
}

