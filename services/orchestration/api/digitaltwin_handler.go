package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/orchestration/digitaltwin"
)

// DigitalTwinHandler provides HTTP handlers for Digital Twin.
type DigitalTwinHandler struct {
	twinSystem *digitaltwin.DigitalTwinSystem
	logger     *log.Logger
}

// NewDigitalTwinHandler creates a new digital twin handler.
func NewDigitalTwinHandler(twinSystem *digitaltwin.DigitalTwinSystem, logger *log.Logger) *DigitalTwinHandler {
	return &DigitalTwinHandler{
		twinSystem: twinSystem,
		logger:     logger,
	}
}

// HandleCreateTwin handles POST /api/digitaltwin/create.
func (h *DigitalTwinHandler) HandleCreateTwin(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DataProductID string `json:"data_product_id"`
		Name          string `json:"name"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.DataProductID == "" {
		http.Error(w, "data_product_id is required", http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		req.Name = fmt.Sprintf("Twin for %s", req.DataProductID)
	}

	// Create twin
	twin, err := h.twinSystem.CreateTwinFromDataProduct(r.Context(), req.DataProductID, req.Name)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create twin: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"twin": map[string]interface{}{
			"id":         twin.ID,
			"name":       twin.Name,
			"type":       twin.Type,
			"source_id":  twin.SourceID,
			"version":    twin.Version,
			"status":     twin.State.Status,
		},
		"message": "Digital twin created successfully",
	})
}

// HandleStartSimulation handles POST /api/digitaltwin/{id}/simulate.
func (h *DigitalTwinHandler) HandleStartSimulation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract twin ID from path
	twinID := extractIDFromPath(r.URL.Path, "/api/digitaltwin/", "/simulate")
	if twinID == "" {
		http.Error(w, "twin ID is required", http.StatusBadRequest)
		return
	}

	var req struct {
		Type   string                           `json:"type"`
		Config digitaltwin.SimulationConfig     `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Type == "" {
		req.Type = "pipeline"
	}

	simReq := digitaltwin.StartSimulationRequest{
		TwinID: twinID,
		Type:   req.Type,
		Config: req.Config,
	}

	// Start simulation
	simulation, err := h.twinSystem.GetSimulationEngine().StartSimulation(r.Context(), simReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to start simulation: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"simulation": map[string]interface{}{
			"id":     simulation.ID,
			"twin_id": simulation.TwinID,
			"type":   simulation.Type,
			"status": simulation.Status,
		},
		"message": "Simulation started successfully",
	})
}

// HandleStartStressTest handles POST /api/digitaltwin/{id}/stress-test.
func (h *DigitalTwinHandler) HandleStartStressTest(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	twinID := extractIDFromPath(r.URL.Path, "/api/digitaltwin/", "/stress-test")
	if twinID == "" {
		http.Error(w, "twin ID is required", http.StatusBadRequest)
		return
	}

	var req struct {
		Config digitaltwin.StressTestConfig `json:"config"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	stressReq := digitaltwin.StressTestRequest{
		TwinID: twinID,
		Config: req.Config,
	}

	// Start stress test
	test, err := h.twinSystem.GetStressTester().RunStressTest(r.Context(), stressReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to start stress test: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"stress_test": map[string]interface{}{
			"id":      test.ID,
			"twin_id": test.TwinID,
			"status":  test.Status,
		},
		"message": "Stress test started successfully",
	})
}

// HandleStartRehearsal handles POST /api/digitaltwin/{id}/rehearse.
func (h *DigitalTwinHandler) HandleStartRehearsal(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	twinID := extractIDFromPath(r.URL.Path, "/api/digitaltwin/", "/rehearse")
	if twinID == "" {
		http.Error(w, "twin ID is required", http.StatusBadRequest)
		return
	}

	var req struct {
		Change        digitaltwin.Change `json:"change"`
		RunSimulation bool              `json:"run_simulation"`
		RunStressTest bool              `json:"run_stress_test"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Start rehearsal
	rehearsal, err := h.twinSystem.RehearseChange(r.Context(), twinID, req.Change, req.RunSimulation, req.RunStressTest)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to start rehearsal: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"rehearsal": map[string]interface{}{
			"id":          rehearsal.ID,
			"twin_id":     rehearsal.TwinID,
			"change_id":   rehearsal.ChangeID,
			"status":      rehearsal.Status,
			"recommendation": rehearsal.Recommendation,
		},
		"message": "Rehearsal started successfully",
	})
}

// extractIDFromPath extracts an ID from a URL path.
func extractIDFromPath(path, prefix, suffix string) string {
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	path = strings.TrimPrefix(path, prefix)
	if suffix != "" {
		if idx := strings.Index(path, suffix); idx != -1 {
			path = path[:idx]
		}
	}
	return path
}

