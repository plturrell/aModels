package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/internal/exporter"
	"github.com/plturrell/aModels/services/telemetry-exporter/internal/sources"
	"github.com/plturrell/aModels/services/testing"
)

// Server provides HTTP API handlers for the telemetry exporter.
type Server struct {
	exporter *exporter.SignavioExporter
	discovery *sources.UnifiedDiscovery
	agentName string
	logger    *log.Logger
}

// NewServer creates a new API server.
func NewServer(
	exporter *exporter.SignavioExporter,
	discovery *sources.UnifiedDiscovery,
	agentName string,
	logger *log.Logger,
) *Server {
	return &Server{
		exporter:  exporter,
		discovery: discovery,
		agentName: agentName,
		logger:    logger,
	}
}

// RegisterRoutes registers HTTP routes.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/export/session/", s.handleExportSession)
	mux.HandleFunc("/export/batch", s.handleExportBatch)
	mux.HandleFunc("/export/sessions", s.handleListSessions)
	mux.HandleFunc("/export/status/", s.handleExportStatus)
	mux.HandleFunc("/health", s.handleHealth)
}

// handleExportSession exports a single session on-demand.
// POST /export/session/{session-id}?source=extract|agent_telemetry
func (s *Server) handleExportSession(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract session ID from path
	path := r.URL.Path
	prefix := "/export/session/"
	if !strings.HasPrefix(path, prefix) {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return
	}

	sessionID := strings.TrimPrefix(path, prefix)
	if sessionID == "" {
		http.Error(w, "session ID is required", http.StatusBadRequest)
		return
	}

	// Get source from query parameter (default: extract)
	source := r.URL.Query().Get("source")
	if source == "" {
		source = "extract"
	}

	// Parse optional request body for dataset override
	var reqBody struct {
		Dataset string `json:"dataset,omitempty"`
	}
	if r.Body != nil && r.Body != http.NoBody {
		json.NewDecoder(r.Body).Decode(&reqBody)
		r.Body.Close()
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	// Fetch telemetry from source
	telemetry, err := s.discovery.GetSessionTelemetry(ctx, sessionID, source)
	if err != nil {
		s.logger.Printf("Error fetching telemetry for session %s: %v", sessionID, err)
		http.Error(w, fmt.Sprintf("failed to fetch telemetry: %v", err), http.StatusInternalServerError)
		return
	}

	// Format for Signavio
	record := exporter.FormatFromExtractService(telemetry, s.agentName)

	// Export to Signavio (use custom dataset if provided)
	dataset := reqBody.Dataset
	if dataset != "" {
		if err := s.exporter.ExportSessionToDataset(ctx, record, dataset); err != nil {
			s.logger.Printf("Error exporting session %s: %v", sessionID, err)
			http.Error(w, fmt.Sprintf("failed to export to Signavio: %v", err), http.StatusInternalServerError)
			return
		}
	} else {
		if err := s.exporter.ExportSession(ctx, record); err != nil {
			s.logger.Printf("Error exporting session %s: %v", sessionID, err)
			http.Error(w, fmt.Sprintf("failed to export to Signavio: %v", err), http.StatusInternalServerError)
			return
		}
	}

	response := map[string]any{
		"status":     "success",
		"session_id": sessionID,
		"source":     source,
		"exported_at": time.Now().Format(time.RFC3339),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleExportBatch exports multiple sessions.
// POST /export/batch
func (s *Server) handleExportBatch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		SessionIDs []string `json:"session_ids"`
		Source     string   `json:"source,omitempty"`
		Dataset    string   `json:"dataset,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	if len(req.SessionIDs) == 0 {
		http.Error(w, "session_ids is required and cannot be empty", http.StatusBadRequest)
		return
	}

	if req.Source == "" {
		req.Source = "extract"
	}

	ctx, cancel := context.WithTimeout(r.Context(), 60*time.Second)
	defer cancel()

	// Fetch and format all sessions
	records := make([]*testing.SignavioTelemetryRecord, 0, len(req.SessionIDs))
	results := make([]map[string]any, 0, len(req.SessionIDs))

	for _, sessionID := range req.SessionIDs {
		telemetry, err := s.discovery.GetSessionTelemetry(ctx, sessionID, req.Source)
		if err != nil {
			results = append(results, map[string]any{
				"session_id": sessionID,
				"status":     "error",
				"error":      err.Error(),
			})
			continue
		}

		record := exporter.FormatFromExtractService(telemetry, s.agentName)
		records = append(records, record)
		results = append(results, map[string]any{
			"session_id": sessionID,
			"status":     "ready",
		})
	}

	// Export batch (use custom dataset if provided)
	if len(records) > 0 {
		if req.Dataset != "" {
			if err := s.exporter.ExportBatchToDataset(ctx, records, req.Dataset); err != nil {
				http.Error(w, fmt.Sprintf("failed to export batch: %v", err), http.StatusInternalServerError)
				return
			}
		} else {
			if err := s.exporter.ExportBatch(ctx, records); err != nil {
				http.Error(w, fmt.Sprintf("failed to export batch: %v", err), http.StatusInternalServerError)
				return
			}
		}
	}

	response := map[string]any{
		"status":         "success",
		"total_count":    len(req.SessionIDs),
		"exported_count": len(records),
		"exported_at":    time.Now().Format(time.RFC3339),
		"results":        results,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleListSessions lists available sessions (discovery placeholder).
// GET /export/sessions
func (s *Server) handleListSessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// For now, return empty list - discovery not yet implemented
	response := map[string]any{
		"sessions": []string{},
		"message":  "Session discovery not yet implemented - provide session IDs manually",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleExportStatus checks export status for a session.
// GET /export/status/{session-id}
func (s *Server) handleExportStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	path := r.URL.Path
	prefix := "/export/status/"
	if !strings.HasPrefix(path, prefix) {
		http.Error(w, "invalid path", http.StatusBadRequest)
		return
	}

	sessionID := strings.TrimPrefix(path, prefix)
	if sessionID == "" {
		http.Error(w, "session ID is required", http.StatusBadRequest)
		return
	}

	exported := s.exporter.GetExportStatus(sessionID)

	response := map[string]any{
		"session_id": sessionID,
		"exported":   exported,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// handleHealth provides health check endpoint.
// GET /health
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()

	health := map[string]any{
		"status":  "healthy",
		"service": "telemetry-exporter",
	}

	// Check Signavio connection if enabled
	if s.exporter.IsEnabled() {
		if err := s.exporter.ValidateConnection(ctx); err != nil {
			health["status"] = "degraded"
			health["signavio_error"] = err.Error()
		} else {
			health["signavio"] = "connected"
		}
	} else {
		health["signavio"] = "disabled"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(health)
}

