package api

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/telemetry-exporter/pkg/exporter"
)

// ExportHandler handles on-demand trace export requests.
type ExportHandler struct {
	exportManager *exporter.ExportManager
	logger        *log.Logger
}

// NewExportHandler creates a new export handler.
func NewExportHandler(em *exporter.ExportManager, logger *log.Logger) *ExportHandler {
	return &ExportHandler{
		exportManager: em,
		logger:        logger,
	}
}

// ExportRequest represents an on-demand export request.
type ExportRequest struct {
	TimeRange    *TimeRange    `json:"time_range,omitempty"`
	ServiceFilter []string     `json:"service_filter,omitempty"`
	AgentTypeFilter []string   `json:"agent_type_filter,omitempty"`
	ExportFormat string        `json:"export_format,omitempty"` // "file", "signavio", "both"
	Destination  string        `json:"destination,omitempty"`    // File path or Signavio dataset
}

// TimeRange specifies a time range for export.
type TimeRange struct {
	StartTime time.Time `json:"start_time"`
	EndTime   time.Time `json:"end_time"`
}

// ExportResponse represents the response from an export request.
type ExportResponse struct {
	ExportID     string    `json:"export_id"`
	Status       string    `json:"status"` // "pending", "completed", "failed"
	FileLocation string    `json:"file_location,omitempty"`
	RecordCount  int       `json:"record_count,omitempty"`
	ExportTime   time.Time `json:"export_time"`
	Error        string    `json:"error,omitempty"`
}

// HandleExport handles POST /api/v1/traces/export
func (h *ExportHandler) HandleExport(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ExportRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	// Validate request
	if req.TimeRange == nil {
		http.Error(w, "time_range is required", http.StatusBadRequest)
		return
	}

	// TODO: Use ctx for actual export operation
	_, _ = context.WithTimeout(r.Context(), 5*time.Minute)

	// For now, return a simple response
	// In production, this would:
	// 1. Query traces from storage based on filters
	// 2. Export to requested format
	// 3. Return export status

	exportID := fmt.Sprintf("export-%d", time.Now().Unix())
	response := ExportResponse{
		ExportID:   exportID,
		Status:     "pending",
		ExportTime: time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	json.NewEncoder(w).Encode(response)
}

// HandleExportStatus handles GET /api/v1/traces/export/{export_id}
func (h *ExportHandler) HandleExportStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	// Extract export ID from path
	exportID := r.URL.Path[len("/api/v1/traces/export/"):]
	if exportID == "" {
		http.Error(w, "Export ID required", http.StatusBadRequest)
		return
	}

	// In production, query export status from storage
	response := ExportResponse{
		ExportID:   exportID,
		Status:     "completed",
		ExportTime: time.Now(),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// RegisterRoutes registers export API routes.
func (h *ExportHandler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/api/v1/traces/export", h.HandleExport)
	mux.HandleFunc("/api/v1/traces/export/", h.HandleExportStatus)
}

