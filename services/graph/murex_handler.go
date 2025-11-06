package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// MurexHandler provides HTTP handlers for Murex integration.
type MurexHandler struct {
	integration *MurexIntegration
	logger      *log.Logger
}

// NewMurexHandler creates a new Murex handler.
func NewMurexHandler(integration *MurexIntegration, logger *log.Logger) *MurexHandler {
	return &MurexHandler{
		integration: integration,
		logger:      logger,
	}
}

// HandleSync handles POST /integrations/murex/sync - Full synchronization.
func (h *MurexHandler) HandleSync(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Minute)
	defer cancel()

	if h.logger != nil {
		h.logger.Printf("Starting Murex full synchronization")
	}

	if err := h.integration.SyncFullSync(ctx); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Synchronization failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Murex synchronization completed successfully",
	})
}

// HandleIngestTrades handles POST /integrations/murex/trades - Ingest trades.
func (h *MurexHandler) HandleIngestTrades(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Filters map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		h.respondError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if h.logger != nil {
		h.logger.Printf("Ingesting Murex trades with filters: %v", req.Filters)
	}

	if err := h.integration.IngestTrades(ctx, req.Filters); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Trade ingestion failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Trades ingested successfully",
	})
}

// HandleIngestCashflows handles POST /integrations/murex/cashflows - Ingest cashflows.
func (h *MurexHandler) HandleIngestCashflows(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Filters map[string]interface{} `json:"filters,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil && err.Error() != "EOF" {
		h.respondError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request: %v", err))
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 2*time.Minute)
	defer cancel()

	if err := h.integration.IngestCashflows(ctx, req.Filters); err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Cashflow ingestion failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, map[string]interface{}{
		"status":  "success",
		"message": "Cashflows ingested successfully",
	})
}

// HandleDiscoverSchema handles GET /integrations/murex/schema - Discover schema.
func (h *MurexHandler) HandleDiscoverSchema(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	schema, err := h.integration.DiscoverSchema(ctx)
	if err != nil {
		h.respondError(w, http.StatusInternalServerError, fmt.Sprintf("Schema discovery failed: %v", err))
		return
	}

	h.respondJSON(w, http.StatusOK, schema)
}

// respondJSON writes a JSON response.
func (h *MurexHandler) respondJSON(w http.ResponseWriter, status int, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(data); err != nil {
		if h.logger != nil {
			h.logger.Printf("Failed to encode JSON response: %v", err)
		}
	}
}

// respondError writes an error response.
func (h *MurexHandler) respondError(w http.ResponseWriter, status int, message string) {
	h.respondJSON(w, status, map[string]interface{}{
		"error":   true,
		"message": message,
	})
}

