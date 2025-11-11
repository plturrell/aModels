package testing

import (
	"net/http"
)

// handleExportSignavioBatch handles batch export to Signavio (stub implementation).
func (ts *TestService) handleExportSignavioBatch(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "not implemented", http.StatusNotImplemented)
}

// handleSignavioHealth handles Signavio health check (stub implementation).
func (ts *TestService) handleSignavioHealth(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "not implemented", http.StatusNotImplemented)
}

// handleExportToSignavio handles export to Signavio (stub implementation).
func (ts *TestService) handleExportToSignavio(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "not implemented", http.StatusNotImplemented)
}

// handleGetSignavioMetrics handles getting Signavio metrics (stub implementation).
func (ts *TestService) handleGetSignavioMetrics(w http.ResponseWriter, r *http.Request) {
	http.Error(w, "not implemented", http.StatusNotImplemented)
}

