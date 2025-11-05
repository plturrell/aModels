package domain

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// DomainLifecycleAPI provides HTTP API for domain lifecycle management
type DomainLifecycleAPI struct {
	lifecycleManager *LifecycleManager
}

// NewDomainLifecycleAPI creates a new domain lifecycle API
func NewDomainLifecycleAPI(lm *LifecycleManager) *DomainLifecycleAPI {
	return &DomainLifecycleAPI{
		lifecycleManager: lm,
	}
}

// HandleCreateDomain handles POST /v1/domains/create
func (api *DomainLifecycleAPI) HandleCreateDomain(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		DomainID string                 `json:"domain_id"`
		Config   *DomainConfig          `json:"config"`
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := api.lifecycleManager.CreateDomain(context.Background(), req.DomainID, req.Config, req.Metadata); err != nil {
		http.Error(w, fmt.Sprintf("failed to create domain: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"domain_id": req.DomainID,
		"message":   "Domain created successfully",
	})
}

// HandleUpdateDomain handles PUT /v1/domains/{domain_id}
func (api *DomainLifecycleAPI) HandleUpdateDomain(w http.ResponseWriter, r *http.Request, domainID string) {
	if r.Method != http.MethodPut {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Config   *DomainConfig          `json:"config"`
		Metadata map[string]interface{} `json:"metadata,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := api.lifecycleManager.UpdateDomain(context.Background(), domainID, req.Config, req.Metadata); err != nil {
		http.Error(w, fmt.Sprintf("failed to update domain: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"domain_id": domainID,
		"message":   "Domain updated successfully",
	})
}

// HandleArchiveDomain handles POST /v1/domains/{domain_id}/archive
func (api *DomainLifecycleAPI) HandleArchiveDomain(w http.ResponseWriter, r *http.Request, domainID string) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Reason string `json:"reason"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if err := api.lifecycleManager.ArchiveDomain(context.Background(), domainID, req.Reason); err != nil {
		http.Error(w, fmt.Sprintf("failed to archive domain: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"domain_id": domainID,
		"message":   "Domain archived successfully",
	})
}

// HandleDeleteDomain handles DELETE /v1/domains/{domain_id}
func (api *DomainLifecycleAPI) HandleDeleteDomain(w http.ResponseWriter, r *http.Request, domainID string) {
	if r.Method != http.MethodDelete {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	force := r.URL.Query().Get("force") == "true"

	if err := api.lifecycleManager.DeleteDomain(context.Background(), domainID, force); err != nil {
		http.Error(w, fmt.Sprintf("failed to delete domain: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":    "success",
		"domain_id": domainID,
		"message":   "Domain deleted successfully",
	})
}

// HandleListDomains handles GET /v1/domains/list
func (api *DomainLifecycleAPI) HandleListDomains(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	statuses, err := api.lifecycleManager.ListDomains(context.Background())
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to list domains: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "success",
		"domains": statuses,
	})
}
