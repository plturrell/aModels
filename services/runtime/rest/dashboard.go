package rest

import (
	"encoding/json"
	"math/rand"
	"net/http"
	"sync"
	"time"

	"github.com/plturrell/aModels/services/framework/analytics"
	"github.com/plturrell/aModels/services/plot/dashboard"
)

// DashboardStore manages custom dashboards
type DashboardStore struct {
	dashboards map[string]*CustomDashboard
	versions   map[string][]*DashboardVersion
	shares     map[string]*DashboardShare
	mu         sync.RWMutex
}

// CustomDashboard represents a user-created dashboard
type CustomDashboard struct {
	ID          string                    `json:"id"`
	Name        string                    `json:"name"`
	Description string                    `json:"description"`
	TemplateID  string                    `json:"template_id"`
	Config      map[string]interface{}    `json:"config"`
	Charts      []ChartConfig             `json:"charts"`
	OwnerID     string                    `json:"owner_id"`
	CreatedAt   time.Time                 `json:"created_at"`
	UpdatedAt   time.Time                 `json:"updated_at"`
	Version     int                       `json:"version"`
	Tags        []string                  `json:"tags"`
	IsPublic    bool                      `json:"is_public"`
}

// DashboardVersion represents a versioned snapshot of a dashboard
type DashboardVersion struct {
	Version     int                    `json:"version"`
	Dashboard   *CustomDashboard       `json:"dashboard"`
	CreatedAt   time.Time              `json:"created_at"`
	CreatedBy   string                  `json:"created_by"`
	ChangeNotes string                  `json:"change_notes"`
}

// DashboardShare represents a shared dashboard
type DashboardShare struct {
	DashboardID string    `json:"dashboard_id"`
	SharedWith  []string  `json:"shared_with"` // User IDs or groups
	Permission  string    `json:"permission"`  // "read", "write", "admin"
	SharedAt    time.Time `json:"shared_at"`
	SharedBy    string    `json:"shared_by"`
}

// ChartConfig represents a chart configuration
type ChartConfig struct {
	ID          string                 `json:"id"`
	Type        string                 `json:"type"` // "bar", "line", "pie", etc.
	Title       string                 `json:"title"`
	DataSource  string                 `json:"data_source"`
	Config      map[string]interface{} `json:"config"`
	Position    Position               `json:"position"`
	Size        Size                  `json:"size"`
}

// Position represents chart position
type Position struct {
	X int `json:"x"`
	Y int `json:"y"`
}

// Size represents chart size
type Size struct {
	Width  int `json:"width"`
	Height int `json:"height"`
}

var globalDashboardStore *DashboardStore

func init() {
	globalDashboardStore = &DashboardStore{
		dashboards: make(map[string]*CustomDashboard),
		versions:   make(map[string][]*DashboardVersion),
		shares:     make(map[string]*DashboardShare),
	}
}

// CreateDashboardRequest represents a request to create a dashboard
type CreateDashboardRequest struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	TemplateID  string                 `json:"template_id"`
	Config      map[string]interface{} `json:"config"`
	Charts      []ChartConfig          `json:"charts"`
	Tags        []string                `json:"tags"`
	IsPublic   bool                    `json:"is_public"`
}

// UpdateDashboardRequest represents a request to update a dashboard
type UpdateDashboardRequest struct {
	Name        *string                `json:"name,omitempty"`
	Description *string                `json:"description,omitempty"`
	Config      map[string]interface{} `json:"config,omitempty"`
	Charts      []ChartConfig          `json:"charts,omitempty"`
	Tags        []string               `json:"tags,omitempty"`
	IsPublic    *bool                 `json:"is_public,omitempty"`
	ChangeNotes string                `json:"change_notes,omitempty"`
}

// ShareDashboardRequest represents a request to share a dashboard
type ShareDashboardRequest struct {
	SharedWith []string `json:"shared_with"`
	Permission string   `json:"permission"`
}

// CreateDashboard creates a new custom dashboard
func (h *Handler) CreateDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req CreateDashboardRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Generate dashboard ID
	dashboardID := generateDashboardID()
	
	// Get user ID from context (in real implementation)
	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}

	dashboard := &CustomDashboard{
		ID:          dashboardID,
		Name:        req.Name,
		Description: req.Description,
		TemplateID:  req.TemplateID,
		Config:      req.Config,
		Charts:      req.Charts,
		OwnerID:     userID,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Version:     1,
		Tags:        req.Tags,
		IsPublic:    req.IsPublic,
	}

	globalDashboardStore.mu.Lock()
	globalDashboardStore.dashboards[dashboardID] = dashboard
	globalDashboardStore.versions[dashboardID] = []*DashboardVersion{
		{
			Version:   1,
			Dashboard: dashboard,
			CreatedAt: time.Now(),
			CreatedBy: userID,
		},
	}
	globalDashboardStore.mu.Unlock()

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(dashboard)
}

// GetDashboard retrieves a dashboard by ID
func (h *Handler) GetDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	globalDashboardStore.mu.RLock()
	dashboard, exists := globalDashboardStore.dashboards[dashboardID]
	globalDashboardStore.mu.RUnlock()

	if !exists {
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(dashboard)
}

// ListDashboards lists all dashboards (with filtering)
func (h *Handler) ListDashboards(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	userID := r.Header.Get("X-User-ID")
	tag := r.URL.Query().Get("tag")
	publicOnly := r.URL.Query().Get("public") == "true"

	globalDashboardStore.mu.RLock()
	dashboards := make([]*CustomDashboard, 0)
	for _, dashboard := range globalDashboardStore.dashboards {
		// Filter by ownership or public
		if dashboard.OwnerID == userID || dashboard.IsPublic {
			if publicOnly && !dashboard.IsPublic {
				continue
			}
			if tag != "" {
				hasTag := false
				for _, t := range dashboard.Tags {
					if t == tag {
						hasTag = true
						break
					}
				}
				if !hasTag {
					continue
				}
			}
			dashboards = append(dashboards, dashboard)
		}
	}
	globalDashboardStore.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"dashboards": dashboards,
		"count":      len(dashboards),
	})
}

// UpdateDashboard updates an existing dashboard
func (h *Handler) UpdateDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPut {
		w.Header().Set("Allow", http.MethodPut)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	var req UpdateDashboardRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	userID := r.Header.Get("X-User-ID")

	globalDashboardStore.mu.Lock()
	dashboard, exists := globalDashboardStore.dashboards[dashboardID]
	if !exists {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	// Check ownership
	if dashboard.OwnerID != userID {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "unauthorized", http.StatusForbidden)
		return
	}

	// Update fields
	if req.Name != nil {
		dashboard.Name = *req.Name
	}
	if req.Description != nil {
		dashboard.Description = *req.Description
	}
	if req.Config != nil {
		dashboard.Config = req.Config
	}
	if req.Charts != nil {
		dashboard.Charts = req.Charts
	}
	if req.Tags != nil {
		dashboard.Tags = req.Tags
	}
	if req.IsPublic != nil {
		dashboard.IsPublic = *req.IsPublic
	}
	dashboard.UpdatedAt = time.Now()
	dashboard.Version++

	// Create version snapshot
	version := &DashboardVersion{
		Version:     dashboard.Version,
		Dashboard:   dashboard,
		CreatedAt:   time.Now(),
		CreatedBy:   userID,
		ChangeNotes: req.ChangeNotes,
	}
	globalDashboardStore.versions[dashboardID] = append(
		globalDashboardStore.versions[dashboardID],
		version,
	)

	globalDashboardStore.mu.Unlock()

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(dashboard)
}

// DeleteDashboard deletes a dashboard
func (h *Handler) DeleteDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodDelete {
		w.Header().Set("Allow", http.MethodDelete)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	userID := r.Header.Get("X-User-ID")

	globalDashboardStore.mu.Lock()
	dashboard, exists := globalDashboardStore.dashboards[dashboardID]
	if !exists {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	if dashboard.OwnerID != userID {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "unauthorized", http.StatusForbidden)
		return
	}

	delete(globalDashboardStore.dashboards, dashboardID)
	delete(globalDashboardStore.versions, dashboardID)
	delete(globalDashboardStore.shares, dashboardID)
	globalDashboardStore.mu.Unlock()

	w.WriteHeader(http.StatusNoContent)
}

// ShareDashboard shares a dashboard with other users
func (h *Handler) ShareDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	var req ShareDashboardRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	userID := r.Header.Get("X-User-ID")

	globalDashboardStore.mu.Lock()
	dashboard, exists := globalDashboardStore.dashboards[dashboardID]
	if !exists {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	if dashboard.OwnerID != userID {
		globalDashboardStore.mu.Unlock()
		http.Error(w, "unauthorized", http.StatusForbidden)
		return
	}

	share := &DashboardShare{
		DashboardID: dashboardID,
		SharedWith:  req.SharedWith,
		Permission:  req.Permission,
		SharedAt:    time.Now(),
		SharedBy:    userID,
	}
	globalDashboardStore.shares[dashboardID] = share
	globalDashboardStore.mu.Unlock()

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(share)
}

// GetDashboardVersions retrieves version history for a dashboard
func (h *Handler) GetDashboardVersions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	globalDashboardStore.mu.RLock()
	versions, exists := globalDashboardStore.versions[dashboardID]
	globalDashboardStore.mu.RUnlock()

	if !exists {
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"versions": versions,
		"count":    len(versions),
	})
}

// ExportDashboard exports a dashboard as JSON
func (h *Handler) ExportDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	dashboardID := r.URL.Query().Get("id")
	if dashboardID == "" {
		http.Error(w, "dashboard ID required", http.StatusBadRequest)
		return
	}

	globalDashboardStore.mu.RLock()
	dashboard, exists := globalDashboardStore.dashboards[dashboardID]
	globalDashboardStore.mu.RUnlock()

	if !exists {
		http.Error(w, "dashboard not found", http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.Header().Set("Content-Disposition", "attachment; filename=dashboard_"+dashboardID+".json")
	json.NewEncoder(w).Encode(dashboard)
}

// ImportDashboard imports a dashboard from JSON
func (h *Handler) ImportDashboard(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", http.MethodPost)
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var dashboard CustomDashboard
	if err := json.NewDecoder(r.Body).Decode(&dashboard); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	userID := r.Header.Get("X-User-ID")
	if userID == "" {
		userID = "anonymous"
	}

	// Generate new ID and reset metadata
	dashboard.ID = generateDashboardID()
	dashboard.OwnerID = userID
	dashboard.CreatedAt = time.Now()
	dashboard.UpdatedAt = time.Now()
	dashboard.Version = 1

	globalDashboardStore.mu.Lock()
	globalDashboardStore.dashboards[dashboard.ID] = &dashboard
	globalDashboardStore.versions[dashboard.ID] = []*DashboardVersion{
		{
			Version:   1,
			Dashboard: &dashboard,
			CreatedAt: time.Now(),
			CreatedBy: userID,
		},
	}
	globalDashboardStore.mu.Unlock()

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(dashboard)
}

// Helper function to generate dashboard ID
func generateDashboardID() string {
	return "dashboard_" + time.Now().Format("20060102150405") + "_" + randomString(8)
}

func randomString(length int) string {
	const charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	rand.Seed(time.Now().UnixNano())
	b := make([]byte, length)
	for i := range b {
		b[i] = charset[rand.Intn(len(charset))]
	}
	return string(b)
}

