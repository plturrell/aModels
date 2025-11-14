package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"time"

	"github.com/plturrell/aModels/services/catalog/discoverability"
)

// DiscoverabilityHandler provides HTTP handlers for discoverability features.
type DiscoverabilityHandler struct {
	discoverSystem *discoverability.DiscoverabilitySystem
	logger         *log.Logger
}

// NewDiscoverabilityHandler creates a new discoverability handler.
func NewDiscoverabilityHandler(discoverSystem *discoverability.DiscoverabilitySystem, logger *log.Logger) *DiscoverabilityHandler {
	return &DiscoverabilityHandler{
		discoverSystem: discoverSystem,
		logger:         logger,
	}
}

// HandleSearch handles GET /api/discover/search.
func (h *DiscoverabilityHandler) HandleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	query := r.URL.Query().Get("q")
	teams := r.URL.Query()["team"]
	categories := r.URL.Query()["category"]
	tags := r.URL.Query()["tag"]
	sortBy := r.URL.Query().Get("sort_by")
	if sortBy == "" {
		sortBy = "relevance"
	}

	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit == 0 {
		limit = 20
	}
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	searchReq := discoverability.SearchRequest{
		Query:      query,
		Teams:      teams,
		Categories: categories,
		Tags:       tags,
		Limit:      limit,
		Offset:     offset,
		SortBy:     sortBy,
	}

	// Perform search
	results, err := h.discoverSystem.GetCrossTeamSearch().Search(r.Context(), searchReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"results":     results.Results,
		"total_count": results.TotalCount,
		"query":       results.Query,
		"duration_ms": results.Duration.Milliseconds(),
	})
}

// HandleMarketplace handles GET /api/discover/marketplace.
func (h *DiscoverabilityHandler) HandleMarketplace(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	category := r.URL.Query().Get("category")
	team := r.URL.Query().Get("team")
	sortBy := r.URL.Query().Get("sort_by")
	if sortBy == "" {
		sortBy = "recent"
	}

	limit, _ := strconv.Atoi(r.URL.Query().Get("limit"))
	if limit == 0 {
		limit = 20
	}
	offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

	filters := discoverability.MarketplaceFilters{
		Category: category,
		Team:     team,
		SortBy:   sortBy,
		Limit:    limit,
		Offset:   offset,
	}

	// List products
	listings, err := h.discoverSystem.GetMarketplace().ListProducts(r.Context(), filters)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to list products: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"listings": listings,
		"count":    len(listings),
	})
}

// HandleCreateTag handles POST /api/discover/tags.
func (h *DiscoverabilityHandler) HandleCreateTag(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		Name        string `json:"name"`
		Category    string `json:"category"`
		Description string `json:"description"`
		ParentTagID string `json:"parent_tag_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.Name == "" {
		http.Error(w, "name is required", http.StatusBadRequest)
		return
	}

	tag := &discoverability.Tag{
		ID:          fmt.Sprintf("tag-%d", time.Now().UnixNano()),
		Name:        req.Name,
		Category:    req.Category,
		ParentTagID: req.ParentTagID,
		Description: req.Description,
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
	}

	// Create tag
	err := h.discoverSystem.GetTagManager().CreateTag(r.Context(), tag)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to create tag: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"tag":     tag,
		"message": "Tag created successfully",
	})
}

// HandleRequestAccess handles POST /api/discover/access-request.
func (h *DiscoverabilityHandler) HandleRequestAccess(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req struct {
		ProductID     string `json:"product_id"`
		RequesterID   string `json:"requester_id"`
		RequesterTeam string `json:"requester_team"`
		Reason        string `json:"reason"`
		Comments      string `json:"comments"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}

	if req.ProductID == "" || req.RequesterID == "" {
		http.Error(w, "product_id and requester_id are required", http.StatusBadRequest)
		return
	}

	accessReq := discoverability.AccessRequest{
		ProductID:     req.ProductID,
		RequesterID:   req.RequesterID,
		RequesterTeam: req.RequesterTeam,
		Reason:        req.Reason,
		Comments:      req.Comments,
	}

	// Request access
	err := h.discoverSystem.GetMarketplace().RequestAccess(r.Context(), accessReq)
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to request access: %v", err), http.StatusInternalServerError)
		return
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"request_id": accessReq.ID,
		"status":     "pending",
		"message":    "Access request submitted successfully",
	})
}
