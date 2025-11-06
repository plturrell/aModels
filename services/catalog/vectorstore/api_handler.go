package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
)

// HANAVectorStoreHandler handles HTTP API requests for HANA vector store
type HANAVectorStoreHandler struct {
	store            *HANACloudVectorStore
	embeddingService *EmbeddingService
	logger           *log.Logger
}

// NewHANAVectorStoreHandler creates a new API handler
func NewHANAVectorStoreHandler(
	store *HANACloudVectorStore,
	embeddingService *EmbeddingService,
	logger *log.Logger,
) *HANAVectorStoreHandler {
	return &HANAVectorStoreHandler{
		store:            store,
		embeddingService: embeddingService,
		logger:           logger,
	}
}

// HandleStoreInformation handles POST /vectorstore/store
func (h *HANAVectorStoreHandler) HandleStoreInformation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	var req struct {
		Type        string                 `json:"type"`
		System      string                 `json:"system"`
		Category    string                 `json:"category"`
		Title       string                 `json:"title"`
		Content     string                 `json:"content"`
		Metadata    map[string]interface{} `json:"metadata"`
		Tags        []string               `json:"tags"`
		IsPublic    bool                   `json:"is_public"`
		GenerateEmbedding bool             `json:"generate_embedding"` // Auto-generate embedding
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Generate embedding if requested
	var vector []float32
	var err error
	if req.GenerateEmbedding && h.embeddingService != nil {
		vector, err = h.embeddingService.GenerateEmbedding(ctx, req.Content)
		if err != nil {
			h.logger.Printf("Warning: Failed to generate embedding: %v", err)
		}
	}

	info := &PublicInformation{
		Type:      req.Type,
		System:    req.System,
		Category:  req.Category,
		Title:     req.Title,
		Content:   req.Content,
		Vector:    vector,
		Metadata:  req.Metadata,
		Tags:      req.Tags,
		IsPublic:  req.IsPublic,
	}

	if err := h.store.StorePublicInformation(ctx, info); err != nil {
		h.logger.Printf("Failed to store information: %v", err)
		http.Error(w, fmt.Sprintf("Failed to store information: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"id":       info.ID,
		"message":  "Information stored successfully",
	})
}

// HandleSearchInformation handles POST /vectorstore/search
func (h *HANAVectorStoreHandler) HandleSearchInformation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	var req struct {
		Query      string                 `json:"query"`       // Text query (will generate embedding)
		Vector     []float32              `json:"vector"`      // Direct vector (optional)
		Type       string                 `json:"type"`
		System     string                 `json:"system"`
		Category   string                 `json:"category"`
		Tags       []string               `json:"tags"`
		IsPublic   *bool                  `json:"is_public"`
		Limit      int                    `json:"limit"`
		Threshold  float64                `json:"threshold"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Failed to decode request: %v", err), http.StatusBadRequest)
		return
	}

	// Generate embedding from query if vector not provided
	var queryVector []float32
	var err error
	if len(req.Vector) > 0 {
		queryVector = req.Vector
	} else if req.Query != "" && h.embeddingService != nil {
		queryVector, err = h.embeddingService.GenerateEmbedding(ctx, req.Query)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to generate embedding: %v", err), http.StatusInternalServerError)
			return
		}
	} else {
		http.Error(w, "Either 'query' or 'vector' must be provided", http.StatusBadRequest)
		return
	}

	options := &SearchOptions{
		Type:      req.Type,
		System:    req.System,
		Category:  req.Category,
		Tags:      req.Tags,
		IsPublic:  req.IsPublic,
		Limit:     req.Limit,
		Threshold: req.Threshold,
	}

	if options.Limit <= 0 {
		options.Limit = 10
	}
	if options.Threshold <= 0 {
		options.Threshold = 0.7
	}

	results, err := h.store.SearchPublicInformation(ctx, queryVector, options)
	if err != nil {
		h.logger.Printf("Search failed: %v", err)
		http.Error(w, fmt.Sprintf("Search failed: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"results": results,
		"count":   len(results),
	})
}

// HandleGetInformation handles GET /vectorstore/{id}
func (h *HANAVectorStoreHandler) HandleGetInformation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Extract ID from path
	path := r.URL.Path
	id := strings.TrimPrefix(path, "/vectorstore/")
	if id == "" {
		http.Error(w, "ID is required", http.StatusBadRequest)
		return
	}

	info, err := h.store.GetPublicInformation(ctx, id)
	if err != nil {
		http.Error(w, fmt.Sprintf("Information not found: %v", err), http.StatusNotFound)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// HandleListPublicInformation handles GET /vectorstore with query parameters
func (h *HANAVectorStoreHandler) HandleListPublicInformation(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	ctx := r.Context()

	// Parse query parameters
	options := &ListOptions{}
	
	if typeParam := r.URL.Query().Get("type"); typeParam != "" {
		options.Type = typeParam
	}
	
	if systemParam := r.URL.Query().Get("system"); systemParam != "" {
		options.System = systemParam
	}
	
	if categoryParam := r.URL.Query().Get("category"); categoryParam != "" {
		options.Category = categoryParam
	}
	
	if tagsParam := r.URL.Query().Get("tags"); tagsParam != "" {
		options.Tags = strings.Split(tagsParam, ",")
	}
	
	if publicParam := r.URL.Query().Get("is_public"); publicParam != "" {
		isPublic := publicParam == "true" || publicParam == "1"
		options.IsPublic = &isPublic
	}
	
	if limitParam := r.URL.Query().Get("limit"); limitParam != "" {
		if limit, err := strconv.Atoi(limitParam); err == nil && limit > 0 {
			options.Limit = limit
		}
	}
	if options.Limit <= 0 {
		options.Limit = 100 // Default limit
	}
	
	if offsetParam := r.URL.Query().Get("offset"); offsetParam != "" {
		if offset, err := strconv.Atoi(offsetParam); err == nil && offset >= 0 {
			options.Offset = offset
		}
	}
	
	if orderByParam := r.URL.Query().Get("order_by"); orderByParam != "" {
		options.OrderBy = orderByParam
	}
	
	if orderDescParam := r.URL.Query().Get("order_desc"); orderDescParam != "" {
		options.OrderDesc = orderDescParam == "true" || orderDescParam == "1"
	}

	// List public information
	results, err := h.store.ListPublicInformation(ctx, options)
	if err != nil {
		h.logger.Printf("List failed: %v", err)
		http.Error(w, fmt.Sprintf("Failed to list information: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"results": results,
		"count":   len(results),
		"limit":   options.Limit,
		"offset":  options.Offset,
	})
}

