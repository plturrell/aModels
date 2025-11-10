package server

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/search"
)

type SearchServer struct {
	model   *search.SearchModelWithLocalAI
	service *search.SearchService
}

func NewSearchServer(model *search.SearchModelWithLocalAI, service *search.SearchService) *SearchServer {
	return &SearchServer{model: model, service: service}
}

func NewSearchServerWithService(service *search.SearchService) *SearchServer {
	srv := &SearchServer{service: service}
	if service != nil {
		srv.model = service.Model()
	}
	return srv
}

type embedRequest struct {
	Text string `json:"text"`
}

type embedResponse struct {
	Embedding []float64 `json:"embedding"`
}

type rerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Scores []float64 `json:"scores"`
}

type searchRequest struct {
	Query string `json:"query"`
	TopK  int    `json:"top_k,omitempty"`
}

type searchResponse struct {
	Results []search.SearchResult `json:"results"`
}

type addDocumentRequest struct {
	ID      string `json:"id"`
	Content string `json:"content"`
}

type addDocumentsRequest struct {
	Documents map[string]string `json:"documents"`
}

type modelInfoResponse struct {
	Info map[string]interface{} `json:"info"`
}

// formatError creates a user-friendly error message from an error.
// For security, it sanitizes error messages to avoid exposing internal details.
func formatError(operation string, err error) string {
	if err == nil {
		return fmt.Sprintf("failed to %s", operation)
	}
	
	errMsg := err.Error()
	
	// Check for common error types and provide user-friendly messages
	if strings.Contains(errMsg, "context deadline exceeded") || strings.Contains(errMsg, "timeout") {
		return fmt.Sprintf("operation '%s' timed out. Please try again.", operation)
	}
	if strings.Contains(errMsg, "connection refused") || strings.Contains(errMsg, "no such host") {
		return fmt.Sprintf("unable to connect to required service for '%s'. Please check service availability.", operation)
	}
	if strings.Contains(errMsg, "not found") {
		return fmt.Sprintf("resource not found for '%s'", operation)
	}
	if strings.Contains(errMsg, "invalid") || strings.Contains(errMsg, "malformed") {
		return fmt.Sprintf("invalid input for '%s': %s", operation, sanitizeErrorMessage(errMsg))
	}
	
	// For other errors, provide a generic but informative message
	return fmt.Sprintf("failed to %s: %s", operation, sanitizeErrorMessage(errMsg))
}

// sanitizeErrorMessage removes potentially sensitive information from error messages.
func sanitizeErrorMessage(msg string) string {
	// Remove file paths
	msg = strings.ReplaceAll(msg, "/", "")
	// Remove common sensitive patterns (simplified - in production, use more sophisticated sanitization)
	sensitivePatterns := []string{
		"password", "secret", "key", "token", "credential",
		"/home/", "/tmp/", "localhost", "127.0.0.1",
	}
	
	lowerMsg := strings.ToLower(msg)
	for _, pattern := range sensitivePatterns {
		if strings.Contains(lowerMsg, pattern) {
			return "internal error (details hidden for security)"
		}
	}
	
	// Truncate very long error messages
	if len(msg) > 200 {
		return msg[:200] + "..."
	}
	
	return msg
}

func (s *SearchServer) HandleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *SearchServer) HandleEmbed(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req embedRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request body: %s", formatError("decode embed request", err)), http.StatusBadRequest)
		return
	}
	
	if req.Text == "" {
		http.Error(w, "text field is required", http.StatusBadRequest)
		return
	}

	model := s.model
	if model == nil && s.service != nil {
		model = s.service.Model()
	}
	if model == nil {
		http.Error(w, "model not configured", http.StatusServiceUnavailable)
		return
	}

	embedding, err := model.Embed(r.Context(), req.Text)
	if err != nil {
		log.Printf("embed error: %v", err)
		http.Error(w, formatError("generate embedding", err), http.StatusInternalServerError)
		return
	}

	resp := embedResponse{Embedding: embedding}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *SearchServer) HandleRerank(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req rerankRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request body: %s", formatError("decode rerank request", err)), http.StatusBadRequest)
		return
	}
	
	if req.Query == "" {
		http.Error(w, "query field is required", http.StatusBadRequest)
		return
	}
	if len(req.Documents) == 0 {
		http.Error(w, "documents array cannot be empty", http.StatusBadRequest)
		return
	}

	model := s.model
	if model == nil && s.service != nil {
		model = s.service.Model()
	}
	if model == nil {
		http.Error(w, "model not configured", http.StatusServiceUnavailable)
		return
	}

	scores, err := model.Rerank(r.Context(), req.Query, req.Documents)
	if err != nil {
		log.Printf("rerank error: %v", err)
		http.Error(w, formatError("rerank documents", err), http.StatusInternalServerError)
		return
	}

	resp := rerankResponse{Scores: scores}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *SearchServer) HandleSearch(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request body: %s", formatError("decode search request", err)), http.StatusBadRequest)
		return
	}
	
	if req.Query == "" {
		http.Error(w, "query field is required", http.StatusBadRequest)
		return
	}

	if req.TopK <= 0 {
		req.TopK = 10 // Default to 10 results
	}

	if s.service == nil {
		http.Error(w, "search service not configured", http.StatusServiceUnavailable)
		return
	}

	results, err := s.service.Search(r.Context(), req.Query, req.TopK)
	if err != nil {
		log.Printf("search error: %v", err)
		http.Error(w, formatError("perform search", err), http.StatusInternalServerError)
		return
	}

	resp := searchResponse{Results: results}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (s *SearchServer) HandleAddDocument(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req addDocumentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request body: %s", formatError("decode add document request", err)), http.StatusBadRequest)
		return
	}
	
	if req.ID == "" {
		http.Error(w, "id field is required", http.StatusBadRequest)
		return
	}
	if req.Content == "" {
		http.Error(w, "content field is required", http.StatusBadRequest)
		return
	}

	if s.service == nil {
		http.Error(w, "search service not configured", http.StatusServiceUnavailable)
		return
	}

	err := s.service.AddDocument(r.Context(), req.ID, req.Content)
	if err != nil {
		log.Printf("add document error: %v", err)
		http.Error(w, formatError("add document", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
}

func (s *SearchServer) HandleAddDocuments(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req addDocumentsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid request body: %s", formatError("decode add documents request", err)), http.StatusBadRequest)
		return
	}
	
	if len(req.Documents) == 0 {
		http.Error(w, "documents map cannot be empty", http.StatusBadRequest)
		return
	}

	if s.service == nil {
		http.Error(w, "search service not configured", http.StatusServiceUnavailable)
		return
	}

	err := s.service.AddDocuments(r.Context(), req.Documents)
	if err != nil {
		log.Printf("add documents error: %v", err)
		http.Error(w, formatError("add documents", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"status": "success"})
}

func (s *SearchServer) HandleModelInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	model := s.model
	if model == nil && s.service != nil {
		model = s.service.Model()
	}
	if model == nil {
		http.Error(w, "model not configured", http.StatusServiceUnavailable)
		return
	}

	info := model.GetModelInfo()
	if s.service != nil {
		if cat, updated := s.service.AgentCatalogSnapshot(); cat != nil {
			info["agent_catalog"] = cat.Suites
			info["agent_tools"] = cat.Tools
			if !updated.IsZero() {
				info["agent_catalog_updated_at"] = updated.Format(time.RFC3339)
			}
			enrichment := search.EnrichCatalog(cat)
			if enrichment.Summary != "" {
				info["agent_catalog_summary"] = enrichment.Summary
			}
			if enrichment.Prompt != "" {
				info["agent_catalog_context"] = enrichment.Prompt
			}
			if enrichment.Stats.SuiteCount > 0 || enrichment.Stats.UniqueToolCount > 0 {
				info["agent_catalog_stats"] = enrichment.Stats
			}
			if len(enrichment.Implementations) > 0 {
				info["agent_catalog_matrix"] = enrichment.Implementations
			}
			if len(enrichment.UniqueTools) > 0 {
				info["agent_catalog_unique_tools"] = enrichment.UniqueTools
			}
			if len(enrichment.StandaloneTools) > 0 {
				info["agent_catalog_tool_details"] = enrichment.StandaloneTools
			}
		}
	}
	resp := modelInfoResponse{Info: info}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func ServeStatic(dir string) http.HandlerFunc {
	fs := http.FileServer(http.Dir(dir))
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			http.StripPrefix("/", fs).ServeHTTP(w, r)
			return
		}
		fs.ServeHTTP(w, r)
	}
}
