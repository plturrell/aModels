package search

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strings"
	"time"
)

// RAGEndpoint provides RAG-specific search functionality
type RAGEndpoint struct {
	documentStore DocumentStore
	embeddingService EmbeddingService
	reranker      Reranker
}

// DocumentStore interface for document storage and retrieval
type DocumentStore interface {
	SearchDocuments(ctx context.Context, query string, topK int, filters map[string]string) ([]Document, error)
	GetDocumentByID(ctx context.Context, id string) (*Document, error)
	GetDocumentsByTaskType(ctx context.Context, taskType string, topK int) ([]Document, error)
}

// EmbeddingService interface for embedding operations
type EmbeddingService interface {
	GetEmbedding(ctx context.Context, text string) ([]float64, error)
	ComputeSimilarity(embedding1, embedding2 []float64) (float64, error)
}

// Reranker interface for document reranking
type Reranker interface {
	RerankDocuments(ctx context.Context, query string, documents []Document) ([]Document, error)
}

// Document represents a document in the search index
type Document struct {
	ID          string                 `json:"id"`
	Content     string                 `json:"content"`
	Title       string                 `json:"title,omitempty"`
	TaskType    string                 `json:"task_type"`
	Domain      string                 `json:"domain"`
	Embedding   []float64              `json:"embedding,omitempty"`
	Metadata    map[string]interface{} `json:"metadata"`
	Score       float64                `json:"score,omitempty"`
	Similarity  float64                `json:"similarity,omitempty"`
}

// RAGSearchRequest represents a request for RAG search
type RAGSearchRequest struct {
	Query           string            `json:"query"`
	TaskType        string            `json:"task_type"`
	TopK            int               `json:"top_k,omitempty"`
	SimilarityThreshold float64        `json:"similarity_threshold,omitempty"`
	IncludePassages bool              `json:"include_passages,omitempty"`
	Filters         map[string]string `json:"filters,omitempty"`
}

// RAGSearchResponse represents the response from RAG search
type RAGSearchResponse struct {
	Context     string     `json:"context"`
	Passages    []string   `json:"passages,omitempty"`
	Results     []Document `json:"results"`
	TaskType    string     `json:"task_type"`
	Domain      string     `json:"domain"`
	QueryTime   float64    `json:"query_time_ms"`
	TotalHits   int        `json:"total_hits"`
}

// NewRAGEndpoint creates a new RAG endpoint
func NewRAGEndpoint(documentStore DocumentStore, embeddingService EmbeddingService, reranker Reranker) *RAGEndpoint {
	return &RAGEndpoint{
		documentStore:   documentStore,
		embeddingService: embeddingService,
		reranker:       reranker,
	}
}

// HandleRAGSearch handles RAG-specific search requests
func (re *RAGEndpoint) HandleRAGSearch(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	
	var req RAGSearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request: %v", err), http.StatusBadRequest)
		return
	}
	
	// Set defaults
	if req.TopK == 0 {
		req.TopK = 5
	}
	if req.SimilarityThreshold == 0 {
		req.SimilarityThreshold = 0.7
	}
	
	ctx := r.Context()
	
	// Perform RAG search
	resp, err := re.performRAGSearch(ctx, &req)
	if err != nil {
		log.Printf("RAG search error: %v", err)
		http.Error(w, fmt.Sprintf("RAG search failed: %v", err), http.StatusInternalServerError)
		return
	}
	
	// Set query time
	resp.QueryTime = float64(time.Since(start).Nanoseconds()) / 1e6
	
	// Return response
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("Failed to encode response: %v", err)
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		return
	}
}

// performRAGSearch performs the actual RAG search
func (re *RAGEndpoint) performRAGSearch(ctx context.Context, req *RAGSearchRequest) (*RAGSearchResponse, error) {
	// Step 1: Get query embedding
	queryEmbedding, err := re.embeddingService.GetEmbedding(ctx, req.Query)
	if err != nil {
		return nil, fmt.Errorf("failed to get query embedding: %w", err)
	}
	
	// Step 2: Search for relevant documents
	documents, err := re.documentStore.SearchDocuments(ctx, req.Query, req.TopK*2, req.Filters) // Get more for reranking
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}
	
	// Step 3: Compute similarities and filter by threshold
	var filteredDocs []Document
	for _, doc := range documents {
		if len(doc.Embedding) > 0 {
			similarity, err := re.embeddingService.ComputeSimilarity(queryEmbedding, doc.Embedding)
			if err != nil {
				continue // Skip documents with embedding errors
			}
			
			doc.Similarity = similarity
			if similarity >= req.SimilarityThreshold {
				filteredDocs = append(filteredDocs, doc)
			}
		}
	}
	
	// Step 4: Rerank documents if reranker is available
	if re.reranker != nil {
		rerankedDocs, err := re.reranker.RerankDocuments(ctx, req.Query, filteredDocs)
		if err != nil {
			log.Printf("Reranking failed, using original order: %v", err)
		} else {
			filteredDocs = rerankedDocs
		}
	}
	
	// Step 5: Sort by similarity score
	sort.Slice(filteredDocs, func(i, j int) bool {
		return filteredDocs[i].Similarity > filteredDocs[j].Similarity
	})
	
	// Step 6: Limit to top-k
	if len(filteredDocs) > req.TopK {
		filteredDocs = filteredDocs[:req.TopK]
	}
	
	// Step 7: Build context and passages
	context := re.buildContext(filteredDocs)
	passages := re.extractPassages(filteredDocs, req.IncludePassages)
	
	// Determine domain from documents
	domain := re.determineDomain(filteredDocs)
	
	return &RAGSearchResponse{
		Context:   context,
		Passages:  passages,
		Results:   filteredDocs,
		TaskType:  req.TaskType,
		Domain:    domain,
		TotalHits: len(filteredDocs),
	}, nil
}

// buildContext builds a context string from documents
func (re *RAGEndpoint) buildContext(documents []Document) string {
	if len(documents) == 0 {
		return ""
	}
	
	var contextParts []string
	for i, doc := range documents {
		// Truncate long documents
		content := doc.Content
		if len(content) > 500 {
			content = content[:500] + "..."
		}
		
		contextPart := fmt.Sprintf("[%d] %s", i+1, content)
		contextParts = append(contextParts, contextPart)
	}
	
	return strings.Join(contextParts, "\n\n")
}

// extractPassages extracts passages from documents
func (re *RAGEndpoint) extractPassages(documents []Document, includePassages bool) []string {
	if !includePassages {
		return nil
	}
	
	var passages []string
	for _, doc := range documents {
		// Split document into passages (simplified)
		passage := doc.Content
		if len(passage) > 200 {
			passage = passage[:200] + "..."
		}
		passages = append(passages, passage)
	}
	
	return passages
}

// determineDomain determines the domain from documents
func (re *RAGEndpoint) determineDomain(documents []Document) string {
	if len(documents) == 0 {
		return "general"
	}
	
	// Count domain occurrences
	domainCounts := make(map[string]int)
	for _, doc := range documents {
		if doc.Domain != "" {
			domainCounts[doc.Domain]++
		}
	}
	
	// Find most common domain
	maxCount := 0
	mostCommonDomain := "general"
	for domain, count := range domainCounts {
		if count > maxCount {
			maxCount = count
			mostCommonDomain = domain
		}
	}
	
	return mostCommonDomain
}

// HandleTaskTypeExamples handles requests for task-specific examples
func (re *RAGEndpoint) HandleTaskTypeExamples(w http.ResponseWriter, r *http.Request) {
	taskType := r.URL.Query().Get("task_type")
	topK := 5
	
	if k := r.URL.Query().Get("top_k"); k != "" {
		if _, err := fmt.Sscanf(k, "%d", &topK); err != nil {
			http.Error(w, "Invalid top_k parameter", http.StatusBadRequest)
			return
		}
	}
	
	if taskType == "" {
		http.Error(w, "task_type parameter is required", http.StatusBadRequest)
		return
	}
	
	ctx := r.Context()
	documents, err := re.documentStore.GetDocumentsByTaskType(ctx, taskType, topK)
	if err != nil {
		log.Printf("Failed to get task type examples: %v", err)
		http.Error(w, "Failed to get examples", http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(documents); err != nil {
		log.Printf("Failed to encode examples: %v", err)
		http.Error(w, "Failed to encode examples", http.StatusInternalServerError)
		return
	}
}

// HandleSimilarQuestions handles requests for similar questions
func (re *RAGEndpoint) HandleSimilarQuestions(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("query")
	taskType := r.URL.Query().Get("task_type")
	topK := 5
	
	if k := r.URL.Query().Get("top_k"); k != "" {
		if _, err := fmt.Sscanf(k, "%d", &topK); err != nil {
			http.Error(w, "Invalid top_k parameter", http.StatusBadRequest)
			return
		}
	}
	
	if query == "" {
		http.Error(w, "query parameter is required", http.StatusBadRequest)
		return
	}
	
	ctx := r.Context()
	filters := map[string]string{
		"type": "question",
	}
	if taskType != "" {
		filters["task_type"] = taskType
	}
	
	documents, err := re.documentStore.SearchDocuments(ctx, query, topK, filters)
	if err != nil {
		log.Printf("Failed to get similar questions: %v", err)
		http.Error(w, "Failed to get similar questions", http.StatusInternalServerError)
		return
	}
	
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(documents); err != nil {
		log.Printf("Failed to encode similar questions: %v", err)
		http.Error(w, "Failed to encode similar questions", http.StatusInternalServerError)
		return
	}
}
