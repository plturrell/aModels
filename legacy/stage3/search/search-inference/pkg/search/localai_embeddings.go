package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Search/search-inference/pkg/storage"
)

// LocalAIEmbedder provides embedding generation using LocalAI
type LocalAIEmbedder struct {
	baseURL    string
	httpClient *http.Client
	apiKey     string
}

// EmbeddingRequest represents a request to LocalAI for embeddings
type EmbeddingRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

// EmbeddingResponse represents the response from LocalAI
type EmbeddingResponse struct {
	Object string `json:"object"`
	Data   []struct {
		Object    string    `json:"object"`
		Embedding []float64 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
	Model string `json:"model"`
	Usage struct {
		PromptTokens int `json:"prompt_tokens"`
		TotalTokens  int `json:"total_tokens"`
	} `json:"usage"`
}

// NewLocalAIEmbedder creates a new LocalAI embedder
func NewLocalAIEmbedder(baseURL, apiKey string) *LocalAIEmbedder {
	return &LocalAIEmbedder{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
		apiKey: apiKey,
	}
}

// Embed generates embeddings for a single text
func (e *LocalAIEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	embeddings, err := e.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}

	return embeddings[0], nil
}

// EmbedBatch generates embeddings for multiple texts
func (e *LocalAIEmbedder) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	req := EmbeddingRequest{
		Model: "0x3579-VectorProcessingAgent", // Use vector processing agent
		Input: texts,
	}

	jsonData, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", e.baseURL+"/v1/embeddings", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+e.apiKey)
	}

	resp, err := e.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API error (status %d)", resp.StatusCode)
	}

	var embeddingResp EmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embeddingResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert response to [][]float64
	embeddings := make([][]float64, len(embeddingResp.Data))
	for i, data := range embeddingResp.Data {
		embeddings[i] = data.Embedding
	}

	return embeddings, nil
}

// GetEmbeddingDimension returns the dimension of embeddings
func (e *LocalAIEmbedder) GetEmbeddingDimension() int {
	// VaultGemma typically produces 768-dimensional embeddings
	return 768
}

// GetModelInfo returns information about the embedding model
func (e *LocalAIEmbedder) GetModelInfo() map[string]interface{} {
	return map[string]interface{}{
		"model_name":     "0x3579-VectorProcessingAgent",
		"model_type":     "embedding",
		"dimension":      e.GetEmbeddingDimension(),
		"max_tokens":     512,
		"base_url":       e.baseURL,
		"supports_batch": true,
	}
}

// Close closes the embedder (no-op for HTTP client)
func (e *LocalAIEmbedder) Close() error {
	return nil
}

// SearchModelWithLocalAI wraps the original SearchModel with LocalAI embedding capabilities
type SearchModelWithLocalAI struct {
	*SearchModel
	embedder *LocalAIEmbedder
}

// NewSearchModelWithLocalAI creates a new search model with LocalAI integration
func NewSearchModelWithLocalAI(modelPath, localAIURL, apiKey string) (*SearchModelWithLocalAI, error) {
	searchModel, err := LoadSearchModel(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load search model: %w", err)
	}

	var embedder *LocalAIEmbedder
	if localAIURL != "" {
		embedder = NewLocalAIEmbedder(localAIURL, apiKey)
	}

	return &SearchModelWithLocalAI{
		SearchModel: searchModel,
		embedder:    embedder,
	}, nil
}

// Embed generates embeddings using LocalAI
func (s *SearchModelWithLocalAI) Embed(ctx context.Context, text string) ([]float64, error) {
	if s.embedder == nil {
		return s.SearchModel.Embed(ctx, text)
	}
	return s.embedder.Embed(ctx, text)
}

// EmbedBatch generates embeddings for multiple texts using LocalAI
func (s *SearchModelWithLocalAI) EmbedBatch(ctx context.Context, texts []string) ([][]float64, error) {
	if s.embedder == nil {
		// Fallback: generate embeddings one by one using base model
		embeddings := make([][]float64, len(texts))
		for i, text := range texts {
			emb, err := s.SearchModel.Embed(ctx, text)
			if err != nil {
				return nil, err
			}
			embeddings[i] = emb
		}
		return embeddings, nil
	}
	return s.embedder.EmbedBatch(ctx, texts)
}

// Rerank reranks documents using the base model
func (s *SearchModelWithLocalAI) Rerank(ctx context.Context, query string, documents []string) ([]float64, error) {
	if s.embedder == nil {
		return s.SearchModel.Rerank(ctx, query, documents)
	}

	// Generate embeddings for query and documents
	queryEmbedding, err := s.embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}

	documentEmbeddings, err := s.embedder.EmbedBatch(ctx, documents)
	if err != nil {
		return nil, fmt.Errorf("failed to embed documents: %w", err)
	}

	// Calculate cosine similarity scores
	scores := make([]float64, len(documents))
	for i, docEmbedding := range documentEmbeddings {
		scores[i] = cosineSimilarity(queryEmbedding, docEmbedding)
	}

	return scores, nil
}

// GetEmbeddingDimension returns the dimension of embeddings
func (s *SearchModelWithLocalAI) GetEmbeddingDimension() int {
	if s.embedder == nil {
		return 0
	}
	return s.embedder.GetEmbeddingDimension()
}

// GetModelInfo returns information about the model
func (s *SearchModelWithLocalAI) GetModelInfo() map[string]interface{} {
	info := map[string]interface{}{
		"base_model": "VaultGemma",
	}
	if s.embedder != nil {
		for k, v := range s.embedder.GetModelInfo() {
			info[k] = v
		}
	}
	info["base_model"] = "VaultGemma"
	info["rerank_capable"] = true
	return info
}

// Close closes the model and embedder
func (s *SearchModelWithLocalAI) Close() error {
	if err := s.SearchModel.Close(); err != nil {
		return err
	}
	if s.embedder != nil {
		return s.embedder.Close()
	}
	return nil
}

// cosineSimilarity calculates cosine similarity between two vectors
func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// SearchService orchestrates embeddings, Elasticsearch search, and HANA persistence.
type SearchService struct {
	model         *SearchModelWithLocalAI
	elastic       *elasticsearchClient
	cache         *vectorCache
	hanaDocuments *storage.HANADocumentStore
	hanaLogger    *storage.HANASearchLogger
	privacyConfig *storage.PrivacyConfig
	ownsModel     bool
	useInMemory   bool
	memMu         sync.RWMutex
	memDocs       map[string]*storedDocument
	agentCatalog  catalogState
}

type storedDocument struct {
	Content   string
	Metadata  map[string]interface{}
	Embedding []float64
	CreatedAt time.Time
	UpdatedAt time.Time
}

// SearchServiceConfig configures SearchService creation.
type SearchServiceConfig struct {
	ModelPath     string
	LocalAIURL    string
	LocalAIKey    string
	Elasticsearch ElasticsearchConfig
	Redis         *RedisConfig
	HANADSN       string
	PrivacyLevel  storage.PrivacyLevel
	ExistingModel *SearchModelWithLocalAI
}

// NewSearchServiceWithConfig wires a SearchService using the provided configuration.
func NewSearchServiceWithConfig(cfg SearchServiceConfig) (*SearchService, error) {
	var (
		model     *SearchModelWithLocalAI
		err       error
		ownsModel bool
	)

	if cfg.ExistingModel != nil {
		model = cfg.ExistingModel
	} else {
		model, err = NewSearchModelWithLocalAI(cfg.ModelPath, cfg.LocalAIURL, cfg.LocalAIKey)
		if err != nil {
			return nil, err
		}
		ownsModel = true
	}

	privacyCfg := storage.DefaultPrivacyConfig()
	if cfg.PrivacyLevel != "" {
		privacyCfg = storage.NewPrivacyConfig(cfg.PrivacyLevel)
	}

	esClient, err := newElasticsearchClient(cfg.Elasticsearch)
	if err != nil {
		if ownsModel {
			_ = model.Close()
		}
		return nil, err
	}
	useInMemory := esClient == nil
	var memDocs map[string]*storedDocument
	if useInMemory {
		memDocs = make(map[string]*storedDocument)
	}

	var cache *vectorCache
	if cfg.Redis != nil {
		cache, err = newVectorCache(*cfg.Redis, 12*time.Hour)
		if err != nil {
			if ownsModel {
				_ = model.Close()
			}
			return nil, fmt.Errorf("init redis cache: %w", err)
		}
	}

	var hanaDocs *storage.HANADocumentStore
	var hanaLogger *storage.HANASearchLogger
	if cfg.HANADSN != "" {
		hanaDocs, err = storage.NewHANADocumentStore(cfg.HANADSN, privacyCfg)
		if err != nil {
			if ownsModel {
				_ = model.Close()
			}
			if cache != nil {
				_ = cache.Close()
			}
			return nil, fmt.Errorf("init HANA document store: %w", err)
		}

		hanaLogger, err = storage.NewHANASearchLogger(cfg.HANADSN, privacyCfg)
		if err != nil {
			if ownsModel {
				_ = model.Close()
			}
			if cache != nil {
				_ = cache.Close()
			}
			_ = hanaDocs.Close()
			return nil, fmt.Errorf("init HANA logger: %w", err)
		}
	}

	return &SearchService{
		model:         model,
		elastic:       esClient,
		cache:         cache,
		hanaDocuments: hanaDocs,
		hanaLogger:    hanaLogger,
		privacyConfig: privacyCfg,
		ownsModel:     ownsModel,
		useInMemory:   useInMemory,
		memDocs:       memDocs,
	}, nil
}

// NewSearchService preserves the previous constructor signature for dev usage.
func NewSearchService(modelPath, localAIURL, apiKey string) (*SearchService, error) {
	return NewSearchServiceWithConfig(SearchServiceConfig{
		ModelPath:  modelPath,
		LocalAIURL: localAIURL,
		LocalAIKey: apiKey,
	})
}

// NewSearchServiceWithHANA constructs a SearchService using HANA for persistence.
func NewSearchServiceWithHANA(modelPath, localAIURL, apiKey, hanaDSN string, privacyLevel storage.PrivacyLevel) (*SearchService, error) {
	return NewSearchServiceWithConfig(SearchServiceConfig{
		ModelPath:    modelPath,
		LocalAIURL:   localAIURL,
		LocalAIKey:   apiKey,
		HANADSN:      hanaDSN,
		PrivacyLevel: privacyLevel,
	})
}

// AddDocument inserts or updates a document across HANA and Elasticsearch.
func (s *SearchService) AddDocument(ctx context.Context, id, content string) error {
	if id == "" {
		return fmt.Errorf("document id is required")
	}

	embedding, err := s.embedWithCache(ctx, content)
	if err != nil {
		return fmt.Errorf("embed document: %w", err)
	}

	metadata := map[string]interface{}{
		"added_at": time.Now().Format(time.RFC3339),
	}

	return s.storeAndIndex(ctx, id, content, metadata, embedding)
}

// AddDocuments ingests a batch of documents.
func (s *SearchService) AddDocuments(ctx context.Context, documents map[string]string) error {
	if len(documents) == 0 {
		return nil
	}

	ids := make([]string, 0, len(documents))
	contents := make([]string, 0, len(documents))
	for id, content := range documents {
		ids = append(ids, id)
		contents = append(contents, content)
	}

	embeddings, err := s.model.EmbedBatch(ctx, contents)
	if err != nil {
		return fmt.Errorf("embed batch: %w", err)
	}

	for i, id := range ids {
		if s.cache != nil {
			s.cache.Set(ctx, contents[i], embeddings[i])
		}
		if err := s.storeAndIndex(ctx, id, contents[i], map[string]interface{}{
			"added_at": time.Now().Format(time.RFC3339),
		}, embeddings[i]); err != nil {
			return err
		}
	}

	return nil
}

// SearchDocuments performs a vector search with optional filters and returns detailed documents.
func (s *SearchService) SearchDocuments(ctx context.Context, query string, topK int, filters map[string]string) ([]Document, error) {
	embedding, err := s.embedWithCache(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}

	if s.useInMemory {
		return s.searchInMemory(embedding, topK, filters), nil
	}

	if s.elastic == nil {
		return nil, fmt.Errorf("search index not configured")
	}

	hits, err := s.elastic.searchSimilarDocuments(ctx, embedding, topK, filters)
	if err != nil {
		return nil, err
	}

	results := make([]Document, 0, len(hits))
	for _, hit := range hits {
		doc := Document{
			ID:         hit.ID,
			Score:      hit.Score,
			Similarity: hit.Score,
		}

		if val, ok := hit.Source["content"].(string); ok {
			doc.Content = val
		}
		if title, ok := hit.Source["title"].(string); ok {
			doc.Title = title
		}
		if metadataRaw, ok := hit.Source["metadata"].(map[string]interface{}); ok {
			doc.Metadata = metadataRaw
			if taskType, ok := metadataRaw["task_type"].(string); ok {
				doc.TaskType = taskType
			}
			if domain, ok := metadataRaw["domain"].(string); ok {
				doc.Domain = domain
			}
		}

		// Fall back to HANA for canonical content if available.
		if s.hanaDocuments != nil {
			hanaDoc, err := s.hanaDocuments.GetDocument(ctx, hit.ID, "system", "search")
			if err == nil && hanaDoc != nil {
				doc.Content = hanaDoc.Content
				if doc.Metadata == nil {
					doc.Metadata = hanaDoc.Metadata
				}
			}
		}

		results = append(results, doc)
	}

	return results, nil
}

func (s *SearchService) searchInMemory(queryEmbedding []float64, topK int, filters map[string]string) []Document {
	s.memMu.RLock()
	defer s.memMu.RUnlock()

	results := make([]Document, 0, len(s.memDocs))
	for id, stored := range s.memDocs {
		if !metadataMatchesFilters(stored.Metadata, filters) {
			continue
		}

		similarity := cosineSimilarity(queryEmbedding, stored.Embedding)
		metadata := cloneMetadata(stored.Metadata)
		doc := Document{
			ID:         id,
			Content:    stored.Content,
			Metadata:   metadata,
			Similarity: similarity,
			Score:      similarity,
		}

		if metadata != nil {
			if taskType := stringFromMetadata(metadata, "task_type"); taskType != "" {
				doc.TaskType = taskType
			}
			if domain := stringFromMetadata(metadata, "domain"); domain != "" {
				doc.Domain = domain
			}
		}

		results = append(results, doc)
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Similarity > results[j].Similarity
	})

	if topK > 0 && topK < len(results) {
		results = results[:topK]
	}

	return results
}

// Search executes a vector search and adapts results into SearchResult structures.
func (s *SearchService) Search(ctx context.Context, query string, topK int) ([]SearchResult, error) {
	start := time.Now()
	docs, err := s.SearchDocuments(ctx, query, topK, nil)
	if err != nil {
		return nil, err
	}

	results := make([]SearchResult, len(docs))
	for i, doc := range docs {
		results[i] = SearchResult{
			ID:         doc.ID,
			Content:    doc.Content,
			Similarity: doc.Similarity,
		}
	}

	s.logSearchAsync(query, len(results), results, start)
	return results, nil
}

// SearchResult represents a search result
type SearchResult struct {
	ID         string  `json:"id"`
	Content    string  `json:"content"`
	Similarity float64 `json:"similarity"`
}

// GetDocument retrieves the raw document content by ID using a background context.
func (s *SearchService) GetDocument(id string) (string, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	doc, err := s.GetDocumentByID(ctx, id)
	if err != nil || doc == nil {
		return "", false
	}
	return doc.Content, true
}

// GetDocumentByID fetches detailed document metadata.
func (s *SearchService) GetDocumentByID(ctx context.Context, id string) (*Document, error) {
	if id == "" {
		return nil, fmt.Errorf("document id is required")
	}

	if s.useInMemory {
		s.memMu.RLock()
		stored, ok := s.memDocs[id]
		s.memMu.RUnlock()
		if ok {
			metadata := cloneMetadata(stored.Metadata)
			doc := &Document{
				ID:       id,
				Content:  stored.Content,
				Metadata: metadata,
			}
			if metadata != nil {
				doc.TaskType = stringFromMetadata(metadata, "task_type")
				doc.Domain = stringFromMetadata(metadata, "domain")
			}
			return doc, nil
		}
	}

	if s.hanaDocuments != nil {
		hanaDoc, err := s.hanaDocuments.GetDocument(ctx, id, "system", "detail")
		if err == nil && hanaDoc != nil {
			return &Document{
				ID:       hanaDoc.ID,
				Content:  hanaDoc.Content,
				Metadata: hanaDoc.Metadata,
			}, nil
		}
	}

	source, err := s.elastic.getDocument(ctx, id)
	if err != nil {
		return nil, err
	}
	if source == nil {
		return nil, nil
	}

	doc := &Document{
		ID: id,
	}
	if content, ok := source["content"].(string); ok {
		doc.Content = content
	}
	if metadata, ok := source["metadata"].(map[string]interface{}); ok {
		doc.Metadata = metadata
	}
	if title, ok := source["title"].(string); ok {
		doc.Title = title
	}
	if taskType, ok := source["task_type"].(string); ok {
		doc.TaskType = taskType
	}
	if domain, ok := source["domain"].(string); ok {
		doc.Domain = domain
	}

	return doc, nil
}

// GetDocumentsByTaskType returns documents filtered by task type using Elasticsearch filters.
func (s *SearchService) GetDocumentsByTaskType(ctx context.Context, taskType string, topK int) ([]Document, error) {
	if s.useInMemory {
		return s.searchInMemory(nil, topK, map[string]string{"metadata.task_type": taskType}), nil
	}

	filters := map[string]string{
		"metadata.task_type": taskType,
	}
	hits, err := s.elastic.searchByFilters(ctx, filters, topK)
	if err != nil {
		return nil, err
	}

	docs := make([]Document, 0, len(hits))
	for _, hit := range hits {
		doc := Document{ID: hit.ID, Score: hit.Score, Similarity: hit.Score}
		if sourceContent, ok := hit.Source["content"].(string); ok {
			doc.Content = sourceContent
		}
		if metadata, ok := hit.Source["metadata"].(map[string]interface{}); ok {
			doc.Metadata = metadata
			if title, ok := metadata["title"].(string); ok {
				doc.Title = title
			}
			if domain, ok := metadata["domain"].(string); ok {
				doc.Domain = domain
			}
		}
		docs = append(docs, doc)
	}

	return docs, nil
}

// RemoveDocument deletes the document from backing stores.
func (s *SearchService) RemoveDocument(id string) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if s.useInMemory {
		s.memMu.Lock()
		delete(s.memDocs, id)
		s.memMu.Unlock()
	}

	if s.hanaDocuments != nil {
		_ = s.hanaDocuments.DeleteDocument(ctx, id, "system")
	}
	if s.elastic != nil {
		_ = s.elastic.deleteDocument(ctx, id)
	}
}

// GetIndexSize returns the document count from Elasticsearch.
func (s *SearchService) GetIndexSize() int {
	if s.useInMemory {
		s.memMu.RLock()
		defer s.memMu.RUnlock()
		return len(s.memDocs)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if s.elastic == nil {
		return 0
	}

	count, err := s.elastic.countDocuments(ctx)
	if err != nil {
		return 0
	}
	return int(count)
}

// Close releases resources.
func (s *SearchService) Close() error {
	var err error

	if s.ownsModel && s.model != nil {
		err = s.model.Close()
	}
	if s.cache != nil {
		_ = s.cache.Close()
	}
	if s.hanaLogger != nil {
		if closeErr := s.hanaLogger.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}
	if s.hanaDocuments != nil {
		if closeErr := s.hanaDocuments.Close(); closeErr != nil && err == nil {
			err = closeErr
		}
	}
	return err
}

// Model exposes the underlying embedding model for HTTP handlers and integrations.
func (s *SearchService) Model() *SearchModelWithLocalAI {
	return s.model
}

// UpdateAgentCatalog updates the cached Agent SDK catalog.
func (s *SearchService) UpdateAgentCatalog(cat AgentCatalog) {
	s.agentCatalog.Update(cat)
}

// AgentCatalogSnapshot returns the cached catalog (if any) and the timestamp it was last refreshed.
func (s *SearchService) AgentCatalogSnapshot() (*AgentCatalog, time.Time) {
	return s.agentCatalog.Snapshot()
}

// SetFlightAddr records the Arrow Flight endpoint associated with the catalog.
func (s *SearchService) SetFlightAddr(addr string) {
	s.agentCatalog.SetFlightAddr(addr)
}

// FlightAddr returns the configured Arrow Flight endpoint, if any.
func (s *SearchService) FlightAddr() string {
	return s.agentCatalog.FlightAddr()
}

func (s *SearchService) embedWithCache(ctx context.Context, text string) ([]float64, error) {
	if s.cache != nil {
		if embedding, ok := s.cache.Get(ctx, text); ok {
			return embedding, nil
		}
	}

	embedding, err := s.model.Embed(ctx, text)
	if err != nil {
		return nil, err
	}

	if s.cache != nil {
		s.cache.Set(ctx, text, embedding)
	}

	return embedding, nil
}

func (s *SearchService) storeAndIndex(ctx context.Context, id, content string, metadata map[string]interface{}, embedding []float64) error {
	now := time.Now()
	metaCopy := make(map[string]interface{}, len(metadata)+1)
	for k, v := range metadata {
		metaCopy[k] = v
	}
	metaCopy["updated_at"] = now.Format(time.RFC3339)
	if _, ok := metaCopy["created_at"]; !ok {
		metaCopy["created_at"] = now.Format(time.RFC3339)
	}

	if s.hanaDocuments != nil {
		doc := &storage.Document{
			ID:           id,
			Content:      content,
			Metadata:     metaCopy,
			PrivacyLevel: s.privacyConfig.PrivacyLevel.String(),
			CreatedAt:    now,
			UpdatedAt:    now,
		}
		if err := s.hanaDocuments.StoreDocument(ctx, doc); err != nil {
			return fmt.Errorf("store document in HANA: %w", err)
		}
	}

	if s.elastic != nil {
		body := map[string]interface{}{
			"content":    content,
			"metadata":   metaCopy,
			"embedding":  embedding,
			"updated_at": now,
		}
		if createdAt, ok := metadata["created_at"]; ok {
			body["created_at"] = createdAt
		} else {
			body["created_at"] = now
		}

		if err := s.elastic.indexDocument(ctx, id, body); err != nil {
			return err
		}
	} else if s.useInMemory {
		if s.memDocs == nil {
			s.memDocs = make(map[string]*storedDocument)
		}
		s.memMu.Lock()
		s.memDocs[id] = &storedDocument{
			Content:   content,
			Metadata:  cloneMetadata(metaCopy),
			Embedding: cloneEmbedding(embedding),
			CreatedAt: now,
			UpdatedAt: now,
		}
		s.memMu.Unlock()
	} else {
		return fmt.Errorf("no search index configured")
	}

	return nil
}

func (s *SearchService) logSearchAsync(query string, resultCount int, results []SearchResult, start time.Time) {
	if s.hanaLogger == nil {
		return
	}

	latencyMs := time.Since(start).Milliseconds()
	topResultID := ""
	if len(results) > 0 {
		topResultID = results[0].ID
	}

	searchLog := &storage.SearchLog{
		QueryHash:         storage.AnonymizeString(query),
		ResultCount:       resultCount,
		TopResultID:       topResultID,
		LatencyMs:         latencyMs,
		UserIDHash:        storage.AnonymizeString("anonymous"),
		SessionID:         "session-" + time.Now().Format("20060102-150405"),
		Timestamp:         time.Now(),
		PrivacyBudgetUsed: storage.PrivacyBudgetCosts.SearchQuery,
	}

	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = s.hanaLogger.LogSearch(ctx, searchLog)
	}()
}

func cloneMetadata(meta map[string]interface{}) map[string]interface{} {
	if meta == nil {
		return nil
	}
	copy := make(map[string]interface{}, len(meta))
	for k, v := range meta {
		copy[k] = v
	}
	return copy
}

func cloneEmbedding(embedding []float64) []float64 {
	if embedding == nil {
		return nil
	}
	dup := make([]float64, len(embedding))
	copy(dup, embedding)
	return dup
}

func metadataMatchesFilters(metadata map[string]interface{}, filters map[string]string) bool {
	if len(filters) == 0 {
		return true
	}

	for rawKey, rawValue := range filters {
		key := strings.TrimSpace(rawKey)
		expected := strings.TrimSpace(rawValue)
		if key == "" || expected == "" {
			continue
		}

		switch {
		case strings.HasPrefix(key, "metadata."):
			field := strings.TrimPrefix(key, "metadata.")
			if strings.TrimSpace(stringFromMetadata(metadata, field)) != expected {
				return false
			}
		case key == "task_type":
			if strings.TrimSpace(stringFromMetadata(metadata, "task_type")) != expected {
				return false
			}
		case key == "domain":
			if strings.TrimSpace(stringFromMetadata(metadata, "domain")) != expected {
				return false
			}
		default:
			if strings.TrimSpace(stringFromMetadata(metadata, key)) != expected {
				return false
			}
		}
	}

	return true
}

func stringFromMetadata(metadata map[string]interface{}, key string) string {
	if metadata == nil {
		return ""
	}
	if value, ok := metadata[key]; ok {
		switch v := value.(type) {
		case string:
			return v
		case fmt.Stringer:
			return v.String()
		default:
			return fmt.Sprintf("%v", v)
		}
	}
	return ""
}
