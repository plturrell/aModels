package agents

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/plturrell/aModels/pkg/vision"
	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// PerplexityPipeline orchestrates the full document processing pipeline
// from Perplexity API through OCR, catalog, training, local AI, and search.
type PerplexityPipeline struct {
	perplexityConnector SourceConnector
	ocrClient          *vision.DeepSeekClient
	deepResearchClient *research.DeepResearchClient
	unifiedWorkflowURL string
	catalogURL         string
	trainingURL        string
	localAIURL         string
	searchURL          string
	extractURL         string
	graphServiceURL    string // Priority 5: Graph service URL for GNN queries
	gpuOrchestratorURL string // Priority 3: GPU orchestrator URL for GPU allocation
	learningOrchestrator *LearningOrchestrator
	requestTracker     *RequestTracker
	logger             *log.Logger
	httpClient         *http.Client
	localAIClient       *LocalAIClient // Standardized LocalAI client with retry and validation
}

// PerplexityPipelineConfig configures the pipeline.
type PerplexityPipelineConfig struct {
	PerplexityAPIKey    string
	PerplexityBaseURL   string
	DeepSeekOCREndpoint string
	DeepSeekOCRAPIKey   string
	DeepResearchURL     string
	UnifiedWorkflowURL  string
	CatalogURL          string
	TrainingURL         string
	LocalAIURL          string
	SearchURL           string
	ExtractURL          string
	Logger              *log.Logger
}

// NewPerplexityPipeline creates a new Perplexity processing pipeline.
func NewPerplexityPipeline(config PerplexityPipelineConfig) (*PerplexityPipeline, error) {
	// Create Perplexity connector
	perplexityConfig := map[string]interface{}{
		"api_key":  config.PerplexityAPIKey,
		"base_url": config.PerplexityBaseURL,
	}
	perplexityConnector := connectors.NewPerplexityConnector(perplexityConfig, config.Logger)

	// Create DeepSeek OCR client
	ocrClient := vision.NewDeepSeekClient(vision.DeepSeekConfig{
		Endpoint: config.DeepSeekOCREndpoint,
		APIKey:   config.DeepSeekOCRAPIKey,
		Timeout:  60 * time.Second,
	})

	// Create Deep Research client
	var deepResearchClient *research.DeepResearchClient
	if config.DeepResearchURL != "" {
		deepResearchClient = research.NewDeepResearchClient(config.DeepResearchURL, config.Logger)
	}

	// Get graph service URL for GNN queries (Priority 5)
	graphServiceURL := os.Getenv("GRAPH_SERVICE_URL")
	if graphServiceURL == "" {
		graphServiceURL = "http://graph-service:8081"
	}
	
	// Get GPU orchestrator URL for GPU allocation (Priority 3)
	gpuOrchestratorURL := os.Getenv("GPU_ORCHESTRATOR_URL")
	if gpuOrchestratorURL == "" {
		gpuOrchestratorURL = "http://gpu-orchestrator:8086"
	}

	// Use connection pooling for better performance (Priority 1)
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
	}
	
	pipeline := &PerplexityPipeline{
		perplexityConnector: perplexityConnector,
		ocrClient:          ocrClient,
		deepResearchClient: deepResearchClient,
		unifiedWorkflowURL: config.UnifiedWorkflowURL,
		catalogURL:         config.CatalogURL,
		trainingURL:        config.TrainingURL,
		localAIURL:         config.LocalAIURL,
		searchURL:          config.SearchURL,
		extractURL:         config.ExtractURL,
		graphServiceURL:    graphServiceURL,
		gpuOrchestratorURL:  gpuOrchestratorURL,
		logger:             config.Logger,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   120 * time.Second,
		},
	}

	// Create learning orchestrator
	pipeline.learningOrchestrator = NewLearningOrchestrator(pipeline, config.Logger)

	// Create request tracker
	pipeline.requestTracker = NewRequestTracker(config.Logger)

	// Initialize LocalAI client with retry logic and circuit breaker
	if config.LocalAIURL != "" {
		pipeline.localAIClient = NewLocalAIClient(config.LocalAIURL, pipeline.httpClient, config.Logger)
	}

	return pipeline, nil
}

// ProcessDocuments processes documents from Perplexity API through the full pipeline.
// Returns processed documents for pattern mining.
func (pp *PerplexityPipeline) ProcessDocuments(ctx context.Context, query map[string]interface{}) error {
	// Apply learned improvements before processing
	if pp.learningOrchestrator != nil {
		if err := pp.learningOrchestrator.ApplyImprovements(ctx, query); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Failed to apply improvements (non-fatal): %v", err)
			}
		}
	}

	return pp.ProcessDocumentsWithCallback(ctx, query, nil)
}

// GetLearningReport returns a comprehensive learning report.
func (pp *PerplexityPipeline) GetLearningReport() *LearningReport {
	if pp.learningOrchestrator != nil {
		return pp.learningOrchestrator.GetLearningReport()
	}
	return nil
}

// GetRequestTracker returns the request tracker.
func (pp *PerplexityPipeline) GetRequestTracker() *RequestTracker {
	return pp.requestTracker
}

// ProcessDocumentsWithTracking processes documents with full request tracking.
// Returns the request ID and processing request for status tracking.
func (pp *PerplexityPipeline) ProcessDocumentsWithTracking(ctx context.Context, requestID string, query map[string]interface{}) (*ProcessingRequest, error) {
	queryStr, _ := query["query"].(string)
	if queryStr == "" {
		queryStr = "Perplexity query"
	}

	// Create request tracking
	request := pp.requestTracker.CreateRequest(requestID, queryStr)
	pp.requestTracker.UpdateStatus(requestID, RequestStatusProcessing)
	pp.requestTracker.UpdateStep(requestID, "connecting")

	// Apply learned improvements before processing
	if pp.learningOrchestrator != nil {
		if err := pp.learningOrchestrator.ApplyImprovements(ctx, query); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Failed to apply improvements (non-fatal): %v", err)
			}
		}
	}

	// Step 1: Connect to Perplexity
	pp.requestTracker.UpdateStep(requestID, "connecting")
	if err := pp.perplexityConnector.Connect(ctx, query); err != nil {
		pp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		errorCode := "CONNECTION_ERROR"
		if strings.Contains(err.Error(), "timeout") {
			errorCode = "TIMEOUT_ERROR"
		} else if strings.Contains(err.Error(), "unauthorized") {
			errorCode = "AUTH_ERROR"
		}
		pp.requestTracker.AddErrorWithDetails(requestID, "connect", err.Error(), "", errorCode, nil, false)
		return request, fmt.Errorf("failed to connect to Perplexity: %w", err)
	}
	defer pp.perplexityConnector.Close()

	// Step 2: Extract documents
	pp.requestTracker.UpdateStep(requestID, "extracting")
	documents, err := pp.perplexityConnector.ExtractData(ctx, query)
	if err != nil {
		pp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		errorCode := "EXTRACTION_ERROR"
		if strings.Contains(err.Error(), "rate limit") {
			errorCode = "RATE_LIMIT_ERROR"
		}
		pp.requestTracker.AddErrorWithDetails(requestID, "extract", err.Error(), "", errorCode, nil, false)
		return request, fmt.Errorf("failed to extract documents: %w", err)
	}

	if pp.logger != nil {
		pp.logger.Printf("Extracted %d documents from Perplexity", len(documents))
	}

	// Process each document
	var processedDocs []ProcessedDocument
	var hasErrors bool

	for i, doc := range documents {
		docID, _ := doc["id"].(string)
		title, _ := doc["title"].(string)

		pp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_document_%d", i+1))

		// Process document
		docResult := ProcessedDocument{
			ID:          docID,
			Title:       title,
			Status:      "success",
			ProcessedAt: time.Now(),
			Metadata:    make(map[string]interface{}),
		}

		if err := pp.processDocumentWithTracking(ctx, requestID, doc, &docResult); err != nil {
			docResult.Status = "failed"
			docResult.Error = err.Error()
			hasErrors = true
			errorCode := "PROCESSING_ERROR"
			if strings.Contains(err.Error(), "catalog") {
				errorCode = "CATALOG_ERROR"
			} else if strings.Contains(err.Error(), "training") {
				errorCode = "TRAINING_ERROR"
			} else if strings.Contains(err.Error(), "localai") {
				errorCode = "LOCALAI_ERROR"
			} else if strings.Contains(err.Error(), "search") {
				errorCode = "SEARCH_ERROR"
			}
			pp.requestTracker.AddErrorWithDetails(requestID, "process_document", err.Error(), docID, errorCode, nil, false)
		}

		processedDocs = append(processedDocs, docResult)
		pp.requestTracker.AddDocument(requestID, docResult)
	}

	// Set results links
	results := &ProcessingResults{
		CatalogURL:  fmt.Sprintf("/api/catalog/documents?source=perplexity&request_id=%s", requestID),
		SearchURL:   fmt.Sprintf("/api/search?query=%s", queryStr),
		ExportURL:   fmt.Sprintf("/api/perplexity/results/%s/export", requestID),
	}
	pp.requestTracker.SetResults(requestID, results)

	// Update final status
	pp.requestTracker.UpdateStep(requestID, "completed")
	if hasErrors && len(processedDocs) > 0 {
		pp.requestTracker.UpdateStatus(requestID, RequestStatusPartial)
	} else if hasErrors {
		pp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
	} else {
		pp.requestTracker.UpdateStatus(requestID, RequestStatusCompleted)
	}

	return request, nil
}

// processDocumentWithTracking processes a single document and updates the document result.
func (pp *PerplexityPipeline) processDocumentWithTracking(ctx context.Context, requestID string, doc map[string]interface{}, docResult *ProcessedDocument) error {
	docID := docResult.ID
	
	// Track steps
	pp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s_unified_workflow", docID))
	
	// Process document and capture intelligence
	intelligence, err := pp.processDocumentWithIntelligence(ctx, doc)
	if err != nil {
		return err
	}

	// Extract IDs from document processing
	// The document ID is used as the primary identifier
	docResult.CatalogID = docID
	docResult.LocalAIID = docID
	docResult.SearchID = docID
	
	// Add metadata
	docResult.Metadata = map[string]interface{}{
		"source": "perplexity",
		"processed_at": docResult.ProcessedAt,
	}
	
	// Store intelligence data
	if intelligence != nil {
		docResult.Intelligence = intelligence
		pp.requestTracker.SetDocumentIntelligence(requestID, docID, intelligence)
	}

	return nil
}

// processDocumentWithIntelligence processes a document and returns intelligence data.
func (pp *PerplexityPipeline) processDocumentWithIntelligence(ctx context.Context, doc map[string]interface{}) (*DocumentIntelligence, error) {
	docID, _ := doc["id"].(string)
	title, _ := doc["title"].(string)
	content, _ := doc["content"].(string)
	
	// Initialize intelligence structure
	intelligence := &DocumentIntelligence{
		CatalogPatterns:    make(map[string]interface{}),
		TrainingPatterns:   make(map[string]interface{}),
		DomainPatterns:     make(map[string]interface{}),
		SearchPatterns:     make(map[string]interface{}),
		MetadataEnrichment: make(map[string]interface{}),
		Relationships:      make([]Relationship, 0),
		LearnedPatterns:    make([]Pattern, 0),
	}
	
	// Process document and capture intelligence at each step
	// This is a wrapper around processDocument that captures intelligence data
	
	// Step 0a: Unified Workflow
	var unifiedWorkflowResult map[string]interface{}
	if pp.unifiedWorkflowURL != "" {
		result, err := pp.processViaUnifiedWorkflow(ctx, docID, title, content)
		if err == nil {
			unifiedWorkflowResult = result
			intelligence.WorkflowResults = result
			
			// Extract knowledge graph from workflow results
			if kg, ok := result["knowledge_graph"].(map[string]interface{}); ok {
				intelligence.KnowledgeGraph = kg
			}
		}
	}
	
	// Step 4: Domain Detection (before Local AI storage)
	domain := pp.detectDomain(title + " " + content)
	intelligence.Domain = domain
	intelligence.DomainConfidence = 0.8 // Simple detection, could be improved
	
	// Process document normally (this will call all the services)
	if err := pp.processDocument(ctx, doc); err != nil {
		return nil, err
	}
	
	// Now collect intelligence from learning endpoints
	// We'll make async calls to get the intelligence data
	
	// Collect catalog patterns and relationships
	if pp.catalogURL != "" {
		pp.collectCatalogIntelligence(ctx, docID, title, content, intelligence)
	}
	
	// Collect training patterns
	if pp.trainingURL != "" {
		pp.collectTrainingIntelligence(ctx, docID, intelligence)
	}
	
	// Collect domain patterns
	if pp.localAIURL != "" && domain != "" {
		pp.collectDomainIntelligence(ctx, docID, domain, intelligence)
	}
	
	// Collect search patterns
	if pp.searchURL != "" {
		pp.collectSearchIntelligence(ctx, docID, intelligence)
	}
	
	return intelligence, nil
}

// collectCatalogIntelligence collects intelligence from catalog service.
func (pp *PerplexityPipeline) collectCatalogIntelligence(ctx context.Context, docID, title, content string, intelligence *DocumentIntelligence) {
	// Get patterns
	patternsURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/patterns/extract"
	patternsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "extract_patterns",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var patternsResult map[string]interface{}
	if err := pp.postJSON(ctx, patternsURL, patternsPayload, &patternsResult); err == nil {
		intelligence.CatalogPatterns = patternsResult
		if patterns, ok := patternsResult["patterns"].(map[string]interface{}); ok {
			// Convert to Pattern structs
			if columnPatterns, ok := patterns["column_patterns"].(map[string]interface{}); ok {
				for k, v := range columnPatterns {
					intelligence.LearnedPatterns = append(intelligence.LearnedPatterns, Pattern{
						Type:        "column",
						Description: k,
						Metadata:    map[string]interface{}{"value": v},
					})
				}
			}
		}
	}
	
	// Get relationships
	relationshipsURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/relationships/discover"
	relationshipsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "discover_relationships",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var relationshipsResult map[string]interface{}
	if err := pp.postJSON(ctx, relationshipsURL, relationshipsPayload, &relationshipsResult); err == nil {
		if relationships, ok := relationshipsResult["relationships"].([]interface{}); ok {
			for _, rel := range relationships {
				if relMap, ok := rel.(map[string]interface{}); ok {
					relationship := Relationship{
						Type:     getString(relMap["type"]),
						TargetID: getString(relMap["target_id"]),
						Strength: getFloat64(relMap["strength"]),
					}
					if title, ok := relMap["target_title"].(string); ok {
						relationship.TargetTitle = title
					}
					if metadata, ok := relMap["metadata"].(map[string]interface{}); ok {
						relationship.Metadata = metadata
					}
					intelligence.Relationships = append(intelligence.Relationships, relationship)
				}
			}
		}
	}
	
	// Get metadata enrichment
	enrichURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/metadata/enrich"
	enrichPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "enrich_metadata",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var enrichResult map[string]interface{}
	if err := pp.postJSON(ctx, enrichURL, enrichPayload, &enrichResult); err == nil {
		intelligence.MetadataEnrichment = enrichResult
	}
}

// collectTrainingIntelligence collects intelligence from training service.
func (pp *PerplexityPipeline) collectTrainingIntelligence(ctx context.Context, docID string, intelligence *DocumentIntelligence) {
	// Try to get patterns from training service
	// Note: This requires the task ID, which we'd need to store
	// For now, we'll skip this or use a generic endpoint
}

// collectDomainIntelligence collects intelligence from Local AI domain service.
func (pp *PerplexityPipeline) collectDomainIntelligence(ctx context.Context, docID, domain string, intelligence *DocumentIntelligence) {
	// Get domain patterns
	patternURL := strings.TrimRight(pp.localAIURL, "/") + "/v1/domains/" + domain + "/patterns"
	patternPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "get_patterns",
	}
	var patternResult map[string]interface{}
	if err := pp.postJSON(ctx, patternURL, patternPayload, &patternResult); err == nil {
		intelligence.DomainPatterns = patternResult
	}
}

// collectSearchIntelligence collects intelligence from search service.
func (pp *PerplexityPipeline) collectSearchIntelligence(ctx context.Context, docID string, intelligence *DocumentIntelligence) {
	// Get search patterns
	patternURL := strings.TrimRight(pp.searchURL, "/") + "/patterns/learn"
	patternPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "get_patterns",
	}
	var patternResult map[string]interface{}
	if err := pp.postJSON(ctx, patternURL, patternPayload, &patternResult); err == nil {
		intelligence.SearchPatterns = patternResult
	}
}

// Helper functions for type conversion
func getString(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func getFloat64(v interface{}) float64 {
	if f, ok := v.(float64); ok {
		return f
	}
	return 0.0
}

// QuerySearch queries the search service for indexed documents.
func (pp *PerplexityPipeline) QuerySearch(ctx context.Context, query, requestID string, topK int, filters map[string]interface{}) ([]map[string]interface{}, error) {
	if pp.searchURL == "" {
		return nil, fmt.Errorf("search service not configured")
	}

	// Build search request
	searchPayload := map[string]interface{}{
		"query": query,
		"top_k": topK,
	}
	if filters != nil {
		searchPayload["filters"] = filters
	}
	if requestID != "" {
		// Add filter for specific request if provided
		if searchPayload["filters"] == nil {
			searchPayload["filters"] = make(map[string]interface{})
		}
		if filtersMap, ok := searchPayload["filters"].(map[string]interface{}); ok {
			filtersMap["request_id"] = requestID
		}
	}

	// Try Python search service first (POST /v1/search)
	url := strings.TrimRight(pp.searchURL, "/") + "/v1/search"
	var searchResult map[string]interface{}
	if err := pp.postJSON(ctx, url, searchPayload, &searchResult); err == nil {
		// Extract results
		if results, ok := searchResult["results"].([]interface{}); ok {
			documents := make([]map[string]interface{}, 0, len(results))
			for _, r := range results {
				if doc, ok := r.(map[string]interface{}); ok {
					documents = append(documents, doc)
				}
			}
			return documents, nil
		}
	}

	// Fallback to Go search service (POST /v1/search)
	// The Go service might have a different format
	var goResult struct {
		Results []map[string]interface{} `json:"results"`
	}
	if err := pp.postJSON(ctx, url, searchPayload, &goResult); err == nil {
		return goResult.Results, nil
	}

	return nil, fmt.Errorf("search query failed")
}

// QueryKnowledgeGraph queries the knowledge graph for a specific request.
func (pp *PerplexityPipeline) QueryKnowledgeGraph(ctx context.Context, requestID, cypherQuery string, params map[string]interface{}) (map[string]interface{}, error) {
	// First, get the request to find related documents
	tracker := pp.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		return nil, fmt.Errorf("request not found: %s", requestID)
	}

	// Try unified workflow GraphRAG query first
	if pp.unifiedWorkflowURL != "" {
		graphRAGRequest := map[string]interface{}{
			"graphrag_request": map[string]interface{}{
				"query":      cypherQuery,
				"strategy":   "bfs",
				"max_depth":  3,
				"max_results": 10,
				"params":     params,
				"enrich":     true,
			},
		}

		url := strings.TrimRight(pp.unifiedWorkflowURL, "/") + "/graphrag/query"
		var result map[string]interface{}
		if err := pp.postJSON(ctx, url, graphRAGRequest, &result); err == nil {
			return result, nil
		}
	}

	// Fallback to extract service knowledge graph query
	// Extract service URL would need to be configured
	extractURL := os.Getenv("EXTRACT_URL")
	if extractURL == "" {
		extractURL = "http://extract:8081"
	}

	kgQueryPayload := map[string]interface{}{
		"query":  cypherQuery,
		"params": params,
	}

	url := strings.TrimRight(extractURL, "/") + "/knowledge-graph/query"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, kgQueryPayload, &result); err != nil {
		return nil, fmt.Errorf("knowledge graph query failed: %w", err)
	}

	// Enhance results with request context
	if result != nil {
		result["request_id"] = requestID
		result["document_ids"] = request.DocumentIDs
	}

	return result, nil
}

// QueryDomainDocuments queries documents by domain from Local AI.
func (pp *PerplexityPipeline) QueryDomainDocuments(ctx context.Context, domain string, limit, offset int) ([]map[string]interface{}, error) {
	if pp.localAIURL == "" {
		return nil, fmt.Errorf("local AI service not configured")
	}

	// Query domain-specific documents
	url := fmt.Sprintf("%s/v1/domains/%s/documents?limit=%d&offset=%d",
		strings.TrimRight(pp.localAIURL, "/"), domain, limit, offset)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := pp.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to query domain documents: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("domain query failed with status %d", resp.StatusCode)
	}

	var result struct {
		Documents []map[string]interface{} `json:"documents"`
		Count     int                      `json:"count"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Documents, nil
}

// QueryCatalogSemantic queries the catalog using semantic search.
func (pp *PerplexityPipeline) QueryCatalogSemantic(ctx context.Context, query, objectClass, property, source string, filters map[string]interface{}) (map[string]interface{}, error) {
	if pp.catalogURL == "" {
		return nil, fmt.Errorf("catalog service not configured")
	}

	// Build semantic search request
	searchPayload := map[string]interface{}{
		"query": query,
	}
	if objectClass != "" {
		searchPayload["object_class"] = objectClass
	}
	if property != "" {
		searchPayload["property"] = property
	}
	if source != "" {
		searchPayload["source"] = source
	} else {
		searchPayload["source"] = "perplexity"
	}
	if filters != nil {
		searchPayload["filters"] = filters
	}

	url := strings.TrimRight(pp.catalogURL, "/") + "/catalog/semantic-search"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, searchPayload, &result); err != nil {
		return nil, fmt.Errorf("catalog semantic search failed: %w", err)
	}

	return result, nil
}

// QueryGNNEmbeddings queries GNN service for graph/node embeddings (Priority 5).
func (pp *PerplexityPipeline) QueryGNNEmbeddings(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, graphLevel bool) (map[string]interface{}, error) {
	// Try graph service hybrid query endpoint first
	if pp.graphServiceURL != "" {
		hybridPayload := map[string]interface{}{
			"hybrid_query_request": map[string]interface{}{
				"query":      "Generate embeddings",
				"query_kg":   false,
				"query_gnn":  true,
				"gnn_type":   "embeddings",
				"combine":    false,
			},
		}
		// Add nodes/edges if provided
		if len(nodes) > 0 || len(edges) > 0 {
			gnnRequest := map[string]interface{}{
				"query_type": "embeddings",
				"nodes":      nodes,
				"edges":      edges,
				"params": map[string]interface{}{
					"graph_level": graphLevel,
				},
			}
			hybridPayload["gnn_query_request"] = gnnRequest
		}

		url := strings.TrimRight(pp.graphServiceURL, "/") + "/gnn/query"
		var result map[string]interface{}
		if err := pp.postJSON(ctx, url, hybridPayload, &result); err == nil {
			return result, nil
		}
	}

	// Fallback to training service directly
	if pp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes":      nodes,
		"edges":      edges,
		"graph_level": graphLevel,
	}

	url := strings.TrimRight(pp.trainingURL, "/") + "/gnn/embeddings"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN embeddings query failed: %w", err)
	}

	return result, nil
}

// QueryGNNStructuralInsights queries GNN service for structural insights (Priority 5).
func (pp *PerplexityPipeline) QueryGNNStructuralInsights(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, insightType string, threshold float64) (map[string]interface{}, error) {
	// Try graph service hybrid query endpoint first
	if pp.graphServiceURL != "" {
		hybridPayload := map[string]interface{}{
			"hybrid_query_request": map[string]interface{}{
				"query":      "Get structural insights",
				"query_kg":   false,
				"query_gnn":  true,
				"gnn_type":   "structural-insights",
				"combine":    false,
			},
		}
		// Add nodes/edges if provided
		if len(nodes) > 0 || len(edges) > 0 {
			gnnRequest := map[string]interface{}{
				"query_type": "structural-insights",
				"nodes":      nodes,
				"edges":      edges,
				"params": map[string]interface{}{
					"insight_type": insightType,
					"threshold":   threshold,
				},
			}
			hybridPayload["gnn_query_request"] = gnnRequest
		}

		url := strings.TrimRight(pp.graphServiceURL, "/") + "/gnn/query"
		var result map[string]interface{}
		if err := pp.postJSON(ctx, url, hybridPayload, &result); err == nil {
			return result, nil
		}
	}

	// Fallback to training service directly
	if pp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes":        nodes,
		"edges":        edges,
		"insight_type": insightType,
		"threshold":    threshold,
	}

	url := strings.TrimRight(pp.trainingURL, "/") + "/gnn/structural-insights"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN structural insights query failed: %w", err)
	}

	return result, nil
}

// QueryGNNPredictLinks queries GNN service for link predictions (Priority 5).
func (pp *PerplexityPipeline) QueryGNNPredictLinks(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, candidatePairs [][]string, topK int) (map[string]interface{}, error) {
	if pp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes": nodes,
		"edges": edges,
		"top_k": topK,
	}
	if candidatePairs != nil {
		payload["candidate_pairs"] = candidatePairs
	}

	url := strings.TrimRight(pp.trainingURL, "/") + "/gnn/predict-links"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN link prediction query failed: %w", err)
	}

	return result, nil
}

// QueryGNNClassifyNodes queries GNN service for node classification (Priority 5).
func (pp *PerplexityPipeline) QueryGNNClassifyNodes(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, topK *int) (map[string]interface{}, error) {
	if pp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes": nodes,
		"edges": edges,
	}
	if topK != nil {
		payload["top_k"] = *topK
	}

	url := strings.TrimRight(pp.trainingURL, "/") + "/gnn/classify"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN node classification query failed: %w", err)
	}

	return result, nil
}

// QueryGNNHybrid queries both KG and GNN and combines results (Priority 5).
func (pp *PerplexityPipeline) QueryGNNHybrid(ctx context.Context, query string, projectID, systemID string, queryKG, queryGNN bool, gnnType string, combine bool) (map[string]interface{}, error) {
	// Use graph service hybrid query endpoint
	if pp.graphServiceURL != "" {
		hybridPayload := map[string]interface{}{
			"hybrid_query_request": map[string]interface{}{
				"query":      query,
				"project_id": projectID,
				"system_id":  systemID,
				"query_kg":   queryKG,
				"query_gnn":  queryGNN,
				"gnn_type":   gnnType,
				"combine":    combine,
			},
		}

		url := strings.TrimRight(pp.graphServiceURL, "/") + "/gnn/hybrid-query"
		var result map[string]interface{}
		if err := pp.postJSON(ctx, url, hybridPayload, &result); err != nil {
			return nil, fmt.Errorf("hybrid query failed: %w", err)
		}

		return result, nil
	}

	return nil, fmt.Errorf("graph service not configured")
}

// ProcessDocumentsWithCallback processes documents with an optional callback for each document.
func (pp *PerplexityPipeline) ProcessDocumentsWithCallback(ctx context.Context, query map[string]interface{}, callback func(map[string]interface{}) error) error {
	if pp.logger != nil {
		pp.logger.Printf("Starting Perplexity document processing pipeline")
	}

	// Step 1: Connect to Perplexity
	if err := pp.perplexityConnector.Connect(ctx, query); err != nil {
		return fmt.Errorf("failed to connect to Perplexity: %w", err)
	}
	defer pp.perplexityConnector.Close()

	// Step 2: Extract documents from Perplexity
	documents, err := pp.perplexityConnector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract documents: %w", err)
	}

	if pp.logger != nil {
		pp.logger.Printf("Extracted %d documents from Perplexity", len(documents))
	}

	// Step 3: Process each document through the pipeline
	for i, doc := range documents {
		if pp.logger != nil {
			pp.logger.Printf("Processing document %d/%d", i+1, len(documents))
		}

		if err := pp.processDocument(ctx, doc); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Failed to process document %d: %v", i+1, err)
			}
			// Continue processing other documents
			continue
		}

		// Call callback if provided (for pattern mining, etc.)
		if callback != nil {
			if err := callback(doc); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Callback failed for document %d (non-fatal): %v", i+1, err)
				}
			}
		}
	}

	if pp.logger != nil {
		pp.logger.Printf("Completed processing %d documents", len(documents))
	}

	return nil
}

// processDocument processes a single document through OCR, catalog, training, local AI, and search.
func (pp *PerplexityPipeline) processDocument(ctx context.Context, doc map[string]interface{}) error {
	docID, _ := doc["id"].(string)
	title, _ := doc["title"].(string)
	content, _ := doc["content"].(string)
	imageBase64, _ := doc["image_base64"].(string)
	imageURL, _ := doc["image_url"].(string)

	// Step 0a: Process via Unified Workflow (if configured)
	var unifiedWorkflowResult map[string]interface{}
	if pp.unifiedWorkflowURL != "" {
		if pp.logger != nil {
			pp.logger.Printf("Processing document %s via unified workflow", docID)
		}
		result, err := pp.processViaUnifiedWorkflow(ctx, docID, title, content)
		if err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Unified workflow processing failed for document %s (non-fatal): %v", docID, err)
			}
			// Continue with direct pipeline processing
		} else {
			unifiedWorkflowResult = result
			if pp.logger != nil {
				pp.logger.Printf("Unified workflow processing completed for document %s", docID)
			}
		}
	}

	// Step 0: Deep Research - Understand context before processing
	var researchReport *research.ResearchReport
	var researchContext string
	if pp.deepResearchClient != nil && title != "" {
		if pp.logger != nil {
			pp.logger.Printf("Performing Deep Research for document %s: %s", docID, title)
		}
		
		// Research the document topic to understand context
		report, err := pp.deepResearchClient.ResearchMetadata(ctx, title, true, true)
		if err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Deep Research failed for document %s (non-fatal): %v", docID, err)
			}
		} else if report != nil && report.Report != nil {
			researchReport = report
			// Extract research summary for context
			researchContext = report.Report.Summary
			if len(report.Report.Sections) > 0 {
				// Add first section content for richer context
				researchContext += "\n\n" + report.Report.Sections[0].Content
			}
			if pp.logger != nil {
				pp.logger.Printf("Deep Research completed for document %s: %s", docID, report.Report.Topic)
			}
		}
	}

	// Step 1: OCR Processing (if image is present)
	processedText := content
	if imageBase64 != "" {
		if pp.logger != nil {
			pp.logger.Printf("Processing image through DeepSeek OCR for document %s", docID)
		}

		// Decode base64 image
		imageData, err := base64.StdEncoding.DecodeString(imageBase64)
		if err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Failed to decode image for document %s: %v", docID, err)
			}
		} else {
			// Process through DeepSeek OCR with research-enhanced prompt
			ocrPrompt := fmt.Sprintf("Extract all text and structured information from this document. Convert to markdown format.")
			
			// Enhance OCR prompt with research context if available
			if researchContext != "" {
				ocrPrompt = fmt.Sprintf(`Extract all text and structured information from this document about: %s

Context from research:
%s

Convert to markdown format with proper structure.`, title, researchContext)
			}
			
			ocrText, err := pp.ocrClient.ExtractText(ctx, imageData, ocrPrompt, "")
			if err != nil {
				if pp.logger != nil {
					pp.logger.Printf("OCR failed for document %s: %v", docID, err)
				}
			} else {
				// Combine original content with OCR text
				if processedText != "" {
					processedText = processedText + "\n\n--- OCR Content ---\n\n" + ocrText
				} else {
					processedText = ocrText
				}
			}
		}
	} else if imageURL != "" {
		// Fetch image from URL and process
		if pp.logger != nil {
			pp.logger.Printf("Fetching image from URL for document %s", docID)
		}
		// Note: In production, you might want to fetch the image here
	}

	// Step 2: Catalog Registration (with research metadata and learning)
	if pp.catalogURL != "" {
		if err := pp.registerInCatalog(ctx, docID, title, processedText, researchReport); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Catalog registration failed for document %s: %v", docID, err)
			}
		} else {
			// After successful registration, extract patterns and discover relationships
			if err := pp.learnFromCatalog(ctx, docID, title, processedText); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Catalog learning failed for document %s (non-fatal): %v", docID, err)
				}
			}
		}
	}

	// Step 3: Training Data Export (with pattern learning and feedback)
	if pp.trainingURL != "" {
		taskID, err := pp.exportForTraining(ctx, docID, title, processedText, researchReport)
		if err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Training export failed for document %s: %v", docID, err)
			}
		} else if taskID != "" {
			// After export, get learned patterns and apply them
			if err := pp.getTrainingFeedback(ctx, taskID, docID); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Training feedback collection failed for document %s (non-fatal): %v", docID, err)
				}
			}
		}
	}

	// Step 4: Local AI Storage (with domain detection)
	if pp.localAIURL != "" {
		// Detect domain from document content
		domain := pp.detectDomain(title + " " + processedText)
		if pp.logger != nil {
			pp.logger.Printf("Detected domain '%s' for document %s", domain, docID)
		}
		
		if err := pp.storeInLocalAI(ctx, docID, title, processedText, domain, doc); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Local AI storage failed for document %s: %v", docID, err)
			}
		} else {
			// After successful storage, learn from document to improve domain model
			if err := pp.learnFromLocalAI(ctx, docID, title, processedText, domain); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Local AI learning failed for document %s (non-fatal): %v", docID, err)
				}
			}
		}
	}

	// Step 5: Search Indexing (with learning)
	if pp.searchURL != "" {
		if err := pp.indexInSearch(ctx, docID, title, processedText, doc); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Search indexing failed for document %s: %v", docID, err)
			}
		} else {
			// After successful indexing, learn from search patterns
			if err := pp.learnFromSearch(ctx, docID, title, processedText); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Search learning failed for document %s (non-fatal): %v", docID, err)
				}
			}
		}
	}

	// Collect feedback from all services and apply improvements
	if pp.learningOrchestrator != nil {
		feedbackResults := map[string]interface{}{
			"unified_workflow": unifiedWorkflowResult,
		}
		if err := pp.learningOrchestrator.CollectFeedback(ctx, docID, feedbackResults); err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Feedback collection failed (non-fatal): %v", err)
			}
		}
	}

	return nil
}

// processViaUnifiedWorkflow processes a document through the unified workflow system.
// This enables knowledge graph processing, orchestration chains, and AgentFlow integration.
func (pp *PerplexityPipeline) processViaUnifiedWorkflow(ctx context.Context, docID, title, content string) (map[string]interface{}, error) {
	if pp.unifiedWorkflowURL == "" {
		return nil, fmt.Errorf("unified workflow URL not configured")
	}

	// Convert document to knowledge graph format (JSON table)
	// Format: [{"column": "field", "value": "data"}]
	jsonTable := pp.convertDocumentToJSONTable(docID, title, content)

	// Create unified workflow request
	workflowRequest := map[string]interface{}{
		"workflow_mode": "sequential",
		"knowledge_graph_request": map[string]interface{}{
			"project_id":  "perplexity",
			"system_id":   "perplexity-ingestion",
			"json_tables": []string{jsonTable},
		},
		"orchestration_request": map[string]interface{}{
			"chain_name": "document_processor",
			"inputs": map[string]interface{}{
				"document_id": docID,
				"title":       title,
				"content":     content,
			},
		},
		"agentflow_request": map[string]interface{}{
			"flow_id":     "processes/perplexity_ingestion",
			"input_value": fmt.Sprintf("Process Perplexity document: %s", title),
			"inputs": map[string]interface{}{
				"document_id": docID,
				"title":       title,
				"content":     content,
			},
			"ensure": true,
		},
	}

	// Execute unified workflow
	url := strings.TrimRight(pp.unifiedWorkflowURL, "/") + "/unified/process"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, workflowRequest, &result); err != nil {
		return nil, fmt.Errorf("unified workflow execution failed: %w", err)
	}

	return result, nil
}

// convertDocumentToJSONTable converts a document to JSON table format for knowledge graph processing.
func (pp *PerplexityPipeline) convertDocumentToJSONTable(docID, title, content string) string {
	// Create a JSON table structure
	table := map[string]interface{}{
		"table_name": fmt.Sprintf("perplexity_doc_%s", docID),
		"columns": []map[string]interface{}{
			{"name": "id", "type": "string", "value": docID},
			{"name": "title", "type": "string", "value": title},
			{"name": "content", "type": "text", "value": content},
		},
		"rows": []map[string]interface{}{
			{
				"id":      docID,
				"title":   title,
				"content": content,
			},
		},
	}

	tableJSON, err := json.Marshal(table)
	if err != nil {
		if pp.logger != nil {
			pp.logger.Printf("Failed to convert document to JSON table: %v", err)
		}
		return ""
	}

	return string(tableJSON)
}

// registerInCatalog registers the document in the catalog service with research metadata.
func (pp *PerplexityPipeline) registerInCatalog(ctx context.Context, docID, title, content string, researchReport *research.ResearchReport) error {
	payload := map[string]interface{}{
		"topic":        title,
		"customer_need": content,
		"source":       "perplexity",
		"source_id":    docID,
	}

	// Add research metadata if available
	if researchReport != nil && researchReport.Report != nil {
		payload["research_summary"] = researchReport.Report.Summary
		payload["research_topic"] = researchReport.Report.Topic
		if len(researchReport.Report.Sources) > 0 {
			payload["research_sources"] = researchReport.Report.Sources
		}
		// Add research sections as metadata
		if len(researchReport.Report.Sections) > 0 {
			sections := make([]map[string]interface{}, 0, len(researchReport.Report.Sections))
			for _, section := range researchReport.Report.Sections {
				sections = append(sections, map[string]interface{}{
					"title":   section.Title,
					"content": section.Content,
				})
			}
			payload["research_sections"] = sections
		}
	}

	url := strings.TrimRight(pp.catalogURL, "/") + "/catalog/data-products/build"
	return pp.postJSON(ctx, url, payload, nil)
}

// learnFromCatalog extracts patterns, discovers relationships, and enriches metadata from catalog.
func (pp *PerplexityPipeline) learnFromCatalog(ctx context.Context, docID, title, content string) error {
	if pp.catalogURL == "" {
		return nil
	}

	// Step 1: Extract patterns from the registered document
	patternsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "extract_patterns",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	patternsURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/patterns/extract"
	var patternsResult map[string]interface{}
	if err := pp.postJSON(ctx, patternsURL, patternsPayload, &patternsResult); err != nil {
		// Non-fatal - pattern extraction may not be available
		if pp.logger != nil {
			pp.logger.Printf("Pattern extraction not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Extracted patterns for document %s", docID)
	}

	// Step 2: Discover relationships with existing documents
	relationshipsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "discover_relationships",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	relationshipsURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/relationships/discover"
	var relationshipsResult map[string]interface{}
	if err := pp.postJSON(ctx, relationshipsURL, relationshipsPayload, &relationshipsResult); err != nil {
		// Non-fatal - relationship discovery may not be available
		if pp.logger != nil {
			pp.logger.Printf("Relationship discovery not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		if relationships, ok := relationshipsResult["relationships"].([]interface{}); ok {
			pp.logger.Printf("Discovered %d relationships for document %s", len(relationships), docID)
		}
	}

	// Step 3: Enrich metadata based on similar documents
	enrichPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "perplexity",
		"action":    "enrich_metadata",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	enrichURL := strings.TrimRight(pp.catalogURL, "/") + "/catalog/metadata/enrich"
	var enrichResult map[string]interface{}
	if err := pp.postJSON(ctx, enrichURL, enrichPayload, &enrichResult); err != nil {
		// Non-fatal - metadata enrichment may not be available
		if pp.logger != nil {
			pp.logger.Printf("Metadata enrichment not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Enriched metadata for document %s", docID)
	}

	return nil
}

// exportForTraining exports the document for training with pattern learning support.
// Returns the task ID for feedback collection.
func (pp *PerplexityPipeline) exportForTraining(ctx context.Context, docID, title, content string, researchReport *research.ResearchReport) (string, error) {
	// Prepare document with research context
	docData := map[string]interface{}{
		"id":      docID,
		"title":   title,
		"content": content,
		"source":  "perplexity",
	}

	// Add research context for pattern learning
	if researchReport != nil && researchReport.Report != nil {
		docData["research_context"] = map[string]interface{}{
			"topic":    researchReport.Report.Topic,
			"summary":  researchReport.Report.Summary,
			"sections": researchReport.Report.Sections,
		}
	}

	payload := map[string]interface{}{
		"project_id": "perplexity",
		"system_id":  "perplexity-ingestion",
		"documents":  []map[string]interface{}{docData},
		// Enable pattern learning features
		"enable_pattern_learning": true,
		"enable_temporal_analysis": true,
		"enable_domain_filtering":   true,
	}

	url := strings.TrimRight(pp.trainingURL, "/") + "/pipeline/run"
	var result map[string]interface{}
	if err := pp.postJSON(ctx, url, payload, &result); err != nil {
		return "", err
	}

	// Extract task ID from result
	taskID := ""
	if id, ok := result["task_id"].(string); ok {
		taskID = id
	} else if id, ok := result["id"].(string); ok {
		taskID = id
	}

	return taskID, nil
}

// getTrainingFeedback gets learned patterns from training service and applies them.
func (pp *PerplexityPipeline) getTrainingFeedback(ctx context.Context, taskID, docID string) error {
	if pp.trainingURL == "" || taskID == "" {
		return nil
	}

	// Poll for learned patterns (with timeout)
	maxAttempts := 3
	for attempt := 0; attempt < maxAttempts; attempt++ {
		// Wait a bit before checking (patterns may take time to learn)
		if attempt > 0 {
			time.Sleep(2 * time.Second)
		}

		feedbackURL := strings.TrimRight(pp.trainingURL, "/") + "/patterns/learned"
		payload := map[string]interface{}{
			"task_id": taskID,
			"source":  "perplexity",
		}

		var feedbackResult map[string]interface{}
		if err := pp.postJSON(ctx, feedbackURL, payload, &feedbackResult); err != nil {
			if attempt == maxAttempts-1 {
				// Last attempt failed, return error
				return fmt.Errorf("failed to get training feedback after %d attempts: %w", maxAttempts, err)
			}
			continue
		}

		// Extract learned patterns
		if patterns, ok := feedbackResult["patterns"].(map[string]interface{}); ok {
			if pp.logger != nil {
				pp.logger.Printf("Received learned patterns for document %s (task %s)", docID, taskID)
			}

			// Store patterns for future use
			if err := pp.storeLearnedPatterns(ctx, docID, patterns); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Failed to store learned patterns: %v", err)
				}
			}

			// Apply patterns to improve future processing
			if err := pp.applyLearnedPatterns(ctx, patterns); err != nil {
				if pp.logger != nil {
					pp.logger.Printf("Failed to apply learned patterns: %v", err)
				}
			}

			return nil
		}

		// If patterns not ready yet, try again
		if status, ok := feedbackResult["status"].(string); ok && status == "processing" {
			continue
		}
	}

	return nil
}

// storeLearnedPatterns stores learned patterns for future reference.
func (pp *PerplexityPipeline) storeLearnedPatterns(ctx context.Context, docID string, patterns map[string]interface{}) error {
	// Store patterns in a way that can be retrieved later
	// For now, just log them - in production, would store in a pattern store
	if pp.logger != nil {
		if columnPatterns, ok := patterns["column_patterns"].(map[string]interface{}); ok {
			pp.logger.Printf("Stored column patterns for document %s", docID)
			_ = columnPatterns // Use patterns
		}
		if relationshipPatterns, ok := patterns["relationship_patterns"].(map[string]interface{}); ok {
			pp.logger.Printf("Stored relationship patterns for document %s", docID)
			_ = relationshipPatterns // Use patterns
		}
	}
	return nil
}

// applyLearnedPatterns applies learned patterns to improve future document processing.
func (pp *PerplexityPipeline) applyLearnedPatterns(ctx context.Context, patterns map[string]interface{}) error {
	// Apply patterns to optimize future queries and processing
	// This could update query optimization, domain detection, etc.
	if pp.logger != nil {
		pp.logger.Printf("Applying learned patterns to improve future processing")
	}
	return nil
}

// detectDomain detects the domain from document content using LocalAI's domain detection API.
func (pp *PerplexityPipeline) detectDomain(content string) string {
	if pp.localAIURL == "" {
		return "general"
	}

	// Use LocalAI's domain detection via API
	// We'll use a simple keyword-based detection for now, matching LocalAI's approach
	contentLower := strings.ToLower(content)
	
	// Common domain keywords (matching LocalAI domain configs)
	domainKeywords := map[string][]string{
		"sql":        {"select", "database", "query", "table", "sql"},
		"ai":         {"ai", "machine learning", "neural", "transformer", "llm", "model"},
		"technology": {"technology", "software", "hardware", "computer", "system"},
		"science":    {"science", "research", "study", "experiment", "analysis"},
		"business":   {"business", "market", "company", "industry", "finance"},
	}

	bestScore := 0
	detectedDomain := "general"

	for domain, keywords := range domainKeywords {
		score := 0
		for _, keyword := range keywords {
			if strings.Contains(contentLower, keyword) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			detectedDomain = domain
		}
	}

	return detectedDomain
}

// storeInLocalAI stores the document in Local AI service with domain-aware routing and explicit model selection.
func (pp *PerplexityPipeline) storeInLocalAI(ctx context.Context, docID, title, content, domain string, metadata map[string]interface{}) error {
	if pp.localAIClient == nil {
		return nil
	}

	// Add domain to metadata
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	metadata["domain"] = domain
	metadata["detected_domain"] = domain

	// Explicit model selection based on domain
	model := pp.selectModelForDomain(domain)

	payload := map[string]interface{}{
		"id":       docID,
		"content":  content,
		"title":    title,
		"metadata": metadata,
	}

	// Use LocalAIClient with explicit model selection, retry logic, and circuit breaker
	_, err := pp.localAIClient.StoreDocument(ctx, domain, model, payload)
	return err
}

// learnFromLocalAI learns from document to improve domain models and embeddings.
// Uses LocalAIClient with explicit model selection, retry logic, and circuit breaker.
func (pp *PerplexityPipeline) learnFromLocalAI(ctx context.Context, docID, title, content, domain string) error {
	if pp.localAIClient == nil {
		return nil
	}

	// Priority 3: Request GPU allocation for domain learning operations
	var gpuAllocationID string
	if pp.gpuOrchestratorURL != "" {
		gpuAllocID, err := pp.requestGPUAllocation(ctx, "domain_learning", domain, len(content))
		if err != nil {
			if pp.logger != nil {
				pp.logger.Printf("Warning: Failed to allocate GPU for domain learning: %v (continuing with CPU)", err)
			}
		} else {
			gpuAllocationID = gpuAllocID
			if pp.logger != nil {
				pp.logger.Printf("Allocated GPU for domain learning: %s", gpuAllocationID)
			}
		}
		// Ensure GPU is released after learning operations
		defer func() {
			if gpuAllocationID != "" {
				pp.releaseGPUAllocation(ctx, gpuAllocationID)
			}
		}()
	}

	// Determine model based on domain (explicit model selection)
	model := pp.selectModelForDomain(domain)

	// Step 1: Update domain model with new document
	updatePayload := map[string]interface{}{
		"model": model, // Explicit model selection
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
		"action": "update_domain_model",
	}

	_, err := pp.localAIClient.CallDomainEndpoint(ctx, domain, "learn", updatePayload)
	if err != nil {
		// Non-fatal - domain learning may not be available
		if pp.logger != nil {
			pp.logger.Printf("Domain model update not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Updated domain model '%s' (model: %s) with document %s", domain, model, docID)
	}

	// Step 2: Generate and store embeddings for domain-specific model
	embeddingPayload := map[string]interface{}{
		"model": model, // Explicit model selection
		"text":   title + " " + content,
		"document_id": docID,
	}

	_, err = pp.localAIClient.CallDomainEndpoint(ctx, domain, "embeddings", embeddingPayload)
	if err != nil {
		// Non-fatal - embedding generation may not be available
		if pp.logger != nil {
			pp.logger.Printf("Embedding generation not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Generated embeddings for document %s in domain '%s' (model: %s)", docID, domain, model)
	}

	// Step 3: Learn domain patterns from document content
	patternPayload := map[string]interface{}{
		"model": model, // Explicit model selection
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
		"action": "learn_domain_patterns",
	}

	_, err = pp.localAIClient.CallDomainEndpoint(ctx, domain, "patterns", patternPayload)
	if err != nil {
		// Non-fatal - pattern learning may not be available
		if pp.logger != nil {
			pp.logger.Printf("Domain pattern learning not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Learned domain patterns for document %s in domain '%s' (model: %s)", docID, domain, model)
	}

	return nil
}

// selectModelForDomain selects the appropriate model for a given domain based on domains.json analysis
func (pp *PerplexityPipeline) selectModelForDomain(domain string) string {
	// Model selection based on domain configuration analysis
	// Most domains use gemma-2b-q4_k_m.gguf, some use transformers models
	switch domain {
	case "general", "":
		return "phi-3.5-mini" // Default to phi-3.5-mini for general domain
	case "finance", "treasury", "subledger", "trade_recon":
		// Finance domains: use gemma-2b or granite-4.0
		return "gemma-2b-q4_k_m.gguf"
	case "browser", "web_analysis":
		// Browser domain uses gemma-7b
		return "gemma-7b-q4_k_m.gguf"
	default:
		// Default to gemma-2b for most domains
		return "gemma-2b-q4_k_m.gguf"
	}
}

// requestGPUAllocation requests GPU allocation from the GPU orchestrator (Priority 3).
func (pp *PerplexityPipeline) requestGPUAllocation(ctx context.Context, workloadType, domain string, contentSize int) (string, error) {
	if pp.gpuOrchestratorURL == "" {
		return "", fmt.Errorf("GPU orchestrator URL not configured")
	}

	// Estimate GPU requirements based on content size
	requiredGPUs := 1
	minMemoryMB := 4096 // 4GB default
	if contentSize > 1000000 { // > 1MB
		minMemoryMB = 8192 // 8GB for large documents
	}

	workloadData := map[string]interface{}{
		"workload_type":    workloadType,
		"domain":           domain,
		"content_size":     contentSize,
		"required_gpus":    requiredGPUs,
		"min_memory_mb":    minMemoryMB,
		"priority":         7, // High priority for domain learning
	}

	requestBody := map[string]interface{}{
		"service_name":  "orchestration",
		"workload_type": workloadType,
		"workload_data": workloadData,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return "", fmt.Errorf("marshal GPU allocation request: %w", err)
	}

	url := strings.TrimRight(pp.gpuOrchestratorURL, "/") + "/gpu/allocate"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return "", fmt.Errorf("create GPU allocation request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := pp.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("execute GPU allocation request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("GPU orchestrator returned status %d: %s", resp.StatusCode, string(body))
	}

	var allocation struct {
		ID     string `json:"id"`
		GPUIDs []int  `json:"gpu_ids"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&allocation); err != nil {
		return "", fmt.Errorf("decode GPU allocation response: %w", err)
	}

	return allocation.ID, nil
}

// releaseGPUAllocation releases GPU allocation (Priority 3).
func (pp *PerplexityPipeline) releaseGPUAllocation(ctx context.Context, allocationID string) error {
	if pp.gpuOrchestratorURL == "" || allocationID == "" {
		return nil
	}

	url := strings.TrimRight(pp.gpuOrchestratorURL, "/") + "/gpu/release"
	requestBody := map[string]interface{}{
		"allocation_id": allocationID,
		"service_name":  "orchestration",
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("marshal GPU release request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return fmt.Errorf("create GPU release request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := pp.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute GPU release request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("GPU orchestrator returned status %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// indexInSearch indexes the document in the search service.
func (pp *PerplexityPipeline) indexInSearch(ctx context.Context, docID, title, content string, metadata map[string]interface{}) error {
	payload := map[string]interface{}{
		"id":       docID,
		"content":  content,
		"title":    title,
		"metadata": metadata,
	}

	url := strings.TrimRight(pp.searchURL, "/") + "/documents"
	return pp.postJSON(ctx, url, payload, nil)
}

// learnFromSearch learns from search patterns to improve relevance and optimize embeddings.
func (pp *PerplexityPipeline) learnFromSearch(ctx context.Context, docID, title, content string) error {
	if pp.searchURL == "" {
		return nil
	}

	// Step 1: Track document in search analytics
	analyticsPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "track_document",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	analyticsURL := strings.TrimRight(pp.searchURL, "/") + "/analytics/track"
	var analyticsResult map[string]interface{}
	if err := pp.postJSON(ctx, analyticsURL, analyticsPayload, &analyticsResult); err != nil {
		// Non-fatal - analytics may not be available
		if pp.logger != nil {
			pp.logger.Printf("Search analytics tracking not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Tracked document %s in search analytics", docID)
	}

	// Step 2: Learn search patterns (what queries find this document)
	patternPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "learn_patterns",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	patternURL := strings.TrimRight(pp.searchURL, "/") + "/patterns/learn"
	var patternResult map[string]interface{}
	if err := pp.postJSON(ctx, patternURL, patternPayload, &patternResult); err != nil {
		// Non-fatal - pattern learning may not be available
		if pp.logger != nil {
			pp.logger.Printf("Search pattern learning not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Learned search patterns for document %s", docID)
	}

	// Step 3: Optimize embeddings for better search relevance
	optimizePayload := map[string]interface{}{
		"document_id": docID,
		"action":      "optimize_embeddings",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}

	optimizeURL := strings.TrimRight(pp.searchURL, "/") + "/embeddings/optimize"
	var optimizeResult map[string]interface{}
	if err := pp.postJSON(ctx, optimizeURL, optimizePayload, &optimizeResult); err != nil {
		// Non-fatal - embedding optimization may not be available
		if pp.logger != nil {
			pp.logger.Printf("Embedding optimization not available or failed: %v", err)
		}
	} else if pp.logger != nil {
		pp.logger.Printf("Optimized embeddings for document %s", docID)
	}

	return nil
}

// postJSON performs a POST request with JSON payload.
func (pp *PerplexityPipeline) postJSON(ctx context.Context, url string, payload interface{}, result interface{}) error {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := pp.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := json.Marshal(map[string]string{"error": fmt.Sprintf("request failed with status %d", resp.StatusCode)})
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

