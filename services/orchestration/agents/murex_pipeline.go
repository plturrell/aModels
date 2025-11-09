package agents

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// MurexPipeline orchestrates the full Murex data processing pipeline
// from API connection through catalog, training, local AI, search, and ETL to SAP.
type MurexPipeline struct {
	murexConnector       SourceConnector
	unifiedWorkflowURL   string
	catalogURL           string
	trainingURL          string
	localAIURL           string
	searchURL            string
	sapGLURL             string // SAP GL endpoint for ETL
	learningOrchestrator *LearningOrchestrator
	requestTracker       *RequestTracker
	logger               *log.Logger
	httpClient           *http.Client
	localAIClient        *LocalAIClient // Standardized LocalAI client with retry and validation
}

// MurexPipelineConfig configures the pipeline.
type MurexPipelineConfig struct {
	BaseURL            string // Murex API base URL
	APIKey             string // Murex API key
	OpenAPISpecURL     string // Optional OpenAPI spec URL
	UnifiedWorkflowURL string
	CatalogURL         string
	TrainingURL        string
	LocalAIURL         string
	SearchURL          string
	SAPGLURL           string // SAP GL endpoint for ETL
	Logger             *log.Logger
}

// NewMurexPipeline creates a new Murex processing pipeline.
func NewMurexPipeline(config MurexPipelineConfig) (*MurexPipeline, error) {
	// Create Murex connector
	murexConfig := map[string]interface{}{
		"base_url":         config.BaseURL,
		"api_key":          config.APIKey,
		"openapi_spec_url": config.OpenAPISpecURL,
		"allow_mock_data":  false, // Production mode - no mocks
	}
	murexConnector := connectors.NewMurexConnector(murexConfig, config.Logger)

	// Use connection pooling for better performance (Priority 1)
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
	}
	
	pipeline := &MurexPipeline{
		murexConnector:     murexConnector,
		unifiedWorkflowURL: config.UnifiedWorkflowURL,
		catalogURL:         config.CatalogURL,
		trainingURL:        config.TrainingURL,
		localAIURL:         config.LocalAIURL,
		searchURL:          config.SearchURL,
		sapGLURL:           config.SAPGLURL,
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

// GetRequestTracker returns the request tracker.
func (mp *MurexPipeline) GetRequestTracker() *RequestTracker {
	return mp.requestTracker
}

// GetMurexConnector returns the Murex connector.
func (mp *MurexPipeline) GetMurexConnector() SourceConnector {
	return mp.murexConnector
}

// ProcessTradesWithTracking processes Murex trades with full request tracking.
func (mp *MurexPipeline) ProcessTradesWithTracking(ctx context.Context, requestID string, query map[string]interface{}) (*ProcessingRequest, error) {
	// Build query string
	queryStr := "Murex trade processing"
	if table, ok := query["table"].(string); ok && table != "" {
		queryStr = fmt.Sprintf("Processing Murex %s", table)
	} else if filters, ok := query["filters"].(map[string]interface{}); ok && len(filters) > 0 {
		queryStr = "Processing Murex trades with filters"
	}

	// Create request tracking
	request := mp.requestTracker.CreateRequest(requestID, queryStr)
	mp.requestTracker.UpdateStatus(requestID, RequestStatusProcessing)
	mp.requestTracker.UpdateStep(requestID, "connecting")

	// Apply learned improvements before processing
	if mp.learningOrchestrator != nil {
		if err := mp.learningOrchestrator.ApplyImprovements(ctx, query); err != nil {
			if mp.logger != nil {
				mp.logger.Printf("Failed to apply improvements (non-fatal): %v", err)
			}
		}
	}

	// Step 1: Connect to Murex
	mp.requestTracker.UpdateStep(requestID, "connecting")
	if err := mp.murexConnector.Connect(ctx, query); err != nil {
		mp.requestTracker.AddErrorWithDetails(requestID, "connect", err.Error(), "", "CONNECTION_ERROR", nil, true)
		mp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mp.murexConnector.Close()

	// Process trades
	mp.processTradesWithIntelligence(ctx, requestID, query, request)

	return request, nil
}

// processTradesWithIntelligence processes trades and collects intelligence.
func (mp *MurexPipeline) processTradesWithIntelligence(ctx context.Context, requestID string, query map[string]interface{}, request *ProcessingRequest) {
	// Extract table/entity name
	tableName, _ := query["table"].(string)
	if tableName == "" {
		tableName = "trades" // Default to trades
	}

	// Extract data from Murex
	mp.requestTracker.UpdateStep(requestID, "extracting")
	extractQuery := map[string]interface{}{
		"table": tableName,
		"limit": 1000, // Default limit
	}
	
	// Add filters if provided
	if filters, ok := query["filters"].(map[string]interface{}); ok {
		for k, v := range filters {
			extractQuery[k] = v
		}
	}

	data, err := mp.murexConnector.ExtractData(ctx, extractQuery)
	if err != nil {
		mp.requestTracker.AddErrorWithDetails(requestID, "extract", err.Error(), "", "EXTRACTION_ERROR", nil, true)
		mp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return
	}

	// Update statistics
	mp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
		DocumentsTotal: len(data),
	})

	// Process each record
	mp.requestTracker.UpdateStep(requestID, "processing")
	documents := []ProcessedDocument{}
	
	for i, record := range data {
		docID := fmt.Sprintf("murex_%s_%d", tableName, i)
		
		doc := ProcessedDocument{
			ID:          docID,
			Title:      fmt.Sprintf("Murex %s: %v", tableName, getRecordID(record)),
			Status:     "processing",
			ProcessedAt: time.Now().UTC().Format(time.RFC3339),
		}

		mp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s", docID))

		// Convert record to document format
		docMap := map[string]interface{}{
			"id":      docID,
			"name":    tableName,
			"title":   doc.Title,
			"content": record,
			"metadata": map[string]interface{}{
				"table":      tableName,
				"source":     "murex",
				"record_data": record,
			},
		}

		// Process through pipeline
		if err := mp.processTradeWithTracking(ctx, requestID, docMap, &doc, tableName); err != nil {
			doc.Status = "failed"
			doc.Error = err.Error()
			mp.requestTracker.AddErrorWithDetails(requestID, "process_trade", err.Error(), docID, "PROCESSING_ERROR", nil, false)
		} else {
			doc.Status = "succeeded"
			mp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
				DocumentsSucceeded: 1,
			})
		}

		documents = append(documents, doc)
		mp.requestTracker.AddDocument(requestID, doc)
	}

	// Collect intelligence
	intelligence := mp.collectIntelligence(ctx, documents, requestID)
	mp.requestTracker.StoreIntelligence(requestID, intelligence)

	// Complete request
	mp.requestTracker.UpdateStep(requestID, "completed")
	mp.requestTracker.UpdateStatus(requestID, RequestStatusCompleted)
}

// getRecordID extracts ID from record.
func getRecordID(record map[string]interface{}) interface{} {
	if id, ok := record["id"]; ok {
		return id
	}
	if tradeID, ok := record["trade_id"]; ok {
		return tradeID
	}
	return "unknown"
}

// processTradeWithTracking processes a single trade through the full pipeline with tracking.
func (mp *MurexPipeline) processTradeWithTracking(ctx context.Context, requestID string, docMap map[string]interface{}, doc *ProcessedDocument, tableName string) error {
	// Step 1: Catalog registration
	if err := mp.registerInCatalog(ctx, doc, docMap, tableName); err != nil {
		return fmt.Errorf("catalog registration failed: %w", err)
	}

	// Step 2: Training export
	if err := mp.exportForTraining(ctx, doc, docMap, tableName); err != nil {
		mp.logger.Printf("Warning: Training export failed (non-fatal): %v", err)
	}

	// Step 3: LocalAI storage
	if err := mp.storeInLocalAI(ctx, doc, docMap, tableName); err != nil {
		mp.logger.Printf("Warning: LocalAI storage failed (non-fatal): %v", err)
	}

	// Step 4: Search indexing
	if err := mp.indexInSearch(ctx, doc, docMap, tableName); err != nil {
		mp.logger.Printf("Warning: Search indexing failed (non-fatal): %v", err)
	}

	// Step 5: ETL to SAP GL
	if err := mp.etlToSAPGL(ctx, doc, docMap, tableName); err != nil {
		mp.logger.Printf("Warning: SAP GL ETL failed (non-fatal): %v", err)
	}

	return nil
}

// registerInCatalog registers trade in catalog.
func (mp *MurexPipeline) registerInCatalog(ctx context.Context, doc *ProcessedDocument, docMap map[string]interface{}, tableName string) error {
	payload := map[string]interface{}{
		"source":        "murex",
		"source_id":     doc.ID,
		"title":         doc.Title,
		"content":       docMap["content"],
		"metadata": map[string]interface{}{
			"table":      tableName,
			"source_type": "murex",
			"processed_at": doc.ProcessedAt,
		},
	}

	resp, err := mp.postJSON(ctx, mp.catalogURL+"/api/catalog/documents", payload)
	if err != nil {
		return err
	}

	if catalogID, ok := resp["catalog_id"].(string); ok {
		doc.CatalogID = catalogID
	}

	return nil
}

// exportForTraining exports trade for training.
func (mp *MurexPipeline) exportForTraining(ctx context.Context, doc *ProcessedDocument, docMap map[string]interface{}, tableName string) error {
	payload := map[string]interface{}{
		"source":    "murex",
		"source_id": doc.ID,
		"data":      docMap["content"],
		"metadata": map[string]interface{}{
			"table": tableName,
		},
	}

	_, err := mp.postJSON(ctx, mp.trainingURL+"/api/training/export", payload)
	return err
}

// storeInLocalAI stores trade in LocalAI with explicit model selection.
func (mp *MurexPipeline) storeInLocalAI(ctx context.Context, doc *ProcessedDocument, docMap map[string]interface{}, tableName string) error {
	if mp.localAIClient == nil {
		return nil
	}

	domain := mp.detectDomain(docMap)
	// Explicit model selection: use gemma-2b for finance domain (from domains.json analysis)
	model := "gemma-2b-q4_k_m.gguf"
	if domain == "finance" || domain == "general" {
		// Finance domain typically uses gemma-2b or granite-4.0
		// Default to gemma-2b for Murex trades
		model = "gemma-2b-q4_k_m.gguf"
	}
	
	payload := map[string]interface{}{
		"source":    "murex",
		"source_id": doc.ID,
		"content":   docMap["content"],
		"metadata": map[string]interface{}{
			"table": tableName,
		},
	}

	// Use LocalAIClient with explicit model selection, retry logic, and circuit breaker
	resp, err := mp.localAIClient.StoreDocument(ctx, domain, model, payload)
	if err != nil {
		return err
	}

	if localAIID, ok := resp["local_ai_id"].(string); ok {
		doc.LocalAIID = localAIID
	}

	return nil
}

// indexInSearch indexes trade in search.
func (mp *MurexPipeline) indexInSearch(ctx context.Context, doc *ProcessedDocument, docMap map[string]interface{}, tableName string) error {
	payload := map[string]interface{}{
		"source":    "murex",
		"source_id": doc.ID,
		"content":   docMap["content"],
		"metadata": map[string]interface{}{
			"table": tableName,
		},
	}

	resp, err := mp.postJSON(ctx, mp.searchURL+"/api/search/index", payload)
	if err != nil {
		return err
	}

	if searchID, ok := resp["search_id"].(string); ok {
		doc.SearchID = searchID
	}

	return nil
}

// etlToSAPGL performs ETL transformation and loads to SAP GL.
func (mp *MurexPipeline) etlToSAPGL(ctx context.Context, doc *ProcessedDocument, docMap map[string]interface{}, tableName string) error {
	if mp.sapGLURL == "" {
		return nil // SAP GL URL not configured, skip ETL
	}

	// Transform Murex trade to SAP GL journal entry format
	record, _ := docMap["content"].(map[string]interface{})
	sapEntry := mp.transformToSAPGL(record, tableName)

	payload := map[string]interface{}{
		"source":    "murex",
		"source_id": doc.ID,
		"entry":     sapEntry,
		"metadata": map[string]interface{}{
			"table":      tableName,
			"transformed_at": time.Now().Format(time.RFC3339),
		},
	}

	_, err := mp.postJSON(ctx, mp.sapGLURL+"/api/sap-gl/journal-entries", payload)
	return err
}

// transformToSAPGL transforms Murex trade data to SAP GL journal entry format.
func (mp *MurexPipeline) transformToSAPGL(record map[string]interface{}, tableName string) map[string]interface{} {
	entry := map[string]interface{}{}

	// Map trade_id to entry_id
	if tradeID, ok := record["trade_id"].(string); ok {
		entry["entry_id"] = fmt.Sprintf("JE-%s", tradeID)
	} else if id, ok := record["id"].(string); ok {
		entry["entry_id"] = fmt.Sprintf("JE-%s", id)
	}

	// Map trade_date to entry_date
	if tradeDate, ok := record["trade_date"].(string); ok {
		entry["entry_date"] = tradeDate
	} else if date, ok := record["date"].(string); ok {
		entry["entry_date"] = date
	}

	// Map notional_amount to debit_amount
	if notional, ok := record["notional_amount"].(float64); ok {
		entry["debit_amount"] = notional
		entry["credit_amount"] = notional // Single-sided entry
	} else if amount, ok := record["amount"].(float64); ok {
		entry["debit_amount"] = amount
		entry["credit_amount"] = amount
	}

	// Map currency
	if currency, ok := record["currency"].(string); ok {
		entry["currency"] = currency
	}

	// Map counterparty_id to account (via lookup if available)
	if counterparty, ok := record["counterparty_id"].(string); ok {
		entry["account"] = counterparty // In production, would lookup account mapping
		entry["counterparty_id"] = counterparty
	}

	// Add description
	entry["description"] = fmt.Sprintf("Murex %s entry", tableName)

	return entry
}

// detectDomain detects domain from trade data.
func (mp *MurexPipeline) detectDomain(docMap map[string]interface{}) string {
	// Finance domain for Murex trades
	if metadata, ok := docMap["metadata"].(map[string]interface{}); ok {
		if tableName, ok := metadata["table"].(string); ok {
			switch tableName {
			case "trades", "transactions":
				return "finance"
			case "cashflows":
				return "finance"
			case "positions":
				return "finance"
			}
		}
	}
	return "finance" // Default to finance domain
}

// collectIntelligence collects intelligence from processed documents.
func (mp *MurexPipeline) collectIntelligence(ctx context.Context, documents []ProcessedDocument, requestID string) *RequestIntelligence {
	intelligence := &RequestIntelligence{
		Domains:              []string{},
		TotalRelationships:   0,
		TotalPatterns:        0,
		KnowledgeGraphNodes:  0,
		KnowledgeGraphEdges:  0,
		WorkflowProcessed:    true,
	}

	// Collect domains (default to finance for Murex)
	intelligence.Domains = []string{"finance"}

	// Aggregate relationships and patterns (would be populated from intelligence collection)
	// For now, set defaults
	intelligence.TotalRelationships = len(documents) // Estimate
	intelligence.TotalPatterns = len(documents)      // Estimate

	return intelligence
}

// postJSON posts JSON data to a URL.
func (mp *MurexPipeline) postJSON(ctx context.Context, url string, payload map[string]interface{}) (map[string]interface{}, error) {
	jsonData, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal JSON: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(jsonData)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := mp.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result, nil
}

// Helper functions
func countSucceeded(docs []ProcessedDocument) int {
	count := 0
	for _, doc := range docs {
		if doc.Status == "succeeded" {
			count++
		}
	}
	return count
}

func countFailed(docs []ProcessedDocument) int {
	count := 0
	for _, doc := range docs {
		if doc.Status == "failed" {
			count++
		}
	}
	return count
}

