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

// RelationalPipeline orchestrates the full relational table processing pipeline
// from database connection through catalog, training, local AI, and search.
type RelationalPipeline struct {
	relationalConnector  SourceConnector
	unifiedWorkflowURL   string
	catalogURL           string
	trainingURL          string
	localAIURL           string
	searchURL            string
	extractURL           string
	learningOrchestrator *LearningOrchestrator
	requestTracker       *RequestTracker
	logger               *log.Logger
	httpClient           *http.Client
	localAIClient        *LocalAIClient // Standardized LocalAI client with retry and validation
}

// RelationalPipelineConfig configures the pipeline.
type RelationalPipelineConfig struct {
	DatabaseURL          string // Connection string or DSN
	DatabaseType         string // "postgres", "mysql", "sqlite"
	UnifiedWorkflowURL   string
	CatalogURL           string
	TrainingURL          string
	LocalAIURL           string
	SearchURL            string
	ExtractURL            string
	Logger               *log.Logger
}

// NewRelationalPipeline creates a new relational processing pipeline.
func NewRelationalPipeline(config RelationalPipelineConfig) (*RelationalPipeline, error) {
	// Create relational connector
	relationalConfig := map[string]interface{}{
		"database_url": config.DatabaseURL,
		"db_type":      config.DatabaseType,
		"database_type": config.DatabaseType,
	}
	relationalConnector := connectors.NewRelationalConnector(relationalConfig, config.Logger)

	pipeline := &RelationalPipeline{
		relationalConnector: relationalConnector,
		unifiedWorkflowURL:  config.UnifiedWorkflowURL,
		catalogURL:         config.CatalogURL,
		trainingURL:        config.TrainingURL,
		localAIURL:         config.LocalAIURL,
		searchURL:          config.SearchURL,
		extractURL:         config.ExtractURL,
		logger:             config.Logger,
		// Use connection pooling for better performance (Priority 1)
		httpClient: &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				MaxConnsPerHost:     50,
			},
			Timeout: 120 * time.Second,
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
func (rp *RelationalPipeline) GetRequestTracker() *RequestTracker {
	return rp.requestTracker
}

// GetRelationalConnector returns the relational connector.
func (rp *RelationalPipeline) GetRelationalConnector() SourceConnector {
	return rp.relationalConnector
}

// ProcessTablesWithTracking processes tables with full request tracking.
func (rp *RelationalPipeline) ProcessTablesWithTracking(ctx context.Context, requestID string, query map[string]interface{}) (*ProcessingRequest, error) {
	// Extract table names or schema from request
	tables, _ := query["tables"].([]interface{})
	table, _ := query["table"].(string)
	schema, _ := query["schema"].(string)

	queryStr := "Relational table processing"
	if table != "" {
		queryStr = fmt.Sprintf("Processing table %s", table)
	} else if len(tables) > 0 {
		queryStr = fmt.Sprintf("Processing %d tables", len(tables))
	} else if schema != "" {
		queryStr = fmt.Sprintf("Processing schema %s", schema)
	}

	// Create request tracking
	request := rp.requestTracker.CreateRequest(requestID, queryStr)
	rp.requestTracker.UpdateStatus(requestID, RequestStatusProcessing)
	rp.requestTracker.UpdateStep(requestID, "connecting")

	// Apply learned improvements before processing
	if rp.learningOrchestrator != nil {
		if err := rp.learningOrchestrator.ApplyImprovements(ctx, query); err != nil {
			if rp.logger != nil {
				rp.logger.Printf("Failed to apply improvements (non-fatal): %v", err)
			}
		}
	}

	// Step 1: Connect to database
	rp.requestTracker.UpdateStep(requestID, "connecting")
	if err := rp.relationalConnector.Connect(ctx, query); err != nil {
		rp.requestTracker.AddErrorWithDetails(requestID, "connect", err.Error(), "", "CONNECTION_ERROR", nil, true)
		rp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("failed to connect to database: %w", err)
	}
	defer rp.relationalConnector.Close()

	// Step 2: Discover schema if needed
	var tableList []string
	if table != "" {
		tableList = []string{table}
	} else if len(tables) > 0 {
		tableList = make([]string, len(tables))
		for i, t := range tables {
			if tStr, ok := t.(string); ok {
				tableList[i] = tStr
			}
		}
	} else {
		// Discover schema to get all tables
		rp.requestTracker.UpdateStep(requestID, "discovering_schema")
		schemaInfo, err := rp.relationalConnector.DiscoverSchema(ctx)
		if err != nil {
			rp.requestTracker.AddErrorWithDetails(requestID, "discover_schema", err.Error(), "", "SCHEMA_DISCOVERY_ERROR", nil, true)
			rp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
			return request, fmt.Errorf("failed to discover schema: %w", err)
		}
		for _, tableDef := range schemaInfo.Tables {
			tableList = append(tableList, tableDef.Name)
		}
	}

	if len(tableList) == 0 {
		rp.requestTracker.AddErrorWithDetails(requestID, "extract", "no tables found", "", "NO_TABLES", nil, false)
		rp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("no tables found")
	}

	rp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
		DocumentsTotal: len(tableList),
	})

	// Step 3: Process each table
	rp.requestTracker.UpdateStep(requestID, "processing")
	processedTables := make([]ProcessedDocument, 0, len(tableList))
	hasErrors := false

	for i, tableName := range tableList {
		tableID := fmt.Sprintf("%s.%s", schema, tableName)
		if schema == "" {
			tableID = tableName
		}

		tableResult := ProcessedDocument{
			ID:          tableID,
			Status:      "processing",
			ProcessedAt: time.Now().UTC().Format(time.RFC3339),
		}

		rp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s", tableID))

		// Extract table data
		extractQuery := map[string]interface{}{
			"table":  tableName,
			"schema": schema,
			"limit":  1000, // Process first 1000 rows
		}

		tableData, err := rp.relationalConnector.ExtractData(ctx, extractQuery)
		if err != nil {
			tableResult.Status = "failed"
			tableResult.Error = err.Error()
			hasErrors = true
			rp.requestTracker.AddErrorWithDetails(requestID, "extract_table", err.Error(), tableID, "EXTRACTION_ERROR", nil, false)
			processedTables = append(processedTables, tableResult)
			rp.requestTracker.AddDocument(requestID, tableResult)
			continue
		}

		// Convert table data to document format
		doc := map[string]interface{}{
			"id":      tableID,
			"name":    tableName,
			"title":   fmt.Sprintf("Table: %s", tableName),
			"content": rp.tableDataToText(tableName, tableData),
			"metadata": map[string]interface{}{
				"table":      tableName,
				"schema":     schema,
				"row_count":  len(tableData),
				"source":     "relational",
				"table_data": tableData,
			},
		}

		// Process table with intelligence collection
		if err := rp.processTableWithTracking(ctx, requestID, doc, &tableResult); err != nil {
			tableResult.Status = "failed"
			tableResult.Error = err.Error()
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
			rp.requestTracker.AddErrorWithDetails(requestID, "process_table", err.Error(), tableID, errorCode, nil, false)
		} else {
			tableResult.Status = "succeeded"
			rp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
				DocumentsSucceeded: 1,
			})
		}

		processedTables = append(processedTables, tableResult)
		rp.requestTracker.AddDocument(requestID, tableResult)
	}

	// Set results links
	results := &ProcessingResults{
		CatalogURL: fmt.Sprintf("/api/catalog/documents?source=relational&request_id=%s", requestID),
		SearchURL:  fmt.Sprintf("/api/search?query=relational&request_id=%s", requestID),
		ExportURL:  fmt.Sprintf("/api/relational/results/%s/export", requestID),
	}
	rp.requestTracker.SetResults(requestID, results)

	// Update final status
	rp.requestTracker.UpdateStep(requestID, "completed")
	if hasErrors && len(processedTables) > 0 {
		rp.requestTracker.UpdateStatus(requestID, RequestStatusPartial)
	} else if hasErrors {
		rp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
	} else {
		rp.requestTracker.UpdateStatus(requestID, RequestStatusCompleted)
	}

	return request, nil
}

// tableDataToText converts table data to a text representation for processing.
func (rp *RelationalPipeline) tableDataToText(tableName string, tableData []map[string]interface{}) string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("# Table: %s\n\n", tableName))
	sb.WriteString(fmt.Sprintf("This table contains %d rows.\n\n", len(tableData)))

	if len(tableData) > 0 {
		sb.WriteString("## Sample Data\n\n")
		// Convert first few rows to markdown table
		maxRows := 10
		if len(tableData) < maxRows {
			maxRows = len(tableData)
		}

		// Get column names from first row
		if len(tableData) > 0 {
			columns := make([]string, 0)
			for col := range tableData[0] {
				if !strings.HasPrefix(col, "_") { // Skip metadata columns
					columns = append(columns, col)
				}
			}

			// Write header
			sb.WriteString("| " + strings.Join(columns, " | ") + " |\n")
			sb.WriteString("|" + strings.Repeat(" --- |", len(columns)) + "\n")

			// Write rows
			for i := 0; i < maxRows; i++ {
				row := make([]string, len(columns))
				for j, col := range columns {
					val := tableData[i][col]
					if val == nil {
						row[j] = "NULL"
					} else {
						row[j] = fmt.Sprintf("%v", val)
					}
				}
				sb.WriteString("| " + strings.Join(row, " | ") + " |\n")
			}

			if len(tableData) > maxRows {
				sb.WriteString(fmt.Sprintf("\n... and %d more rows\n", len(tableData)-maxRows))
			}
		}
	}

	return sb.String()
}

// processTableWithTracking processes a single table and updates the table result.
func (rp *RelationalPipeline) processTableWithTracking(ctx context.Context, requestID string, doc map[string]interface{}, tableResult *ProcessedDocument) error {
	tableID := tableResult.ID

	// Track steps
	rp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s_unified_workflow", tableID))

	// Process table and capture intelligence
	intelligence, err := rp.processTableWithIntelligence(ctx, doc)
	if err != nil {
		return err
	}

	// Extract IDs from table processing
	tableResult.CatalogID = tableID
	tableResult.LocalAIID = tableID
	tableResult.SearchID = tableID

	// Add metadata
	tableResult.Metadata = map[string]interface{}{
		"source":       "relational",
		"processed_at": tableResult.ProcessedAt,
	}

	// Store intelligence data
	if intelligence != nil {
		tableResult.Intelligence = intelligence
		rp.requestTracker.SetDocumentIntelligence(requestID, tableID, intelligence)
	}

	return nil
}

// processTableWithIntelligence processes a table and returns intelligence data.
func (rp *RelationalPipeline) processTableWithIntelligence(ctx context.Context, doc map[string]interface{}) (*DocumentIntelligence, error) {
	tableID, _ := doc["id"].(string)
	title, _ := doc["title"].(string)
	content, _ := doc["content"].(string)
	metadata, _ := doc["metadata"].(map[string]interface{})

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

	// Add table metadata
	if metadata != nil {
		intelligence.MetadataEnrichment["table_name"] = metadata["table"]
		intelligence.MetadataEnrichment["schema"] = metadata["schema"]
		intelligence.MetadataEnrichment["row_count"] = metadata["row_count"]
	}

	// Step 0a: Unified Workflow
	var unifiedWorkflowResult map[string]interface{}
	if rp.unifiedWorkflowURL != "" {
		result, err := rp.processViaUnifiedWorkflow(ctx, tableID, title, content)
		if err == nil {
			unifiedWorkflowResult = result
			intelligence.WorkflowResults = result

			// Extract knowledge graph from workflow results
			if kg, ok := result["knowledge_graph"].(map[string]interface{}); ok {
				intelligence.KnowledgeGraph = kg
			}
		}
	}

	// Step 4: Domain Detection
	domain := rp.detectDomain(title + " " + content)
	intelligence.Domain = domain
	intelligence.DomainConfidence = 0.8

	// Process table normally
	if err := rp.processTable(ctx, doc); err != nil {
		return nil, err
	}

	// Collect intelligence from various services
	rp.collectCatalogIntelligence(ctx, tableID, title, content, intelligence)
	rp.collectTrainingIntelligence(ctx, tableID, title, content, intelligence)
	rp.collectDomainIntelligence(ctx, tableID, title, content, intelligence)
	rp.collectSearchIntelligence(ctx, tableID, title, content, intelligence)

	return intelligence, nil
}

// processTable processes a single table through the full pipeline.
func (rp *RelationalPipeline) processTable(ctx context.Context, doc map[string]interface{}) error {
	tableID, _ := doc["id"].(string)
	title, _ := doc["title"].(string)
	content, _ := doc["content"].(string)

	// Step 0a: Process via Unified Workflow
	if rp.unifiedWorkflowURL != "" {
		if rp.logger != nil {
			rp.logger.Printf("Processing table %s via unified workflow", tableID)
		}
		_, err := rp.processViaUnifiedWorkflow(ctx, tableID, title, content)
		if err != nil && rp.logger != nil {
			rp.logger.Printf("Unified workflow processing failed for table %s (non-fatal): %v", tableID, err)
		}
	}

	// Step 2: Catalog Registration
	if rp.catalogURL != "" && content != "" {
		if rp.logger != nil {
			rp.logger.Printf("Registering table %s in catalog", tableID)
		}
		catalogPayload := map[string]interface{}{
			"topic":         title,
			"customer_need": content,
			"source":        "relational",
			"source_id":     tableID,
		}
		var catalogResult map[string]interface{}
		if err := rp.postJSON(ctx, strings.TrimRight(rp.catalogURL, "/")+"/catalog/data-products/build", catalogPayload, &catalogResult); err == nil {
			if rp.logger != nil {
				rp.logger.Printf("Table %s registered in catalog", tableID)
			}
		}
	}

	// Step 3: Training Data Export
	if rp.trainingURL != "" && content != "" {
		if rp.logger != nil {
			rp.logger.Printf("Exporting table %s for training", tableID)
		}
		trainingPayload := map[string]interface{}{
			"table_id": tableID,
			"title":    title,
			"content":  content,
			"source":   "relational",
		}
		var trainingResult map[string]interface{}
		if err := rp.postJSON(ctx, strings.TrimRight(rp.trainingURL, "/")+"/training/export", trainingPayload, &trainingResult); err == nil {
			if rp.logger != nil {
				rp.logger.Printf("Table %s exported for training", tableID)
			}
		}
	}

	// Step 4: Domain Detection
	domain := rp.detectDomain(title + " " + content)

	// Step 5: Local AI Storage
	if rp.localAIURL != "" && content != "" {
		if rp.logger != nil {
			rp.logger.Printf("Storing table %s in Local AI (domain: %s)", tableID, domain)
		}
		localAIPayload := map[string]interface{}{
			"document_id": tableID,
			"title":       title,
			"content":     content,
			"domain":      domain,
			"source":      "relational",
		}
		var localAIResult map[string]interface{}
		// Use LocalAIClient with explicit model selection, retry logic, and circuit breaker
		if rp.localAIClient != nil {
			model := rp.selectModelForDomain(domain)
			if _, err := rp.localAIClient.StoreDocument(ctx, domain, model, localAIPayload); err == nil {
				if rp.logger != nil {
					rp.logger.Printf("Table %s stored in Local AI (domain: %s, model: %s)", tableID, domain, model)
				}
			}
		} else if err := rp.postJSON(ctx, strings.TrimRight(rp.localAIURL, "/")+"/v1/documents", localAIPayload, &localAIResult); err == nil {
			if rp.logger != nil {
				rp.logger.Printf("Table %s stored in Local AI", tableID)
			}
		}
	}

	// Step 6: Search Indexing
	if rp.searchURL != "" && content != "" {
		if rp.logger != nil {
			rp.logger.Printf("Indexing table %s in search", tableID)
		}
		searchPayload := map[string]interface{}{
			"document_id": tableID,
			"title":       title,
			"content":     content,
			"source":      "relational",
		}
		var searchResult map[string]interface{}
		if err := rp.postJSON(ctx, strings.TrimRight(rp.searchURL, "/")+"/v1/index", searchPayload, &searchResult); err == nil {
			if rp.logger != nil {
				rp.logger.Printf("Table %s indexed in search", tableID)
			}
		}
	}

	return nil
}

// Helper methods (similar to DMSPipeline)

func (rp *RelationalPipeline) processViaUnifiedWorkflow(ctx context.Context, tableID, title, content string) (map[string]interface{}, error) {
	if rp.unifiedWorkflowURL == "" {
		return nil, fmt.Errorf("unified workflow not configured")
	}

	payload := map[string]interface{}{
		"document_id": tableID,
		"title":       title,
		"content":     content,
		"source":      "relational",
	}

	url := strings.TrimRight(rp.unifiedWorkflowURL, "/") + "/process"
	var result map[string]interface{}
	if err := rp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("unified workflow processing failed: %w", err)
	}

	return result, nil
}

func (rp *RelationalPipeline) detectDomain(text string) string {
	// Simple domain detection - can be enhanced
	text = strings.ToLower(text)
	if strings.Contains(text, "financial") || strings.Contains(text, "bank") || strings.Contains(text, "accounting") {
		return "finance"
	}
	if strings.Contains(text, "legal") || strings.Contains(text, "law") || strings.Contains(text, "contract") {
		return "legal"
	}
	if strings.Contains(text, "medical") || strings.Contains(text, "health") || strings.Contains(text, "patient") {
		return "healthcare"
	}
	if strings.Contains(text, "technical") || strings.Contains(text, "engineering") || strings.Contains(text, "code") {
		return "technical"
	}
	return "general"
}

// selectModelForDomain selects the appropriate model for a given domain based on domains.json analysis
func (rp *RelationalPipeline) selectModelForDomain(domain string) string {
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

func (rp *RelationalPipeline) postJSON(ctx context.Context, url string, payload interface{}, result interface{}) error {
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal payload: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := rp.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("request failed with status %d: %s", resp.StatusCode, string(body))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("failed to decode response: %w", err)
		}
	}

	return nil
}

// Intelligence collection methods (similar to DMSPipeline)
func (rp *RelationalPipeline) collectCatalogIntelligence(ctx context.Context, tableID, title, content string, intelligence *DocumentIntelligence) {
	if rp.catalogURL == "" {
		return
	}

	patternsURL := strings.TrimRight(rp.catalogURL, "/") + "/catalog/patterns/extract"
	patternsPayload := map[string]interface{}{
		"source_id": tableID,
		"source":    "relational",
		"action":    "extract_patterns",
		"document": map[string]interface{}{
			"id":      tableID,
			"title":   title,
			"content": content,
		},
	}
	var patternsResult map[string]interface{}
	if err := rp.postJSON(ctx, patternsURL, patternsPayload, &patternsResult); err == nil {
		intelligence.CatalogPatterns = patternsResult
	}
}

func (rp *RelationalPipeline) collectTrainingIntelligence(ctx context.Context, tableID, title, content string, intelligence *DocumentIntelligence) {
	if rp.trainingURL == "" {
		return
	}

	feedbackURL := strings.TrimRight(rp.trainingURL, "/") + "/training/feedback"
	feedbackPayload := map[string]interface{}{
		"source_id": tableID,
		"source":    "relational",
	}
	var feedbackResult map[string]interface{}
	if err := rp.postJSON(ctx, feedbackURL, feedbackPayload, &feedbackResult); err == nil {
		intelligence.TrainingPatterns = feedbackResult
	}
}

func (rp *RelationalPipeline) collectDomainIntelligence(ctx context.Context, tableID, title, content string, intelligence *DocumentIntelligence) {
	// Domain intelligence is already set in processTableWithIntelligence
	// This can be enhanced with additional domain-specific analysis
}

func (rp *RelationalPipeline) collectSearchIntelligence(ctx context.Context, tableID, title, content string, intelligence *DocumentIntelligence) {
	if rp.searchURL == "" {
		return
	}

	patternsURL := strings.TrimRight(rp.searchURL, "/") + "/v1/patterns"
	patternsPayload := map[string]interface{}{
		"document_id": tableID,
		"source":      "relational",
	}
	var patternsResult map[string]interface{}
	if err := rp.postJSON(ctx, patternsURL, patternsPayload, &patternsResult); err == nil {
		intelligence.SearchPatterns = patternsResult
	}
}

// Query methods (similar to DMSPipeline)
func (rp *RelationalPipeline) QuerySearch(ctx context.Context, query string, requestID string, topK int) ([]map[string]interface{}, error) {
	if rp.searchURL == "" {
		return nil, fmt.Errorf("search service not configured")
	}

	searchPayload := map[string]interface{}{
		"query":      query,
		"request_id": requestID,
		"top_k":      topK,
		"source":     "relational",
	}

	url := strings.TrimRight(rp.searchURL, "/") + "/v1/search"
	var result map[string]interface{}
	if err := rp.postJSON(ctx, url, searchPayload, &result); err != nil {
		return nil, fmt.Errorf("search query failed: %w", err)
	}

	if results, ok := result["results"].([]interface{}); ok {
		searchResults := make([]map[string]interface{}, len(results))
		for i, r := range results {
			if m, ok := r.(map[string]interface{}); ok {
				searchResults[i] = m
			}
		}
		return searchResults, nil
	}

	return []map[string]interface{}{}, nil
}

func (rp *RelationalPipeline) QueryKnowledgeGraph(ctx context.Context, requestID string, cypherQuery string) (map[string]interface{}, error) {
	if rp.unifiedWorkflowURL == "" {
		return nil, fmt.Errorf("unified workflow not configured")
	}

	payload := map[string]interface{}{
		"request_id": requestID,
		"query":      cypherQuery,
		"source":     "relational",
	}

	url := strings.TrimRight(rp.unifiedWorkflowURL, "/") + "/graph/query"
	var result map[string]interface{}
	if err := rp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("knowledge graph query failed: %w", err)
	}

	return result, nil
}

func (rp *RelationalPipeline) QueryDomainTables(ctx context.Context, domain string, limit, offset int) ([]map[string]interface{}, error) {
	if rp.localAIURL == "" {
		return nil, fmt.Errorf("local AI service not configured")
	}

	payload := map[string]interface{}{
		"domain": domain,
		"limit":  limit,
		"offset": offset,
		"source": "relational",
	}

	url := strings.TrimRight(rp.localAIURL, "/") + "/v1/documents"
	var result map[string]interface{}
	if err := rp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("domain query failed: %w", err)
	}

	if documents, ok := result["documents"].([]interface{}); ok {
		docs := make([]map[string]interface{}, len(documents))
		for i, d := range documents {
			if m, ok := d.(map[string]interface{}); ok {
				docs[i] = m
			}
		}
		return docs, nil
	}

	return []map[string]interface{}{}, nil
}

func (rp *RelationalPipeline) QueryCatalogSemantic(ctx context.Context, query string, objectClass string, property string) ([]map[string]interface{}, error) {
	if rp.catalogURL == "" {
		return nil, fmt.Errorf("catalog service not configured")
	}

	payload := map[string]interface{}{
		"query":        query,
		"object_class": objectClass,
		"property":     property,
		"source":       "relational",
	}

	url := strings.TrimRight(rp.catalogURL, "/") + "/catalog/search"
	var result map[string]interface{}
	if err := rp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("catalog search failed: %w", err)
	}

	if results, ok := result["results"].([]interface{}); ok {
		searchResults := make([]map[string]interface{}, len(results))
		for i, r := range results {
			if m, ok := r.(map[string]interface{}); ok {
				searchResults[i] = m
			}
		}
		return searchResults, nil
	}

	return []map[string]interface{}{}, nil
}

