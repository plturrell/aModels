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
	"path/filepath"
	"strings"
	"time"

	"github.com/plturrell/aModels/pkg/vision"
	"github.com/plturrell/aModels/services/catalog/research"
	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// DMSPipeline orchestrates the full document processing pipeline
// from DMS through OCR, catalog, training, local AI, and search.
type DMSPipeline struct {
	dmsConnector        SourceConnector
	ocrClient           *vision.DeepSeekClient
	deepResearchClient  *research.DeepResearchClient
	unifiedWorkflowURL  string
	catalogURL          string
	trainingURL          string
	localAIURL          string
	searchURL           string
	extractURL          string
	dmsURL              string
	graphServiceURL     string // Priority 5: Graph service URL for GNN queries
	learningOrchestrator *LearningOrchestrator
	requestTracker      *RequestTracker
	logger              *log.Logger
	httpClient          *http.Client
	localAIClient        *LocalAIClient // Standardized LocalAI client with retry and validation
	gpuHelper              *GPUHelper // Phase 3: GPU allocation helper
}

// DMSPipelineConfig configures the pipeline.
// DMSURL is kept for backward compatibility but now points to Extract service.
type DMSPipelineConfig struct {
	DMSURL              string // Deprecated: Use ExtractURL instead
	ExtractURL          string // Extract service URL (replaces DMS)
	DeepSeekOCREndpoint string
	DeepSeekOCRAPIKey   string
	DeepResearchURL     string
	UnifiedWorkflowURL  string
	CatalogURL          string
	TrainingURL         string
	LocalAIURL          string
	SearchURL           string
	Logger              *log.Logger
}

// NewDMSPipeline creates a new document processing pipeline (migrated from DMS to Extract service).
func NewDMSPipeline(config DMSPipelineConfig) (*DMSPipeline, error) {
	// Determine Extract service URL (prefer ExtractURL, fallback to DMSURL for backward compatibility)
	extractURL := config.ExtractURL
	if extractURL == "" {
		extractURL = config.DMSURL
	}
	if extractURL == "" {
		extractURL = "http://localhost:8083" // Default Extract service port
	}
	
	// Create connector for Extract service (replaces DMS connector)
	dmsConfig := map[string]interface{}{
		"base_url":   extractURL,
		"EXTRACT_URL": extractURL,
		"DMS_URL":    extractURL, // Backward compatibility
	}
	dmsConnector := connectors.NewDMSConnector(dmsConfig, config.Logger)

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

	// Get GPU orchestrator URL for GPU allocation (Phase 3)
	gpuOrchestratorURL := os.Getenv("GPU_ORCHESTRATOR_URL")
	if gpuOrchestratorURL == "" {
		gpuOrchestratorURL = "http://gpu-orchestrator:8086"
	}

	pipeline := &DMSPipeline{
		dmsConnector:       dmsConnector,
		ocrClient:          ocrClient,
		deepResearchClient: deepResearchClient,
		unifiedWorkflowURL: config.UnifiedWorkflowURL,
		catalogURL:         config.CatalogURL,
		trainingURL:        config.TrainingURL,
		localAIURL:         config.LocalAIURL,
		searchURL:          config.SearchURL,
		extractURL:         extractURL,
		dmsURL:             extractURL, // Now points to Extract service
		graphServiceURL:    graphServiceURL,
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

	// Phase 3: Initialize GPU helper for GPU allocation
	pipeline.gpuHelper = NewGPUHelper(gpuOrchestratorURL, pipeline.httpClient, config.Logger)

	return pipeline, nil
}

// GetRequestTracker returns the request tracker.
func (dp *DMSPipeline) GetRequestTracker() *RequestTracker {
	return dp.requestTracker
}

// GetDMSConnector returns the DMS connector (for direct document access).
func (dp *DMSPipeline) GetDMSConnector() SourceConnector {
	return dp.dmsConnector
}

// ProcessDocumentsWithTracking processes documents with full request tracking.
// Returns the request ID and processing request for status tracking.
func (dp *DMSPipeline) ProcessDocumentsWithTracking(ctx context.Context, requestID string, query map[string]interface{}) (*ProcessingRequest, error) {
	// Extract document IDs or query from request
	documentIDs, _ := query["document_ids"].([]interface{})
	documentID, _ := query["document_id"].(string)
	
	queryStr := "DMS document processing"
	if documentID != "" {
		queryStr = fmt.Sprintf("Processing document %s", documentID)
	} else if len(documentIDs) > 0 {
		queryStr = fmt.Sprintf("Processing %d documents", len(documentIDs))
	}

	// Create request tracking
	request := dp.requestTracker.CreateRequest(requestID, queryStr)
	dp.requestTracker.UpdateStatus(requestID, RequestStatusProcessing)
	dp.requestTracker.UpdateStep(requestID, "connecting")

	// Apply learned improvements before processing
	if dp.learningOrchestrator != nil {
		if err := dp.learningOrchestrator.ApplyImprovements(ctx, query); err != nil {
			if dp.logger != nil {
				dp.logger.Printf("Failed to apply improvements (non-fatal): %v", err)
			}
		}
	}

	// Step 1: Connect to DMS
	dp.requestTracker.UpdateStep(requestID, "connecting")
	if err := dp.dmsConnector.Connect(ctx, query); err != nil {
		dp.requestTracker.AddErrorWithDetails(requestID, "connect", err.Error(), "", "CONNECTION_ERROR", nil, true)
		dp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("failed to connect to DMS: %w", err)
	}
	defer dp.dmsConnector.Close()

	// Step 2: Extract documents from DMS
	dp.requestTracker.UpdateStep(requestID, "extracting")
	documents, err := dp.dmsConnector.ExtractData(ctx, query)
	if err != nil {
		dp.requestTracker.AddErrorWithDetails(requestID, "extract", err.Error(), "", "EXTRACTION_ERROR", nil, true)
		dp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("failed to extract documents from DMS: %w", err)
	}

	if len(documents) == 0 {
		dp.requestTracker.AddErrorWithDetails(requestID, "extract", "no documents found", "", "NO_DOCUMENTS", nil, false)
		dp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
		return request, fmt.Errorf("no documents found")
	}

	dp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
		DocumentsTotal: len(documents),
	})

	// Step 3: Process each document
	dp.requestTracker.UpdateStep(requestID, "processing")
	processedDocs := make([]ProcessedDocument, 0, len(documents))
	hasErrors := false

	for i, doc := range documents {
		docID, _ := doc["id"].(string)
		if docID == "" {
			docID = fmt.Sprintf("dms_doc_%d", i+1)
		}

		docResult := ProcessedDocument{
			ID:          docID,
			Status:      "processing",
			ProcessedAt: time.Now().UTC().Format(time.RFC3339),
		}

		dp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s", docID))

		// Process document with intelligence collection
		if err := dp.processDocumentWithTracking(ctx, requestID, doc, &docResult); err != nil {
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
			dp.requestTracker.AddErrorWithDetails(requestID, "process_document", err.Error(), docID, errorCode, nil, false)
		} else {
			docResult.Status = "succeeded"
			dp.requestTracker.UpdateStatistics(requestID, ProcessingStatistics{
				DocumentsSucceeded: 1,
			})
		}

		processedDocs = append(processedDocs, docResult)
		dp.requestTracker.AddDocument(requestID, docResult)
	}

	// Set results links
	results := &ProcessingResults{
		CatalogURL: fmt.Sprintf("/api/catalog/documents?source=dms&request_id=%s", requestID),
		SearchURL:  fmt.Sprintf("/api/search?query=dms&request_id=%s", requestID),
		ExportURL:  fmt.Sprintf("/api/dms/results/%s/export", requestID),
	}
	dp.requestTracker.SetResults(requestID, results)

	// Update final status
	dp.requestTracker.UpdateStep(requestID, "completed")
	if hasErrors && len(processedDocs) > 0 {
		dp.requestTracker.UpdateStatus(requestID, RequestStatusPartial)
	} else if hasErrors {
		dp.requestTracker.UpdateStatus(requestID, RequestStatusFailed)
	} else {
		dp.requestTracker.UpdateStatus(requestID, RequestStatusCompleted)
	}

	return request, nil
}

// processDocumentWithTracking processes a single document and updates the document result.
func (dp *DMSPipeline) processDocumentWithTracking(ctx context.Context, requestID string, doc map[string]interface{}, docResult *ProcessedDocument) error {
	docID := docResult.ID

	// Track steps
	dp.requestTracker.UpdateStep(requestID, fmt.Sprintf("processing_%s_unified_workflow", docID))

	// Process document and capture intelligence
	intelligence, err := dp.processDocumentWithIntelligence(ctx, doc)
	if err != nil {
		return err
	}

	// Extract IDs from document processing
	docResult.CatalogID = docID
	docResult.LocalAIID = docID
	docResult.SearchID = docID

	// Add metadata
	docResult.Metadata = map[string]interface{}{
		"source":       "dms",
		"processed_at": docResult.ProcessedAt,
	}

	// Store intelligence data
	if intelligence != nil {
		docResult.Intelligence = intelligence
		dp.requestTracker.SetDocumentIntelligence(requestID, docID, intelligence)
	}

	return nil
}

// processDocumentWithIntelligence processes a document and returns intelligence data.
func (dp *DMSPipeline) processDocumentWithIntelligence(ctx context.Context, doc map[string]interface{}) (*DocumentIntelligence, error) {
	docID, _ := doc["id"].(string)
	name, _ := doc["name"].(string)
	description, _ := doc["description"].(string)
	storagePath, _ := doc["storage_path"].(string)
	content, _ := doc["content"].(string)

	// If content is not provided, try to read from storage path
	if content == "" && storagePath != "" {
		// Try to read from file system (if accessible)
		if filepath.IsAbs(storagePath) {
			if data, err := os.ReadFile(storagePath); err == nil {
				content = string(data)
			}
		}
		// If not accessible, try to fetch from DMS API
		if content == "" && dp.dmsURL != "" {
			content = dp.fetchDocumentContent(ctx, docID)
		}
	}

	title := name
	if title == "" {
		title = docID
	}

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

	// Step 0a: Unified Workflow
	var unifiedWorkflowResult map[string]interface{}
	if dp.unifiedWorkflowURL != "" {
		result, err := dp.processViaUnifiedWorkflow(ctx, docID, title, content)
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
	domain := dp.detectDomain(title + " " + content)
	intelligence.Domain = domain
	intelligence.DomainConfidence = 0.8

	// Process document normally (this will call all the services)
	if err := dp.processDocument(ctx, doc); err != nil {
		return nil, err
	}

	// Collect intelligence from learning endpoints
	if dp.catalogURL != "" {
		dp.collectCatalogIntelligence(ctx, docID, title, content, intelligence)
	}

	if dp.trainingURL != "" {
		dp.collectTrainingIntelligence(ctx, docID, intelligence)
	}

	if dp.localAIURL != "" && domain != "" {
		dp.collectDomainIntelligence(ctx, docID, domain, intelligence)
	}

	if dp.searchURL != "" {
		dp.collectSearchIntelligence(ctx, docID, intelligence)
	}

	return intelligence, nil
}

// fetchDocumentContent fetches document content from Extract service (replaces DMS).
func (dp *DMSPipeline) fetchDocumentContent(ctx context.Context, documentID string) string {
	if dp.dmsURL == "" {
		return ""
	}

	// Fetch document from Extract service and extract content from response
	// Extract service stores content in knowledge graph nodes
	url := fmt.Sprintf("%s/documents/%s", strings.TrimRight(dp.dmsURL, "/"), documentID)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return ""
	}

	resp, err := dp.httpClient.Do(req)
	if err != nil {
		return ""
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return ""
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return ""
	}

	// Extract content from Extract service response
	if content, ok := result["content"].(string); ok && content != "" {
		return content
	}
	if content, ok := result["extraction_summary"].(string); ok && content != "" {
		return content
	}
	// Content may be stored in Gitea, accessible via gitea_url
	return ""
}

// processDocument processes a single document through OCR, catalog, training, local AI, and search.
// This is similar to PerplexityPipeline.processDocument but adapted for DMS documents.
func (dp *DMSPipeline) processDocument(ctx context.Context, doc map[string]interface{}) error {
	docID, _ := doc["id"].(string)
	name, _ := doc["name"].(string)
	storagePath, _ := doc["storage_path"].(string)
	content, _ := doc["content"].(string)

	// If content is not provided, try to read from storage path
	if content == "" && storagePath != "" {
		if filepath.IsAbs(storagePath) {
			if data, err := os.ReadFile(storagePath); err == nil {
				content = string(data)
			}
		}
		if content == "" {
			content = dp.fetchDocumentContent(ctx, docID)
		}
	}

	title := name
	if title == "" {
		title = docID
	}

	// Step 0a: Process via Unified Workflow (if configured)
	if dp.unifiedWorkflowURL != "" {
		if dp.logger != nil {
			dp.logger.Printf("Processing document %s via unified workflow", docID)
		}
		_, err := dp.processViaUnifiedWorkflow(ctx, docID, title, content)
		if err != nil && dp.logger != nil {
			dp.logger.Printf("Unified workflow processing failed for document %s (non-fatal): %v", docID, err)
		}
	}

	// Step 0: Deep Research - Understand context before processing
	var researchContext string
	if dp.deepResearchClient != nil && title != "" {
		if dp.logger != nil {
			dp.logger.Printf("Performing Deep Research for document %s: %s", docID, title)
		}

		report, err := dp.deepResearchClient.ResearchMetadata(ctx, title, true, true)
		if err != nil && dp.logger != nil {
			dp.logger.Printf("Deep Research failed for document %s (non-fatal): %v", docID, err)
		} else if report != nil && report.Report != nil {
			researchContext = report.Report.Summary
			if len(report.Report.Sections) > 0 {
				researchContext += "\n\n" + report.Report.Sections[0].Content
			}
		}
	}

	// Step 1: OCR Processing (if image file)
	processedText := content
	if storagePath != "" {
		ext := strings.ToLower(filepath.Ext(storagePath))
		imageExts := []string{".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
		for _, imgExt := range imageExts {
			if ext == imgExt {
				// Read image file
				imageData, err := os.ReadFile(storagePath)
				if err == nil {
					ocrPrompt := fmt.Sprintf("Extract all text and structured information from this document. Convert to markdown format.")
					if researchContext != "" {
						ocrPrompt = fmt.Sprintf(`Extract all text and structured information from this document about: %s

Context from research:
%s

Convert to markdown format with proper structure.`, title, researchContext)
					}

					ocrText, err := dp.ocrClient.ExtractText(ctx, imageData, ocrPrompt, "")
					if err == nil && ocrText != "" {
						processedText = ocrText
						if dp.logger != nil {
							dp.logger.Printf("OCR extracted text for document %s", docID)
						}
					}
				}
				break
			}
		}
	}

	// Step 2: Catalog Registration
	if dp.catalogURL != "" && processedText != "" {
		if dp.logger != nil {
			dp.logger.Printf("Registering document %s in catalog", docID)
		}
		catalogPayload := map[string]interface{}{
			"topic":         title,
			"customer_need": processedText,
			"source":        "dms",
			"source_id":     docID,
		}
		var catalogResult map[string]interface{}
		if err := dp.postJSON(ctx, strings.TrimRight(dp.catalogURL, "/")+"/catalog/data-products/build", catalogPayload, &catalogResult); err == nil {
			if dp.logger != nil {
				dp.logger.Printf("Document %s registered in catalog", docID)
			}
		}
	}

	// Step 3: Training Data Export
	if dp.trainingURL != "" && processedText != "" {
		if dp.logger != nil {
			dp.logger.Printf("Exporting document %s for training", docID)
		}
		trainingPayload := map[string]interface{}{
			"document_id": docID,
			"title":       title,
			"content":     processedText,
			"source":      "dms",
		}
		var trainingResult map[string]interface{}
		if err := dp.postJSON(ctx, strings.TrimRight(dp.trainingURL, "/")+"/training/export", trainingPayload, &trainingResult); err == nil {
			if dp.logger != nil {
				dp.logger.Printf("Document %s exported for training", docID)
			}
		}
	}

	// Step 4: Domain Detection
	domain := dp.detectDomain(title + " " + processedText)

	// Step 5: Local AI Storage
	if dp.localAIURL != "" && processedText != "" {
		if dp.logger != nil {
			dp.logger.Printf("Storing document %s in Local AI (domain: %s)", docID, domain)
		}
		localAIPayload := map[string]interface{}{
			"document_id": docID,
			"title":       title,
			"content":     processedText,
			"domain":      domain,
			"source":      "dms",
		}
		var localAIResult map[string]interface{}
		// Use LocalAIClient with explicit model selection, retry logic, and circuit breaker
		if dp.localAIClient != nil {
			model := dp.selectModelForDomain(domain)
			if _, err := dp.localAIClient.StoreDocument(ctx, domain, model, localAIPayload); err == nil {
				if dp.logger != nil {
					dp.logger.Printf("Document %s stored in Local AI (domain: %s, model: %s)", docID, domain, model)
				}
			}
		} else if err := dp.postJSON(ctx, strings.TrimRight(dp.localAIURL, "/")+"/v1/documents", localAIPayload, &localAIResult); err == nil {
			if dp.logger != nil {
				dp.logger.Printf("Document %s stored in Local AI", docID)
			}
		}
	}

	// Step 6: Search Indexing
	if dp.searchURL != "" && processedText != "" {
		if dp.logger != nil {
			dp.logger.Printf("Indexing document %s in search", docID)
		}
		searchPayload := map[string]interface{}{
			"document_id": docID,
			"title":       title,
			"content":     processedText,
			"source":      "dms",
		}
		var searchResult map[string]interface{}
		if err := dp.postJSON(ctx, strings.TrimRight(dp.searchURL, "/")+"/v1/index", searchPayload, &searchResult); err == nil {
			if dp.logger != nil {
				dp.logger.Printf("Document %s indexed in search", docID)
			}
		}
	}

	return nil
}

// Helper methods (reused from PerplexityPipeline pattern)
func (dp *DMSPipeline) processViaUnifiedWorkflow(ctx context.Context, docID, title, content string) (map[string]interface{}, error) {
	if dp.unifiedWorkflowURL == "" {
		return nil, fmt.Errorf("unified workflow not configured")
	}

	payload := map[string]interface{}{
		"document_id": docID,
		"title":       title,
		"content":     content,
		"source":      "dms",
	}

	url := strings.TrimRight(dp.unifiedWorkflowURL, "/") + "/process"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("unified workflow processing failed: %w", err)
	}

	return result, nil
}

func (dp *DMSPipeline) detectDomain(text string) string {
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
func (dp *DMSPipeline) selectModelForDomain(domain string) string {
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

func (dp *DMSPipeline) postJSON(ctx context.Context, url string, payload interface{}, result interface{}) error {
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

	resp, err := dp.httpClient.Do(req)
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

// collectCatalogIntelligence, collectTrainingIntelligence, collectDomainIntelligence, collectSearchIntelligence
// These methods are similar to PerplexityPipeline - reusing the same pattern
func (dp *DMSPipeline) collectCatalogIntelligence(ctx context.Context, docID, title, content string, intelligence *DocumentIntelligence) {
	if dp.catalogURL == "" {
		return
	}

	// Get patterns
	patternsURL := strings.TrimRight(dp.catalogURL, "/") + "/catalog/patterns/extract"
	patternsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "dms",
		"action":    "extract_patterns",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var patternsResult map[string]interface{}
	if err := dp.postJSON(ctx, patternsURL, patternsPayload, &patternsResult); err == nil {
		intelligence.CatalogPatterns = patternsResult
		if patterns, ok := patternsResult["patterns"].(map[string]interface{}); ok {
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
	relationshipsURL := strings.TrimRight(dp.catalogURL, "/") + "/catalog/relationships/discover"
	relationshipsPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "dms",
		"action":    "discover_relationships",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var relationshipsResult map[string]interface{}
	if err := dp.postJSON(ctx, relationshipsURL, relationshipsPayload, &relationshipsResult); err == nil {
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
	enrichURL := strings.TrimRight(dp.catalogURL, "/") + "/catalog/metadata/enrich"
	enrichPayload := map[string]interface{}{
		"source_id": docID,
		"source":    "dms",
		"action":    "enrich_metadata",
		"document": map[string]interface{}{
			"id":      docID,
			"title":   title,
			"content": content,
		},
	}
	var enrichResult map[string]interface{}
	if err := dp.postJSON(ctx, enrichURL, enrichPayload, &enrichResult); err == nil {
		intelligence.MetadataEnrichment = enrichResult
	}
}

func (dp *DMSPipeline) collectTrainingIntelligence(ctx context.Context, docID string, intelligence *DocumentIntelligence) {
	// Placeholder - similar to PerplexityPipeline
}

func (dp *DMSPipeline) collectDomainIntelligence(ctx context.Context, docID, domain string, intelligence *DocumentIntelligence) {
	if dp.localAIURL == "" {
		return
	}

	patternURL := strings.TrimRight(dp.localAIURL, "/") + "/v1/domains/" + domain + "/patterns"
	patternPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "get_patterns",
	}
	var patternResult map[string]interface{}
	if err := dp.postJSON(ctx, patternURL, patternPayload, &patternResult); err == nil {
		intelligence.DomainPatterns = patternResult
	}
}

func (dp *DMSPipeline) collectSearchIntelligence(ctx context.Context, docID string, intelligence *DocumentIntelligence) {
	if dp.searchURL == "" {
		return
	}

	patternURL := strings.TrimRight(dp.searchURL, "/") + "/patterns/learn"
	patternPayload := map[string]interface{}{
		"document_id": docID,
		"action":      "get_patterns",
	}
	var patternResult map[string]interface{}
	if err := dp.postJSON(ctx, patternURL, patternPayload, &patternResult); err == nil {
		intelligence.SearchPatterns = patternResult
	}
}

// Query methods (similar to PerplexityPipeline)
func (dp *DMSPipeline) QuerySearch(ctx context.Context, query, requestID string, topK int, filters map[string]interface{}) ([]map[string]interface{}, error) {
	if dp.searchURL == "" {
		return nil, fmt.Errorf("search service not configured")
	}

	searchPayload := map[string]interface{}{
		"query": query,
		"top_k": topK,
	}
	if filters != nil {
		searchPayload["filters"] = filters
	}
	if requestID != "" {
		if searchPayload["filters"] == nil {
			searchPayload["filters"] = make(map[string]interface{})
		}
		if filtersMap, ok := searchPayload["filters"].(map[string]interface{}); ok {
			filtersMap["request_id"] = requestID
			filtersMap["source"] = "dms"
		}
	} else {
		if searchPayload["filters"] == nil {
			searchPayload["filters"] = make(map[string]interface{})
		}
		if filtersMap, ok := searchPayload["filters"].(map[string]interface{}); ok {
			filtersMap["source"] = "dms"
		}
	}

	url := strings.TrimRight(dp.searchURL, "/") + "/v1/search"
	var searchResult map[string]interface{}
	if err := dp.postJSON(ctx, url, searchPayload, &searchResult); err != nil {
		return nil, fmt.Errorf("search query failed: %w", err)
	}

	if results, ok := searchResult["results"].([]interface{}); ok {
		documents := make([]map[string]interface{}, 0, len(results))
		for _, r := range results {
			if doc, ok := r.(map[string]interface{}); ok {
				documents = append(documents, doc)
			}
		}
		return documents, nil
	}

	return nil, fmt.Errorf("search query failed")
}

func (dp *DMSPipeline) QueryKnowledgeGraph(ctx context.Context, requestID, cypherQuery string, params map[string]interface{}) (map[string]interface{}, error) {
	tracker := dp.GetRequestTracker()
	request, exists := tracker.GetRequest(requestID)
	if !exists {
		return nil, fmt.Errorf("request not found: %s", requestID)
	}

	if dp.unifiedWorkflowURL != "" {
		graphRAGRequest := map[string]interface{}{
			"graphrag_request": map[string]interface{}{
				"query":       cypherQuery,
				"strategy":    "bfs",
				"max_depth":   3,
				"max_results": 10,
				"params":      params,
				"enrich":      true,
			},
		}

		url := strings.TrimRight(dp.unifiedWorkflowURL, "/") + "/graphrag/query"
		var result map[string]interface{}
		if err := dp.postJSON(ctx, url, graphRAGRequest, &result); err == nil {
			return result, nil
		}
	}

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
	if err := dp.postJSON(ctx, url, kgQueryPayload, &result); err != nil {
		return nil, fmt.Errorf("knowledge graph query failed: %w", err)
	}

	if result != nil {
		result["request_id"] = requestID
		result["document_ids"] = request.DocumentIDs
	}

	return result, nil
}

func (dp *DMSPipeline) QueryDomainDocuments(ctx context.Context, domain string, limit, offset int) ([]map[string]interface{}, error) {
	if dp.localAIURL == "" {
		return nil, fmt.Errorf("local AI service not configured")
	}

	url := fmt.Sprintf("%s/v1/domains/%s/documents?limit=%d&offset=%d",
		strings.TrimRight(dp.localAIURL, "/"), domain, limit, offset)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := dp.httpClient.Do(req)
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

func (dp *DMSPipeline) QueryCatalogSemantic(ctx context.Context, query, objectClass, property, source string, filters map[string]interface{}) (map[string]interface{}, error) {
	if dp.catalogURL == "" {
		return nil, fmt.Errorf("catalog service not configured")
	}

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
		searchPayload["source"] = "dms"
	}
	if filters != nil {
		searchPayload["filters"] = filters
	}

	url := strings.TrimRight(dp.catalogURL, "/") + "/catalog/semantic-search"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, searchPayload, &result); err != nil {
		return nil, fmt.Errorf("catalog semantic search failed: %w", err)
	}

	return result, nil
}

// QueryGNNEmbeddings queries GNN service for graph/node embeddings (Priority 5).
func (dp *DMSPipeline) QueryGNNEmbeddings(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, graphLevel bool) (map[string]interface{}, error) {
	// Try graph service hybrid query endpoint first
	if dp.graphServiceURL != "" {
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

		url := strings.TrimRight(dp.graphServiceURL, "/") + "/gnn/query"
		var result map[string]interface{}
		if err := dp.postJSON(ctx, url, hybridPayload, &result); err == nil {
			return result, nil
		}
	}

	// Fallback to training service directly
	if dp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes":      nodes,
		"edges":      edges,
		"graph_level": graphLevel,
	}

	url := strings.TrimRight(dp.trainingURL, "/") + "/gnn/embeddings"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN embeddings query failed: %w", err)
	}

	return result, nil
}

// QueryGNNStructuralInsights queries GNN service for structural insights (Priority 5).
func (dp *DMSPipeline) QueryGNNStructuralInsights(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, insightType string, threshold float64) (map[string]interface{}, error) {
	// Try graph service hybrid query endpoint first
	if dp.graphServiceURL != "" {
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
					"threshold":    threshold,
				},
			}
			hybridPayload["gnn_query_request"] = gnnRequest
		}

		url := strings.TrimRight(dp.graphServiceURL, "/") + "/gnn/query"
		var result map[string]interface{}
		if err := dp.postJSON(ctx, url, hybridPayload, &result); err == nil {
			return result, nil
		}
	}

	// Fallback to training service directly
	if dp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes":        nodes,
		"edges":        edges,
		"insight_type": insightType,
		"threshold":    threshold,
	}

	url := strings.TrimRight(dp.trainingURL, "/") + "/gnn/structural-insights"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN structural insights query failed: %w", err)
	}

	return result, nil
}

// QueryGNNPredictLinks queries GNN service for link predictions (Priority 5).
func (dp *DMSPipeline) QueryGNNPredictLinks(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, candidatePairs [][]string, topK int) (map[string]interface{}, error) {
	if dp.trainingURL == "" {
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

	url := strings.TrimRight(dp.trainingURL, "/") + "/gnn/predict-links"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN link prediction query failed: %w", err)
	}

	return result, nil
}

// QueryGNNClassifyNodes queries GNN service for node classification (Priority 5).
func (dp *DMSPipeline) QueryGNNClassifyNodes(ctx context.Context, nodes []map[string]interface{}, edges []map[string]interface{}, topK *int) (map[string]interface{}, error) {
	if dp.trainingURL == "" {
		return nil, fmt.Errorf("training service not configured")
	}

	payload := map[string]interface{}{
		"nodes": nodes,
		"edges": edges,
	}
	if topK != nil {
		payload["top_k"] = *topK
	}

	url := strings.TrimRight(dp.trainingURL, "/") + "/gnn/classify"
	var result map[string]interface{}
	if err := dp.postJSON(ctx, url, payload, &result); err != nil {
		return nil, fmt.Errorf("GNN node classification query failed: %w", err)
	}

	return result, nil
}

// QueryGNNHybrid queries both KG and GNN and combines results (Priority 5).
func (dp *DMSPipeline) QueryGNNHybrid(ctx context.Context, query string, projectID, systemID string, queryKG, queryGNN bool, gnnType string, combine bool) (map[string]interface{}, error) {
	// Use graph service hybrid query endpoint
	if dp.graphServiceURL != "" {
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

		url := strings.TrimRight(dp.graphServiceURL, "/") + "/gnn/hybrid-query"
		var result map[string]interface{}
		if err := dp.postJSON(ctx, url, hybridPayload, &result); err != nil {
			return nil, fmt.Errorf("hybrid query failed: %w", err)
		}

		return result, nil
	}

	return nil, fmt.Errorf("graph service not configured")
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

