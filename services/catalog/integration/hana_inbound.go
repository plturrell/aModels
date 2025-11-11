package integration

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/pkg/sap"
	"github.com/plturrell/aModels/services/catalog/httpclient"
	"github.com/plturrell/aModels/services/catalog/security"
)

// HANAInboundIntegration handles inbound SAP HANA Cloud table integration
type HANAInboundIntegration struct {
	hanaClient              *sap.HANAClient
	extractServiceURL       string
	trainingServiceURL       string
	localaiURL              string
	searchServiceURL        string
	privacyDomainIntegration *security.PrivacyDomainIntegration
	logger                  *log.Logger
	extractHTTPClient       *httpclient.Client
}

// HANAInboundRequest represents a request to process HANA tables
type HANAInboundRequest struct {
	Schema      string   `json:"schema"`
	Tables      []string `json:"tables"`
	DomainID    string   `json:"domain_id,omitempty"` // Optional: specify domain for processing
	ProjectID   string   `json:"project_id"`
	SystemID    string   `json:"system_id,omitempty"`
	EnablePrivacy bool   `json:"enable_privacy,omitempty"` // Apply differential privacy
	OutputFormat string  `json:"output_format,omitempty"` // json, jsonl, csv
}

// HANAInboundResponse represents the response from processing
type HANAInboundResponse struct {
	RequestID      string                 `json:"request_id"`
	Status         string                 `json:"status"`
	ExtractedNodes int                    `json:"extracted_nodes"`
	ExtractedEdges int                    `json:"extracted_edges"`
	DomainID       string                 `json:"domain_id"`
	PrivacyApplied bool                   `json:"privacy_applied"`
	TrainingJobID  string                 `json:"training_job_id,omitempty"`
	SearchIndexID  string                 `json:"search_index_id,omitempty"`
	ProcessingTime time.Duration          `json:"processing_time_ms"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
}

// NewHANAInboundIntegration creates a new HANA inbound integration
func NewHANAInboundIntegration(
	hanaClient *sap.HANAClient,
	extractServiceURL string,
	trainingServiceURL string,
	localaiURL string,
	searchServiceURL string,
	privacyDomainIntegration *security.PrivacyDomainIntegration,
	logger *log.Logger,
) *HANAInboundIntegration {
	var extractHTTPClient *httpclient.Client
	if extractServiceURL != "" {
		extractHTTPClient = httpclient.NewClient(httpclient.ClientConfig{
			Timeout:         5 * time.Minute,
			MaxRetries:      3,
			InitialBackoff:  1 * time.Second,
			MaxBackoff:      10 * time.Second,
			BaseURL:         extractServiceURL,
			HealthCheckPath: "/healthz",
			Logger:          logger,
		})
	}
	
	return &HANAInboundIntegration{
		hanaClient:              hanaClient,
		extractServiceURL:       extractServiceURL,
		trainingServiceURL:      trainingServiceURL,
		localaiURL:              localaiURL,
		searchServiceURL:        searchServiceURL,
		privacyDomainIntegration: privacyDomainIntegration,
		logger:                  logger,
		extractHTTPClient:       extractHTTPClient,
	}
}

// ProcessHANATables processes HANA tables through the full pipeline
func (hii *HANAInboundIntegration) ProcessHANATables(
	ctx context.Context,
	req HANAInboundRequest,
) (*HANAInboundResponse, error) {
	startTime := time.Now()
	requestID := fmt.Sprintf("hana-inbound-%d", time.Now().UnixNano())

	// Get user's domain access from XSUAA context
	var domainID string
	var privacyConfig *security.PrivacyConfig
	var err error

	if hii.privacyDomainIntegration != nil {
		// Auto-detect domain if not specified
		if req.DomainID == "" {
			domains, err := hii.privacyDomainIntegration.GetUserDomains(ctx)
			if err == nil && len(domains) > 0 {
				domainID = domains[0] // Use first accessible domain
			} else {
				domainID = "default"
			}
		} else {
			domainID = req.DomainID
			// Verify user has access to specified domain
			canAccess, _, err := hii.privacyDomainIntegration.CanAccessDomain(ctx, domainID)
			if err != nil {
				return nil, fmt.Errorf("failed to verify domain access: %w", err)
			}
			if !canAccess {
				return nil, fmt.Errorf("user does not have access to domain: %s", domainID)
			}
		}

		// Get privacy configuration for domain
		if req.EnablePrivacy {
			privacyConfig, err = hii.privacyDomainIntegration.GetPrivacyConfig(ctx, domainID)
			if err != nil {
				hii.logger.Printf("WARNING: Failed to get privacy config: %v, proceeding without privacy", err)
			}
		}
	} else {
		domainID = req.DomainID
		if domainID == "" {
			domainID = "default"
		}
	}

	// Step 1: Extract data from HANA tables
	hii.logger.Printf("[HANA_INBOUND] Starting extraction: schema=%s tables=%v domain=%s", req.Schema, req.Tables, domainID)
	
	extractResult, err := hii.extractFromHANA(ctx, req, domainID)
	if err != nil {
		return nil, fmt.Errorf("extraction failed: %w", err)
	}

	// Step 2: Process through extraction service to create knowledge graph
	hii.logger.Printf("[HANA_INBOUND] Processing through extraction service: nodes=%d edges=%d", 
		extractResult.Nodes, extractResult.Edges)
	
	graphData, err := hii.processThroughExtraction(ctx, extractResult, domainID, privacyConfig)
	if err != nil {
		return nil, fmt.Errorf("extraction processing failed: %w", err)
	}

	// Step 3: Send to training service (if enabled)
	var trainingJobID string
	if hii.trainingServiceURL != "" {
		hii.logger.Printf("[HANA_INBOUND] Sending to training service")
		trainingJobID, err = hii.sendToTraining(ctx, graphData, domainID, privacyConfig)
		if err != nil {
			hii.logger.Printf("WARNING: Training service failed: %v", err)
			// Continue even if training fails
		}
	}

	// Step 4: Index in search service
	var searchIndexID string
	if hii.searchServiceURL != "" {
		hii.logger.Printf("[HANA_INBOUND] Indexing in search service")
		searchIndexID, err = hii.indexInSearch(ctx, graphData, domainID)
		if err != nil {
			hii.logger.Printf("WARNING: Search indexing failed: %v", err)
			// Continue even if search indexing fails
		}
	}

	processingTime := time.Since(startTime)

	response := &HANAInboundResponse{
		RequestID:      requestID,
		Status:         "completed",
		ExtractedNodes: graphData.Nodes,
		ExtractedEdges: graphData.Edges,
		DomainID:       domainID,
		PrivacyApplied: privacyConfig != nil,
		TrainingJobID:  trainingJobID,
		SearchIndexID:  searchIndexID,
		ProcessingTime: processingTime,
		Metadata: map[string]interface{}{
			"schema":        req.Schema,
			"tables":       req.Tables,
			"project_id":   req.ProjectID,
			"system_id":    req.SystemID,
			"privacy_level": getPrivacyLevel(privacyConfig),
		},
	}

	hii.logger.Printf("[HANA_INBOUND] Processing completed: request_id=%s nodes=%d edges=%d time=%v",
		requestID, graphData.Nodes, graphData.Edges, processingTime)

	return response, nil
}

// ExtractResult represents extracted data from HANA
type ExtractResult struct {
	Schema  string
	Tables  []string
	Data    []map[string]interface{}
	Nodes   int
	Edges   int
}

// GraphData represents knowledge graph data
type GraphData struct {
	Nodes int
	Edges int
	Data  map[string]interface{}
}

// extractFromHANA extracts data from HANA tables
func (hii *HANAInboundIntegration) extractFromHANA(
	ctx context.Context,
	req HANAInboundRequest,
	domainID string,
) (*ExtractResult, error) {
	if hii.hanaClient == nil {
		return nil, fmt.Errorf("HANA client not configured")
	}

	var allData []map[string]interface{}
	tables := req.Tables

	// If no tables specified, get all tables from schema
	if len(tables) == 0 {
		// Query information_schema for tables
		query := fmt.Sprintf(`
			SELECT TABLE_NAME 
			FROM SYS.TABLES 
			WHERE SCHEMA_NAME = '%s'
		`, req.Schema)
		
		rows, err := hii.hanaClient.ExecuteQuery(ctx, query)
		if err != nil {
			return nil, fmt.Errorf("failed to list tables: %w", err)
		}
		defer rows.Close()

		for rows.Next() {
			var tableName string
			if err := rows.Scan(&tableName); err != nil {
				continue
			}
			tables = append(tables, tableName)
		}
	}

	// Extract data from each table
	for _, tableName := range tables {
		query := fmt.Sprintf(`SELECT * FROM "%s"."%s" LIMIT 10000`, req.Schema, tableName)
		
		rows, err := hii.hanaClient.ExecuteQuery(ctx, query)
		if err != nil {
			hii.logger.Printf("WARNING: Failed to query table %s: %v", tableName, err)
			continue
		}
		defer rows.Close()

		columns, err := rows.Columns()
		if err != nil {
			continue
		}

		for rows.Next() {
			values := make([]interface{}, len(columns))
			valuePtrs := make([]interface{}, len(columns))
			for i := range values {
				valuePtrs[i] = &values[i]
			}

			if err := rows.Scan(valuePtrs...); err != nil {
				continue
			}

			row := make(map[string]interface{})
			row["_table"] = tableName
			row["_schema"] = req.Schema
			row["_domain"] = domainID
			for i, col := range columns {
				row[col] = values[i]
			}

			allData = append(allData, row)
		}
	}

	// Estimate nodes and edges (will be refined by extraction service)
	nodes := len(allData) * 2 // Rough estimate: 2 nodes per row
	edges := len(allData)      // Rough estimate: 1 edge per row

	return &ExtractResult{
		Schema: req.Schema,
		Tables: tables,
		Data:   allData,
		Nodes:  nodes,
		Edges:  edges,
	}, nil
}

// processThroughExtraction processes data through extraction service
func (hii *HANAInboundIntegration) processThroughExtraction(
	ctx context.Context,
	extractResult *ExtractResult,
	domainID string,
	privacyConfig *security.PrivacyConfig,
) (*GraphData, error) {
	if hii.extractServiceURL == "" {
		return nil, fmt.Errorf("extract service URL not configured")
	}

	// Prepare extraction request
	extractReq := map[string]interface{}{
		"source":      "hana",
		"schema":      extractResult.Schema,
		"tables":      extractResult.Tables,
		"data":        extractResult.Data,
		"domain_id":   domainID,
		"project_id":  "hana-inbound",
		"system_id":   "hana-cloud",
	}

	// Add privacy configuration if available
	if privacyConfig != nil {
		extractReq["privacy"] = map[string]interface{}{
			"epsilon":      privacyConfig.Epsilon,
			"delta":       privacyConfig.Delta,
			"noise_scale":  privacyConfig.NoiseScale,
			"max_queries":  privacyConfig.MaxQueries,
			"privacy_level": privacyConfig.PrivacyLevel,
		}
	}

	// Use enhanced HTTP client if available, otherwise fall back to basic client
	var extractResp map[string]interface{}
	
	if hii.extractHTTPClient != nil {
		// Use enhanced HTTP client with retry, circuit breaker, health checks, etc.
		// Add auth token to context if available
		if req, ok := ctx.Value("http_request").(*http.Request); ok {
			if authHeader := req.Header.Get("Authorization"); authHeader != "" {
				ctx = context.WithValue(ctx, httpclient.AuthTokenKey, authHeader)
			}
		}
		
		// Validate response structure
		validator := func(data map[string]interface{}) error {
			if _, ok := data["nodes"]; !ok {
				return fmt.Errorf("response missing 'nodes' field")
			}
			if _, ok := data["edges"]; !ok {
				return fmt.Errorf("response missing 'edges' field")
			}
			return nil
		}
		
		err := hii.extractHTTPClient.PostJSON(ctx, "/extract", extractReq, &extractResp, validator)
		if err != nil {
			return nil, fmt.Errorf("extraction service request failed: %w", err)
		}
	} else {
		// Fallback to basic HTTP client
		reqBody, err := json.Marshal(extractReq)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request: %w", err)
		}

		httpReq, err := http.NewRequestWithContext(ctx, "POST", hii.extractServiceURL+"/extract", 
			bytes.NewBuffer(reqBody))
		if err != nil {
			return nil, fmt.Errorf("failed to create request: %w", err)
		}

		// Forward XSUAA token if available (from original request)
		if req, ok := ctx.Value("http_request").(*http.Request); ok {
			if authHeader := req.Header.Get("Authorization"); authHeader != "" {
				httpReq.Header.Set("Authorization", authHeader)
			}
		}

		httpReq.Header.Set("Content-Type", "application/json")

		client := &http.Client{Timeout: 5 * time.Minute}
		resp, err := client.Do(httpReq)
		if err != nil {
			return nil, fmt.Errorf("extraction service request failed: %w", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			// Read response body for better error messages
			body, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("extraction service returned status %d: %s", resp.StatusCode, string(body))
		}

		if err := json.NewDecoder(resp.Body).Decode(&extractResp); err != nil {
			return nil, fmt.Errorf("failed to decode response: %w", err)
		}
	}

	nodes, _ := extractResp["nodes"].(float64)
	edges, _ := extractResp["edges"].(float64)

	return &GraphData{
		Nodes: int(nodes),
		Edges: int(edges),
		Data:  extractResp,
	}, nil
}

// sendToTraining sends data to training service
func (hii *HANAInboundIntegration) sendToTraining(
	ctx context.Context,
	graphData *GraphData,
	domainID string,
	privacyConfig *security.PrivacyConfig,
) (string, error) {
	if hii.trainingServiceURL == "" {
		return "", fmt.Errorf("training service URL not configured")
	}

	trainingReq := map[string]interface{}{
		"source":     "hana",
		"graph_data": graphData.Data,
		"domain_id":  domainID,
		"project_id": "hana-inbound",
	}

	// Add privacy configuration
	if privacyConfig != nil {
		trainingReq["privacy_config"] = map[string]interface{}{
			"epsilon":       privacyConfig.Epsilon,
			"delta":         privacyConfig.Delta,
			"privacy_level": privacyConfig.PrivacyLevel,
		}
	}

	reqBody, err := json.Marshal(trainingReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", hii.trainingServiceURL+"/training/process",
		bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Forward XSUAA token if available
	if req, ok := ctx.Value("http_request").(*http.Request); ok {
		if authHeader := req.Header.Get("Authorization"); authHeader != "" {
			httpReq.Header.Set("Authorization", authHeader)
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("training service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		return "", fmt.Errorf("training service returned status %d", resp.StatusCode)
	}

	var trainingResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&trainingResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	jobID, _ := trainingResp["job_id"].(string)
	return jobID, nil
}

// indexInSearch indexes data in search service
func (hii *HANAInboundIntegration) indexInSearch(
	ctx context.Context,
	graphData *GraphData,
	domainID string,
) (string, error) {
	if hii.searchServiceURL == "" {
		return "", fmt.Errorf("search service URL not configured")
	}

	searchReq := map[string]interface{}{
		"source":     "hana",
		"graph_data": graphData.Data,
		"domain_id":  domainID,
	}

	reqBody, err := json.Marshal(searchReq)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", hii.searchServiceURL+"/v1/index",
		bytes.NewBuffer(reqBody))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}

	// Forward XSUAA token if available
	if req, ok := ctx.Value("http_request").(*http.Request); ok {
		if authHeader := req.Header.Get("Authorization"); authHeader != "" {
			httpReq.Header.Set("Authorization", authHeader)
		}
	}

	httpReq.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Minute}
	resp, err := client.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("search service request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		return "", fmt.Errorf("search service returned status %d", resp.StatusCode)
	}

	var searchResp map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&searchResp); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	indexID, _ := searchResp["index_id"].(string)
	return indexID, nil
}

// getTokenFromContext extracts XSUAA token from context
func getTokenFromContext(ctx context.Context) string {
	// Try to get token from request context (stored by XSUAA middleware)
	// The token is typically in the Authorization header of the original request
	// For now, we'll extract it from the request if available
	if req, ok := ctx.Value("http_request").(*http.Request); ok {
		authHeader := req.Header.Get("Authorization")
		if authHeader != "" && len(authHeader) > 7 && authHeader[:7] == "Bearer " {
			return authHeader[7:]
		}
	}
	return ""
}

// getPrivacyLevel returns privacy level string from config
func getPrivacyLevel(config *security.PrivacyConfig) string {
	if config == nil {
		return "none"
	}
	return config.PrivacyLevel
}

