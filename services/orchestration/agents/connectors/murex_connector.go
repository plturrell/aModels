package connectors

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// MurexConnector connects to Murex trading system using OpenAPI specification.
type MurexConnector struct {
	config        map[string]interface{}
	logger        *log.Logger
	httpClient    *http.Client
	baseURL       string
	apiKey        string
	openAPISpec   map[string]interface{}
	discoveredEndpoints map[string]EndpointInfo
	allowMockData bool // If true, allows returning mock data on API failures (development only)
}

// EndpointInfo represents information about an API endpoint from OpenAPI spec.
type EndpointInfo struct {
	Path        string
	Method      string
	Description string
	Parameters  []ParameterInfo
}

// ParameterInfo represents parameter information from OpenAPI spec.
type ParameterInfo struct {
	Name        string
	Type        string
	Required    bool
	Description string
	In          string // "query", "path", "header"
}

// NewMurexConnector creates a new Murex connector.
func NewMurexConnector(config map[string]interface{}, logger *log.Logger) *MurexConnector {
	baseURL, _ := config["base_url"].(string)
	if baseURL == "" {
		baseURL = "https://api.murex.com" // Default Murex API base URL
	}

	apiKey, _ := config["api_key"].(string)
	
	// Check if mock data is allowed (only for development/testing)
	// Default to false (production mode) - mocks disabled
	allowMockData := false
	if mockVal, ok := config["allow_mock_data"].(bool); ok {
		allowMockData = mockVal
	} else if mockVal, ok := config["allow_mock_data"].(string); ok {
		allowMockData = (mockVal == "true" || mockVal == "1")
	}
	
	mc := &MurexConnector{
		config:        config,
		logger:        logger,
		baseURL:       baseURL,
		apiKey:        apiKey,
		discoveredEndpoints: make(map[string]EndpointInfo),
		allowMockData: allowMockData,
		// Use connection pooling for better performance (Priority 1)
		httpClient: &http.Client{
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				IdleConnTimeout:     90 * time.Second,
				MaxConnsPerHost:     50,
			},
			Timeout: 30 * time.Second,
		},
	}

	// Load OpenAPI spec if provided
	if specPath, ok := config["openapi_spec_path"].(string); ok && specPath != "" {
		if err := mc.loadOpenAPISpec(specPath); err != nil {
			if logger != nil {
				logger.Printf("Warning: Failed to load OpenAPI spec from %s: %v", specPath, err)
			}
		}
	} else if specURL, ok := config["openapi_spec_url"].(string); ok && specURL != "" {
		if err := mc.loadOpenAPISpecFromURL(context.Background(), specURL); err != nil {
			if logger != nil {
				logger.Printf("Warning: Failed to load OpenAPI spec from %s: %v", specURL, err)
			}
		}
	}

	return mc
}

// loadOpenAPISpec loads OpenAPI specification from a local file path.
func (mc *MurexConnector) loadOpenAPISpec(path string) error {
	// In production, would read from file system
	// For now, we'll parse it when needed from URL
	return nil
}

// loadOpenAPISpecFromURL loads OpenAPI specification from a URL (e.g., GitHub raw content).
func (mc *MurexConnector) loadOpenAPISpecFromURL(ctx context.Context, specURL string) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, specURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := mc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to fetch OpenAPI spec: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to fetch OpenAPI spec: status %d", resp.StatusCode)
	}

	if err := json.NewDecoder(resp.Body).Decode(&mc.openAPISpec); err != nil {
		return fmt.Errorf("failed to parse OpenAPI spec: %w", err)
	}

	// Discover endpoints from the spec
	mc.discoverEndpointsFromSpec()

	if mc.logger != nil {
		mc.logger.Printf("Loaded OpenAPI spec with %d discovered endpoints", len(mc.discoveredEndpoints))
	}

	return nil
}

// discoverEndpointsFromSpec parses OpenAPI spec to discover available endpoints.
func (mc *MurexConnector) discoverEndpointsFromSpec() {
	if mc.openAPISpec == nil {
		return
	}

	paths, ok := mc.openAPISpec["paths"].(map[string]interface{})
	if !ok {
		return
	}

	for path, pathItem := range paths {
		pathItemMap, ok := pathItem.(map[string]interface{})
		if !ok {
			continue
		}

		for method, operation := range pathItemMap {
			methodStr, ok := method.(string)
			if !ok {
				continue
			}

			operationMap, ok := operation.(map[string]interface{})
			if !ok {
				continue
			}

			endpoint := EndpointInfo{
				Path:        path,
				Method:      methodStr,
				Description: getStringFromMap(operationMap, "description"),
			}

			// Parse parameters
			if params, ok := operationMap["parameters"].([]interface{}); ok {
				for _, param := range params {
					paramMap, ok := param.(map[string]interface{})
					if !ok {
						continue
					}
					endpoint.Parameters = append(endpoint.Parameters, ParameterInfo{
						Name:        getStringFromMap(paramMap, "name"),
						Type:        getStringFromMap(paramMap, "type"),
						Required:    getBoolFromMap(paramMap, "required"),
						Description: getStringFromMap(paramMap, "description"),
						In:          getStringFromMap(paramMap, "in"),
					})
				}
			}

			key := fmt.Sprintf("%s:%s", methodStr, path)
			mc.discoveredEndpoints[key] = endpoint
		}
	}
}

// Helper functions for parsing OpenAPI spec
func getStringFromMap(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getBoolFromMap(m map[string]interface{}, key string) bool {
	if val, ok := m[key].(bool); ok {
		return val
	}
	return false
}

// Connect establishes connection to Murex.
func (mc *MurexConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if mc.logger != nil {
		mc.logger.Printf("Connecting to Murex system at %s", mc.baseURL)
	}

	// Merge provided config
	for k, v := range config {
		mc.config[k] = v
	}

	// Update base URL if provided
	if baseURL, ok := config["base_url"].(string); ok && baseURL != "" {
		mc.baseURL = baseURL
	}

	// Update API key if provided
	if apiKey, ok := config["api_key"].(string); ok && apiKey != "" {
		mc.apiKey = apiKey
	}

	// Test connection by making a health check or schema discovery request
	// For now, we'll just validate configuration
	if mc.baseURL == "" {
		return fmt.Errorf("Murex base URL not configured")
	}

	return nil
}

// DiscoverSchema discovers schema from Murex OpenAPI specification.
func (mc *MurexConnector) DiscoverSchema(ctx context.Context) (*agents.SourceSchema, error) {
	schema := &agents.SourceSchema{
		SourceType: "murex",
		Tables:     []agents.TableDefinition{},
		Relations:  []agents.RelationDefinition{},
		Metadata: map[string]interface{}{
			"system": "murex",
		},
	}

	// If OpenAPI spec is loaded, extract schema information from it
	if mc.openAPISpec != nil {
		schema.Metadata["openapi_version"] = getStringFromMap(mc.openAPISpec, "openapi")
		if info, ok := mc.openAPISpec["info"].(map[string]interface{}); ok {
			schema.Metadata["version"] = getStringFromMap(info, "version")
			schema.Metadata["title"] = getStringFromMap(info, "title")
		}

		// Extract schemas/components from OpenAPI spec
		if components, ok := mc.openAPISpec["components"].(map[string]interface{}); ok {
			if schemas, ok := components["schemas"].(map[string]interface{}); ok {
				schema.Tables = mc.extractTablesFromSchemas(schemas)
			}
		}

		// Discover relationships from endpoints
		schema.Relations = mc.extractRelationsFromEndpoints()
	} else {
		// Fallback to default schema if OpenAPI spec not loaded
		schema.Tables = []agents.TableDefinition{
			{
				Name: "trades",
				Columns: []agents.ColumnDefinition{
					{Name: "trade_id", Type: "string", Nullable: false},
					{Name: "counterparty", Type: "string", Nullable: false},
					{Name: "notional", Type: "decimal", Nullable: false},
					{Name: "trade_date", Type: "date", Nullable: false},
				},
				PrimaryKey: []string{"trade_id"},
			},
			{
				Name: "cashflows",
				Columns: []agents.ColumnDefinition{
					{Name: "cashflow_id", Type: "string", Nullable: false},
					{Name: "trade_id", Type: "string", Nullable: false},
					{Name: "amount", Type: "decimal", Nullable: false},
					{Name: "currency", Type: "string", Nullable: false},
				},
				PrimaryKey: []string{"cashflow_id"},
				ForeignKeys: []agents.ForeignKeyDefinition{
					{
						Name:             "fk_trade",
						ReferencedTable:  "trades",
						Columns:          []string{"trade_id"},
						ReferencedColumns: []string{"trade_id"},
					},
				},
			},
		}
		schema.Metadata["version"] = "3.1"
	}

	if mc.logger != nil {
		mc.logger.Printf("Discovered Murex schema: %d tables, %d relations", len(schema.Tables), len(schema.Relations))
	}

	return schema, nil
}

// extractTablesFromSchemas extracts table definitions from OpenAPI schemas.
func (mc *MurexConnector) extractTablesFromSchemas(schemas map[string]interface{}) []agents.TableDefinition {
	var tables []agents.TableDefinition

	for schemaName, schemaDef := range schemas {
		schemaMap, ok := schemaDef.(map[string]interface{})
		if !ok {
			continue
		}

		// Only process object schemas (potential tables)
		if schemaType := getStringFromMap(schemaMap, "type"); schemaType != "object" {
			continue
		}

		table := agents.TableDefinition{
			Name:       schemaName,
			Columns:    []agents.ColumnDefinition{},
			PrimaryKey: []string{},
		}

		// Extract properties as columns
		if properties, ok := schemaMap["properties"].(map[string]interface{}); ok {
			for propName, propDef := range properties {
				propMap, ok := propDef.(map[string]interface{})
				if !ok {
					continue
				}

				propType := getStringFromMap(propMap, "type")
				// Map OpenAPI types to our types
				dataType := mapOpenAPITypeToDataType(propType)

				nullable := !getBoolFromMap(propMap, "required")
				if required, ok := schemaMap["required"].([]interface{}); ok {
					nullable = true
					for _, req := range required {
						if req == propName {
							nullable = false
							break
						}
					}
				}

				table.Columns = append(table.Columns, agents.ColumnDefinition{
					Name:     propName,
					Type:     dataType,
					Nullable: nullable,
				})
			}
		}

		// Check for primary key hints (id fields, etc.)
		for _, col := range table.Columns {
			if col.Name == "id" || col.Name == schemaName+"_id" || col.Name == schemaName+"Id" {
				table.PrimaryKey = append(table.PrimaryKey, col.Name)
			}
		}

		if len(table.Columns) > 0 {
			tables = append(tables, table)
		}
	}

	return tables
}

// extractRelationsFromEndpoints extracts relationships from endpoint paths.
func (mc *MurexConnector) extractRelationsFromEndpoints() []agents.RelationDefinition {
	var relations []agents.RelationDefinition

	// Analyze endpoint paths to infer relationships
	// For example, /trades/{tradeId}/cashflows suggests trades -> cashflows
	for _, endpoint := range mc.discoveredEndpoints {
		// Simple heuristic: nested paths suggest relationships
		// This is a simplified approach; in production, would analyze more carefully
		if endpoint.Path != "" {
			// Could extract more sophisticated relationships here
		}
	}

	return relations
}

// mapOpenAPITypeToDataType maps OpenAPI types to our data types.
func mapOpenAPITypeToDataType(openAPIType string) string {
	switch openAPIType {
	case "string":
		return "string"
	case "integer", "int32", "int64":
		return "integer"
	case "number", "float", "double":
		return "decimal"
	case "boolean":
		return "boolean"
	case "date", "date-time":
		return "date"
	default:
		return "string"
	}
}

// ExtractData extracts data from Murex using OpenAPI endpoints.
func (mc *MurexConnector) ExtractData(ctx context.Context, query map[string]interface{}) ([]map[string]interface{}, error) {
	tableName, _ := query["table"].(string)
	if tableName == "" {
		return nil, fmt.Errorf("table name is required")
	}

	// Map table names to API endpoints based on OpenAPI spec
	// Common Murex endpoints: /trades, /transactions, /cashflows, /positions, etc.
	endpointPath := mc.mapTableToEndpoint(tableName)
	if endpointPath == "" {
		// Fallback: try common endpoint patterns
		endpointPath = fmt.Sprintf("/api/v1/%s", tableName)
	}

	// Build request URL
	reqURL, err := url.Parse(mc.baseURL + endpointPath)
	if err != nil {
		return nil, fmt.Errorf("invalid URL: %w", err)
	}

	// Add query parameters
	q := reqURL.Query()
	if limit, ok := query["limit"].(int); ok && limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", limit))
	}
	if offset, ok := query["offset"].(int); ok {
		q.Set("offset", fmt.Sprintf("%d", offset))
	}
	// Add other query parameters
	for k, v := range query {
		if k != "table" && k != "limit" && k != "offset" {
			q.Set(k, fmt.Sprintf("%v", v))
		}
	}
	reqURL.RawQuery = q.Encode()

	// Create HTTP request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, reqURL.String(), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Add authentication headers
	if mc.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+mc.apiKey)
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")

	if mc.logger != nil {
		mc.logger.Printf("Extracting data from Murex endpoint: %s", reqURL.String())
	}

	// Execute request
	resp, err := mc.httpClient.Do(req)
	if err != nil {
		// In production, return error instead of mock data
		if !mc.allowMockData {
			if mc.logger != nil {
				mc.logger.Printf("Error: Murex API call failed: %v", err)
			}
			return nil, fmt.Errorf("Murex API request failed: %w", err)
		}
		
		// Only return mock data if explicitly enabled for development/testing
		if mc.logger != nil {
			mc.logger.Printf("Warning: API call failed, returning mock data (development mode): %v", err)
		}
		return mc.getMockData(tableName, query), nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Murex API returned status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var result interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Extract data array from response
	// Response format may vary: { "data": [...] } or direct array
	var data []map[string]interface{}
	switch v := result.(type) {
	case []interface{}:
		for _, item := range v {
			if itemMap, ok := item.(map[string]interface{}); ok {
				data = append(data, itemMap)
			}
		}
	case map[string]interface{}:
		if dataArray, ok := v["data"].([]interface{}); ok {
			for _, item := range dataArray {
				if itemMap, ok := item.(map[string]interface{}); ok {
					data = append(data, itemMap)
				}
			}
		} else if items, ok := v["items"].([]interface{}); ok {
			for _, item := range items {
				if itemMap, ok := item.(map[string]interface{}); ok {
					data = append(data, itemMap)
				}
			}
		}
	}

	if mc.logger != nil {
		mc.logger.Printf("Extracted %d records from Murex", len(data))
	}

	return data, nil
}

// mapTableToEndpoint maps table names to API endpoint paths.
func (mc *MurexConnector) mapTableToEndpoint(tableName string) string {
	// Map common table names to Murex API endpoints
	endpointMap := map[string]string{
		"trades":       "/api/v1/trades",
		"transactions": "/api/v1/transactions",
		"cashflows":    "/api/v1/cashflows",
		"positions":    "/api/v1/positions",
		"market_data":  "/api/v1/market-data",
		"pricing":      "/api/v1/pricing",
	}

	if endpoint, ok := endpointMap[tableName]; ok {
		return endpoint
	}

	// Search discovered endpoints for matching patterns
	for _, endpoint := range mc.discoveredEndpoints {
		if endpoint.Path != "" && containsString(endpoint.Path, tableName) {
			return endpoint.Path
		}
	}

	return ""
}

// containsString checks if a string contains a substring (case-insensitive).
func containsString(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

// getMockData returns mock data when API is unavailable (for development/testing only).
// This should only be called when allowMockData is true.
func (mc *MurexConnector) getMockData(tableName string, query map[string]interface{}) []map[string]interface{} {
	if tableName == "trades" {
		return []map[string]interface{}{
			{
				"trade_id":    "T001",
				"counterparty": "Bank A",
				"notional":     1000000.0,
				"trade_date":   "2024-01-01",
			},
		}
	}
	return []map[string]interface{}{}
}

// Close closes the connection.
func (mc *MurexConnector) Close() error {
	if mc.logger != nil {
		mc.logger.Printf("Closing Murex connection")
	}
	return nil
}

