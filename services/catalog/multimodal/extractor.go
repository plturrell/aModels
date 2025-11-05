package multimodal

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/catalog/iso11179"
)

// MultiModalExtractor provides extraction capabilities for various data formats.
type MultiModalExtractor struct {
	deepSeekOCRURL string
	httpClient     *http.Client
	logger         *log.Logger
}

// NewMultiModalExtractor creates a new multi-modal extractor.
func NewMultiModalExtractor(deepSeekOCRURL string, logger *log.Logger) *MultiModalExtractor {
	return &MultiModalExtractor{
		deepSeekOCRURL: deepSeekOCRURL,
		httpClient:     &http.Client{Timeout: 120 * time.Second},
		logger:         logger,
	}
}

// ExtractionRequest represents a request to extract metadata.
type ExtractionRequest struct {
	Source      string            `json:"source"`       // File path, URL, or API endpoint
	SourceType  string            `json:"source_type"` // "pdf", "image", "api_rest", "api_graphql", "api_grpc"
	Options     ExtractionOptions `json:"options,omitempty"`
}

// ExtractionOptions configures extraction behavior.
type ExtractionOptions struct {
	ExtractTables    bool `json:"extract_tables"`    // Extract tables from PDFs/images
	ExtractText      bool `json:"extract_text"`      // Extract text content
	ExtractSchemas   bool `json:"extract_schemas"`   // Extract schemas from APIs
	ExtractEndpoints bool `json:"extract_endpoints"` // Extract API endpoints
}

// ExtractedMetadata represents extracted metadata.
type ExtractedMetadata struct {
	Source        string                        `json:"source"`
	SourceType    string                        `json:"source_type"`
	DataElements  []*iso11179.DataElement       `json:"data_elements"`
	Tables        []TableMetadata               `json:"tables,omitempty"`
	Text          string                        `json:"text,omitempty"`
	Endpoints     []EndpointMetadata            `json:"endpoints,omitempty"`
	Schemas       []SchemaMetadata              `json:"schemas,omitempty"`
	ExtractedAt   time.Time                     `json:"extracted_at"`
	Confidence    float64                       `json:"confidence"`
}

// TableMetadata represents table metadata extracted from PDFs/images.
type TableMetadata struct {
	TableName    string                 `json:"table_name"`
	Columns      []ColumnMetadata       `json:"columns"`
	Rows         int                    `json:"rows"`
	Location     string                 `json:"location,omitempty"` // Page number, coordinates
	Confidence   float64                `json:"confidence"`
}

// ColumnMetadata represents column metadata.
type ColumnMetadata struct {
	Name        string `json:"name"`
	DataType    string `json:"data_type"`
	SampleValue string `json:"sample_value,omitempty"`
}

// EndpointMetadata represents API endpoint metadata.
type EndpointMetadata struct {
	Path        string            `json:"path"`
	Method      string            `json:"method"`
	Description string            `json:"description,omitempty"`
	Parameters  []ParameterMetadata `json:"parameters,omitempty"`
	Response    SchemaMetadata    `json:"response,omitempty"`
}

// ParameterMetadata represents API parameter metadata.
type ParameterMetadata struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Required    bool   `json:"required"`
	Description string `json:"description,omitempty"`
}

// SchemaMetadata represents API schema metadata.
type SchemaMetadata struct {
	Name        string                 `json:"name"`
	Type        string                 `json:"type"`
	Properties  map[string]interface{} `json:"properties,omitempty"`
	Description string                 `json:"description,omitempty"`
}

// Extract extracts metadata from a source.
func (mme *MultiModalExtractor) Extract(ctx context.Context, req ExtractionRequest) (*ExtractedMetadata, error) {
	mme.logger.Printf("Extracting metadata from %s (type: %s)", req.Source, req.SourceType)

	extracted := &ExtractedMetadata{
		Source:      req.Source,
		SourceType:  req.SourceType,
		ExtractedAt: time.Now(),
		Confidence:  0.0,
	}

	switch req.SourceType {
	case "pdf", "image":
		return mme.extractFromPDFOrImage(ctx, req, extracted)
	case "api_rest":
		return mme.extractFromRESTAPI(ctx, req, extracted)
	case "api_graphql":
		return mme.extractFromGraphQLAPI(ctx, req, extracted)
	case "api_grpc":
		return mme.extractFromgRPCAPI(ctx, req, extracted)
	default:
		return nil, fmt.Errorf("unsupported source type: %s", req.SourceType)
	}
}

// extractFromPDFOrImage extracts metadata from PDF or image files.
func (mme *MultiModalExtractor) extractFromPDFOrImage(
	ctx context.Context,
	req ExtractionRequest,
	extracted *ExtractedMetadata,
) (*ExtractedMetadata, error) {
	if mme.deepSeekOCRURL == "" {
		return nil, fmt.Errorf("DeepSeek OCR service not configured")
	}

	// Call DeepSeek OCR service
	url := fmt.Sprintf("%s/extract", mme.deepSeekOCRURL)
	payload := map[string]interface{}{
		"source": req.Source,
		"type":   req.SourceType,
		"options": map[string]bool{
			"extract_tables": req.Options.ExtractTables,
			"extract_text":   req.Options.ExtractText,
		},
	}

	jsonData, _ := json.Marshal(payload)
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := mme.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to call OCR service: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OCR service returned status: %d", resp.StatusCode)
	}

	var result struct {
		Text    string         `json:"text"`
		Tables  []TableMetadata `json:"tables"`
		Confidence float64    `json:"confidence"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode OCR response: %w", err)
	}

	extracted.Text = result.Text
	extracted.Tables = result.Tables
	extracted.Confidence = result.Confidence

	// Convert tables to data elements
	for _, table := range result.Tables {
		element := iso11179.NewDataElement(
			fmt.Sprintf("table:%s", table.TableName),
			table.TableName,
			fmt.Sprintf("Table extracted from %s", req.Source),
			"table",
			fmt.Sprintf("http://amodels.org/catalog/tables/%s", table.TableName),
		)
		element.AddMetadata("extracted_from", req.Source)
		element.AddMetadata("extraction_confidence", table.Confidence)
		extracted.DataElements = append(extracted.DataElements, element)
	}

	return extracted, nil
}

// extractFromRESTAPI extracts metadata from a REST API.
func (mme *MultiModalExtractor) extractFromRESTAPI(
	ctx context.Context,
	req ExtractionRequest,
	extracted *ExtractedMetadata,
) (*ExtractedMetadata, error) {
	// Discovery: Try to fetch OpenAPI/Swagger spec
	specURL := fmt.Sprintf("%s/openapi.json", req.Source)
	if specURL == "" {
		specURL = fmt.Sprintf("%s/swagger.json", req.Source)
	}

	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodGet, specURL, nil)
	resp, err := mme.httpClient.Do(httpReq)
	if err == nil && resp.StatusCode == http.StatusOK {
		// Parse OpenAPI spec
		var spec map[string]interface{}
		if err := json.NewDecoder(resp.Body).Decode(&spec); err == nil {
			extracted.Endpoints = mme.parseOpenAPISpec(spec)
			extracted.Confidence = 0.9
		}
		resp.Body.Close()
	}

	// If no spec found, try to discover endpoints
	if len(extracted.Endpoints) == 0 {
		extracted.Endpoints = mme.discoverRESTEndpoints(ctx, req.Source)
		extracted.Confidence = 0.6
	}

	// Convert endpoints to data elements
	for _, endpoint := range extracted.Endpoints {
		element := iso11179.NewDataElement(
			fmt.Sprintf("endpoint:%s:%s", endpoint.Method, endpoint.Path),
			endpoint.Path,
			endpoint.Description,
			"api_endpoint",
			fmt.Sprintf("http://amodels.org/catalog/endpoints/%s/%s", endpoint.Method, endpoint.Path),
		)
		element.AddMetadata("method", endpoint.Method)
		extracted.DataElements = append(extracted.DataElements, element)
	}

	return extracted, nil
}

// extractFromGraphQLAPI extracts metadata from a GraphQL API.
func (mme *MultiModalExtractor) extractFromGraphQLAPI(
	ctx context.Context,
	req ExtractionRequest,
	extracted *ExtractedMetadata,
) (*ExtractedMetadata, error) {
	// Try to fetch GraphQL schema via introspection
	url := req.Source
	payload := map[string]interface{}{
		"query": `
			query IntrospectionQuery {
				__schema {
					types {
						name
						kind
						fields {
							name
							type {
								name
								kind
							}
						}
					}
				}
			}
		`,
	}

	jsonData, _ := json.Marshal(payload)
	httpReq, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := mme.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to introspect GraphQL API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GraphQL API returned status: %d", resp.StatusCode)
	}

	var result struct {
		Data struct {
			Schema struct {
				Types []struct {
					Name   string `json:"name"`
					Kind   string `json:"kind"`
					Fields []struct {
						Name string `json:"name"`
						Type struct {
							Name string `json:"name"`
							Kind string `json:"kind"`
						} `json:"type"`
					} `json:"fields"`
				} `json:"types"`
			} `json:"__schema"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode GraphQL response: %w", err)
	}

	// Convert GraphQL types to schemas
	for _, gqlType := range result.Data.Schema.Types {
		if gqlType.Kind == "OBJECT" {
			schema := SchemaMetadata{
				Name:       gqlType.Name,
				Type:       "object",
				Properties: make(map[string]interface{}),
			}
			for _, field := range gqlType.Fields {
				schema.Properties[field.Name] = map[string]string{
					"type": field.Type.Name,
					"kind": field.Type.Kind,
				}
			}
			extracted.Schemas = append(extracted.Schemas, schema)

			// Create data element
			element := iso11179.NewDataElement(
				fmt.Sprintf("graphql_type:%s", gqlType.Name),
				gqlType.Name,
				fmt.Sprintf("GraphQL type: %s", gqlType.Name),
				"graphql_type",
				fmt.Sprintf("http://amodels.org/catalog/graphql/%s", gqlType.Name),
			)
			extracted.DataElements = append(extracted.DataElements, element)
		}
	}

	extracted.Confidence = 0.9
	return extracted, nil
}

// extractFromgRPCAPI extracts metadata from a gRPC API.
func (mme *MultiModalExtractor) extractFromgRPCAPI(
	ctx context.Context,
	req ExtractionRequest,
	extracted *ExtractedMetadata,
) (*ExtractedMetadata, error) {
	// gRPC reflection would be used here
	// For now, return a placeholder
	extracted.Confidence = 0.5
	return extracted, fmt.Errorf("gRPC API extraction not yet fully implemented")
}

// parseOpenAPISpec parses an OpenAPI specification.
func (mme *MultiModalExtractor) parseOpenAPISpec(spec map[string]interface{}) []EndpointMetadata {
	var endpoints []EndpointMetadata

	paths, ok := spec["paths"].(map[string]interface{})
	if !ok {
		return endpoints
	}

	for path, pathItem := range paths {
		pathItemMap, ok := pathItem.(map[string]interface{})
		if !ok {
			continue
		}

		for method, operation := range pathItemMap {
			operationMap, ok := operation.(map[string]interface{})
			if !ok {
				continue
			}

			endpoint := EndpointMetadata{
				Path:        path,
				Method:      method,
				Description: getString(operationMap, "description"),
			}

			// Parse parameters
			if params, ok := operationMap["parameters"].([]interface{}); ok {
				for _, param := range params {
					paramMap, ok := param.(map[string]interface{})
					if !ok {
						continue
					}
					endpoint.Parameters = append(endpoint.Parameters, ParameterMetadata{
						Name:        getString(paramMap, "name"),
						Type:        getString(paramMap, "type"),
						Required:    getBool(paramMap, "required"),
						Description: getString(paramMap, "description"),
					})
				}
			}

			endpoints = append(endpoints, endpoint)
		}
	}

	return endpoints
}

// discoverRESTEndpoints attempts to discover REST endpoints.
func (mme *MultiModalExtractor) discoverRESTEndpoints(ctx context.Context, baseURL string) []EndpointMetadata {
	// Placeholder: Would use common REST discovery techniques
	// For now, return empty list
	return []EndpointMetadata{}
}

// Helper functions
func getString(m map[string]interface{}, key string) string {
	if val, ok := m[key].(string); ok {
		return val
	}
	return ""
}

func getBool(m map[string]interface{}, key string) bool {
	if val, ok := m[key].(bool); ok {
		return val
	}
	return false
}

