package connectors

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// DMSConnector connects to DMS FastAPI service to fetch documents.
type DMSConnector struct {
	config     map[string]interface{}
	logger     *log.Logger
	httpClient *http.Client
	baseURL    string
}

// DMSDocument represents a document from DMS API.
type DMSDocument struct {
	ID                string    `json:"id"`
	Name              string    `json:"name"`
	Description       *string   `json:"description"`
	StoragePath       string    `json:"storage_path"`
	CatalogIdentifier *string   `json:"catalog_identifier"`
	ExtractionSummary *string   `json:"extraction_summary"`
	CreatedAt         string    `json:"created_at"`
	UpdatedAt         string    `json:"updated_at"`
	Content           string    `json:"content,omitempty"` // Will be populated from storage_path if needed
}

// NewDMSConnector creates a new DMS connector.
func NewDMSConnector(config map[string]interface{}, logger *log.Logger) *DMSConnector {
	baseURL, _ := config["base_url"].(string)
	if baseURL == "" {
		baseURL = config["DMS_URL"].(string)
	}
	if baseURL == "" {
		baseURL = "http://localhost:8096" // Default DMS port
	}

	return &DMSConnector{
		config:     config,
		logger:     logger,
		baseURL:    baseURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

// Connect establishes connection to DMS API.
func (dc *DMSConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if dc.logger != nil {
		dc.logger.Printf("Connecting to DMS API at %s", dc.baseURL)
	}

	// Merge provided config
	for k, v := range config {
		dc.config[k] = v
	}

	// Update base URL if provided
	if baseURL, ok := config["base_url"].(string); ok && baseURL != "" {
		dc.baseURL = baseURL
	}

	// Test connection with a simple request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, dc.baseURL+"/documents", nil)
	if err != nil {
		return fmt.Errorf("failed to create test request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := dc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to DMS API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("DMS API connection test failed with status %d", resp.StatusCode)
	}

	return nil
}

// DiscoverSchema discovers schema from DMS API.
// For DMS, we treat each document as a "table" with content fields.
func (dc *DMSConnector) DiscoverSchema(ctx context.Context) (*agents.SourceSchema, error) {
	schema := &agents.SourceSchema{
		SourceType: "dms",
		Tables: []agents.TableDefinition{
			{
				Name: "documents",
				Columns: []agents.ColumnDefinition{
					{Name: "id", Type: "string", Nullable: false},
					{Name: "name", Type: "string", Nullable: false},
					{Name: "description", Type: "string", Nullable: true},
					{Name: "storage_path", Type: "string", Nullable: false},
					{Name: "catalog_identifier", Type: "string", Nullable: true},
					{Name: "extraction_summary", Type: "text", Nullable: true},
					{Name: "created_at", Type: "timestamp", Nullable: false},
					{Name: "updated_at", Type: "timestamp", Nullable: false},
					{Name: "content", Type: "text", Nullable: true},
				},
				PrimaryKey: []string{"id"},
			},
		},
		Relations: []agents.RelationDefinition{},
		Metadata: map[string]interface{}{
			"system":      "dms",
			"source_type": "internal_api",
			"api_version": "v1",
		},
	}

	if dc.logger != nil {
		dc.logger.Printf("Discovered DMS schema: %d tables", len(schema.Tables))
	}

	return schema, nil
}

// ExtractData extracts documents from DMS API.
// Query parameters:
//   - document_id: Specific document ID to fetch
//   - limit: Maximum number of documents to fetch (default: 10)
//   - offset: Offset for pagination (default: 0)
//   - include_content: Whether to include document content (default: false)
func (dc *DMSConnector) ExtractData(ctx context.Context, query map[string]interface{}) ([]map[string]interface{}, error) {
	// Check if specific document ID is requested
	if documentID, ok := query["document_id"].(string); ok && documentID != "" {
		doc, err := dc.fetchDocumentByID(ctx, documentID, query)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch document %s: %w", documentID, err)
		}
		return []map[string]interface{}{doc}, nil
	}

	// Otherwise, list documents
	limit := 10
	if l, ok := query["limit"].(int); ok && l > 0 {
		limit = l
	}

	offset := 0
	if o, ok := query["offset"].(int); ok && o >= 0 {
		offset = o
	}

	documents, err := dc.listDocuments(ctx, limit, offset, query)
	if err != nil {
		return nil, fmt.Errorf("failed to list DMS documents: %w", err)
	}

	// Convert to map format
	result := make([]map[string]interface{}, len(documents))
	for i, doc := range documents {
		result[i] = map[string]interface{}{
			"id":                 doc.ID,
			"name":               doc.Name,
			"description":        doc.Description,
			"storage_path":       doc.StoragePath,
			"catalog_identifier": doc.CatalogIdentifier,
			"extraction_summary": doc.ExtractionSummary,
			"created_at":         doc.CreatedAt,
			"updated_at":         doc.UpdatedAt,
			"content":            doc.Content,
		}
	}

	if dc.logger != nil {
		dc.logger.Printf("Extracted %d documents from DMS", len(result))
	}

	return result, nil
}

// fetchDocumentByID fetches a specific document by ID.
func (dc *DMSConnector) fetchDocumentByID(ctx context.Context, documentID string, query map[string]interface{}) (map[string]interface{}, error) {
	url := fmt.Sprintf("%s/documents/%s", dc.baseURL, documentID)
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	resp, err := dc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("document %s not found", documentID)
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("DMS API returned status %d: %s", resp.StatusCode, string(body))
	}

	var doc DMSDocument
	if err := json.NewDecoder(resp.Body).Decode(&doc); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Optionally fetch content if requested
	includeContent, _ := query["include_content"].(bool)
	if includeContent && doc.StoragePath != "" {
		content, err := dc.readDocumentContent(ctx, doc.StoragePath)
		if err == nil {
			doc.Content = content
		}
	}

	return map[string]interface{}{
		"id":                 doc.ID,
		"name":               doc.Name,
		"description":        doc.Description,
		"storage_path":       doc.StoragePath,
		"catalog_identifier": doc.CatalogIdentifier,
		"extraction_summary": doc.ExtractionSummary,
		"created_at":         doc.CreatedAt,
		"updated_at":         doc.UpdatedAt,
		"content":            doc.Content,
	}, nil
}

// listDocuments lists documents from DMS API.
func (dc *DMSConnector) listDocuments(ctx context.Context, limit, offset int, query map[string]interface{}) ([]DMSDocument, error) {
	url := fmt.Sprintf("%s/documents", dc.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	// Add query parameters if needed (DMS API may support pagination)
	q := req.URL.Query()
	if limit > 0 {
		q.Set("limit", fmt.Sprintf("%d", limit))
	}
	if offset > 0 {
		q.Set("offset", fmt.Sprintf("%d", offset))
	}
	req.URL.RawQuery = q.Encode()

	resp, err := dc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("DMS API returned status %d: %s", resp.StatusCode, string(body))
	}

	var documents []DMSDocument
	if err := json.NewDecoder(resp.Body).Decode(&documents); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Optionally fetch content if requested
	includeContent, _ := query["include_content"].(bool)
	if includeContent {
		for i := range documents {
			if documents[i].StoragePath != "" {
				content, err := dc.readDocumentContent(ctx, documents[i].StoragePath)
				if err == nil {
					documents[i].Content = content
				}
			}
		}
	}

	return documents, nil
}

// readDocumentContent reads content from a document's storage path.
// This is a placeholder - in production, you might need to fetch from DMS service
// or read directly from storage if accessible.
func (dc *DMSConnector) readDocumentContent(ctx context.Context, storagePath string) (string, error) {
	// For now, return empty - content should be fetched via DMS API if needed
	// In production, this might call a DMS endpoint to get document content
	return "", nil
}

// Close closes the connection.
func (dc *DMSConnector) Close() error {
	if dc.logger != nil {
		dc.logger.Printf("Closing DMS connection")
	}
	return nil
}

