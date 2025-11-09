package connectors

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
)

// PerplexityConnector connects to Perplexity API to fetch documents.
type PerplexityConnector struct {
	config     map[string]interface{}
	logger     *log.Logger
	httpClient *http.Client
	apiKey     string
	baseURL    string
}

// PerplexityDocument represents a document from Perplexity API.
type PerplexityDocument struct {
	ID          string                 `json:"id"`
	Title       string                 `json:"title"`
	Content     string                 `json:"content"`
	ImageURL    string                 `json:"image_url,omitempty"`
	ImageBase64 string                 `json:"image_base64,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedAt   string                 `json:"created_at,omitempty"`
}

// NewPerplexityConnector creates a new Perplexity connector.
func NewPerplexityConnector(config map[string]interface{}, logger *log.Logger) *PerplexityConnector {
	apiKey, _ := config["api_key"].(string)
	if apiKey == "" {
		apiKey = config["PERPLEXITY_API_KEY"].(string)
	}

	baseURL, _ := config["base_url"].(string)
	if baseURL == "" {
		baseURL = "https://api.perplexity.ai"
	}

	// Use connection pooling for better performance (Priority 1)
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 10,
		IdleConnTimeout:     90 * time.Second,
		MaxConnsPerHost:     50,
	}
	
	return &PerplexityConnector{
		config:     config,
		logger:     logger,
		apiKey:     apiKey,
		baseURL:    baseURL,
		httpClient: &http.Client{
			Transport: transport,
			Timeout:   60 * time.Second,
		},
	}
}

// Connect establishes connection to Perplexity API.
func (pc *PerplexityConnector) Connect(ctx context.Context, config map[string]interface{}) error {
	if pc.logger != nil {
		pc.logger.Printf("Connecting to Perplexity API at %s", pc.baseURL)
	}

	// Merge provided config
	for k, v := range config {
		pc.config[k] = v
	}

	// Update API key if provided
	if apiKey, ok := config["api_key"].(string); ok && apiKey != "" {
		pc.apiKey = apiKey
	}

	// Update base URL if provided
	if baseURL, ok := config["base_url"].(string); ok && baseURL != "" {
		pc.baseURL = baseURL
	}

	if pc.apiKey == "" {
		return fmt.Errorf("Perplexity API key not configured")
	}

	// Test connection with a simple request
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, pc.baseURL+"/models", nil)
	if err != nil {
		return fmt.Errorf("failed to create test request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+pc.apiKey)

	resp, err := pc.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to connect to Perplexity API: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("Perplexity API connection test failed with status %d", resp.StatusCode)
	}

	return nil
}

// DiscoverSchema discovers schema from Perplexity API.
// For Perplexity, we treat each document as a "table" with content fields.
func (pc *PerplexityConnector) DiscoverSchema(ctx context.Context) (*agents.SourceSchema, error) {
	schema := &agents.SourceSchema{
		SourceType: "perplexity",
		Tables: []agents.TableDefinition{
			{
				Name: "documents",
				Columns: []agents.ColumnDefinition{
					{Name: "id", Type: "string", Nullable: false},
					{Name: "title", Type: "string", Nullable: true},
					{Name: "content", Type: "text", Nullable: false},
					{Name: "image_url", Type: "string", Nullable: true},
					{Name: "image_base64", Type: "string", Nullable: true},
					{Name: "created_at", Type: "timestamp", Nullable: true},
					{Name: "metadata", Type: "json", Nullable: true},
				},
				PrimaryKey: []string{"id"},
			},
		},
		Relations: []agents.RelationDefinition{},
		Metadata: map[string]interface{}{
			"system":        "perplexity",
			"source_type":   "api",
			"api_version":  "v1",
		},
	}

	if pc.logger != nil {
		pc.logger.Printf("Discovered Perplexity schema: %d tables", len(schema.Tables))
	}

	return schema, nil
}

// ExtractData extracts documents from Perplexity API.
// Query parameters:
//   - query: Search query string
//   - model: Model to use (default: "sonar")
//   - limit: Maximum number of documents to fetch
//   - include_images: Whether to include image data
func (pc *PerplexityConnector) ExtractData(ctx context.Context, query map[string]interface{}) ([]map[string]interface{}, error) {
	searchQuery, _ := query["query"].(string)
	if searchQuery == "" {
		// If no query provided, try to fetch recent documents or use a default query
		searchQuery, _ = query["default_query"].(string)
		if searchQuery == "" {
			searchQuery = "latest research documents"
		}
	}

	model, _ := query["model"].(string)
	if model == "" {
		model = "sonar"
	}

	limit := 10
	if l, ok := query["limit"].(int); ok && l > 0 {
		limit = l
	}

	includeImages, _ := query["include_images"].(bool)

	// Call Perplexity API to search for documents
	documents, err := pc.searchDocuments(ctx, searchQuery, model, limit, includeImages)
	if err != nil {
		return nil, fmt.Errorf("failed to search Perplexity documents: %w", err)
	}

	// Convert to map format
	result := make([]map[string]interface{}, len(documents))
	for i, doc := range documents {
		result[i] = map[string]interface{}{
			"id":           doc.ID,
			"title":       doc.Title,
			"content":     doc.Content,
			"image_url":   doc.ImageURL,
			"image_base64": doc.ImageBase64,
			"metadata":    doc.Metadata,
			"created_at":  doc.CreatedAt,
		}
	}

	if pc.logger != nil {
		pc.logger.Printf("Extracted %d documents from Perplexity", len(result))
	}

	return result, nil
}

// searchDocuments searches for documents using Perplexity API.
func (pc *PerplexityConnector) searchDocuments(ctx context.Context, query string, model string, limit int, includeImages bool) ([]PerplexityDocument, error) {
	// Perplexity API chat completion endpoint
	url := pc.baseURL + "/chat/completions"

	payload := map[string]interface{}{
		"model": model,
		"messages": []map[string]interface{}{
			{
				"role":    "user",
				"content": query,
			},
		},
		"max_tokens": 4000,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payloadBytes)))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+pc.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := pc.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("Perplexity API returned status %d: %s", resp.StatusCode, string(body))
	}

	var response struct {
		ID      string `json:"id"`
		Model   string `json:"model"`
		Choices []struct {
			Message struct {
				Role    string `json:"role"`
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Citations []struct {
			URL   string `json:"url"`
			Title string `json:"title"`
		} `json:"citations,omitempty"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	// Convert response to documents
	documents := make([]PerplexityDocument, 0)

	// Create document from the main response
	if len(response.Choices) > 0 {
		content := response.Choices[0].Message.Content
		if content != "" {
			doc := PerplexityDocument{
				ID:        fmt.Sprintf("perplexity_%d", time.Now().UnixNano()),
				Title:     query,
				Content:   content,
				CreatedAt: time.Now().UTC().Format(time.RFC3339),
				Metadata: map[string]interface{}{
					"model":     response.Model,
					"query":     query,
					"citations": response.Citations,
				},
			}

			// If images are requested and citations contain image URLs, fetch them
			if includeImages && len(response.Citations) > 0 {
				// Try to fetch first image from citations
				for _, citation := range response.Citations {
					if citation.URL != "" {
						imageData, err := pc.fetchImage(ctx, citation.URL)
						if err == nil && imageData != "" {
							doc.ImageURL = citation.URL
							doc.ImageBase64 = imageData
							break
						}
					}
				}
			}

			documents = append(documents, doc)
		}
	}

	// Limit results
	if len(documents) > limit {
		documents = documents[:limit]
	}

	return documents, nil
}

// fetchImage fetches an image from a URL and returns it as base64.
func (pc *PerplexityConnector) fetchImage(ctx context.Context, imageURL string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, imageURL, nil)
	if err != nil {
		return "", fmt.Errorf("failed to create image request: %w", err)
	}

	resp, err := pc.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to fetch image: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("image fetch returned status %d", resp.StatusCode)
	}

	// Check content type
	contentType := resp.Header.Get("Content-Type")
	if !strings.HasPrefix(contentType, "image/") {
		return "", fmt.Errorf("URL does not point to an image (content-type: %s)", contentType)
	}

	imageData, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read image data: %w", err)
	}

	return base64.StdEncoding.EncodeToString(imageData), nil
}

// Close closes the connection.
func (pc *PerplexityConnector) Close() error {
	if pc.logger != nil {
		pc.logger.Printf("Closing Perplexity connection")
	}
	return nil
}

