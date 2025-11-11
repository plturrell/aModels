package sapbdc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// Client represents a client for SAP Business Data Cloud.
type Client struct {
	baseURL      string
	apiToken     string
	httpClient   *http.Client
	logger       *log.Logger
	formationID  string
	datasphereURL string
}

// Config holds configuration for SAP BDC client.
type Config struct {
	BaseURL       string
	APIToken      string
	FormationID  string
	DatasphereURL string
	Timeout       time.Duration
}

// NewClient creates a new SAP Business Data Cloud client.
func NewClient(cfg Config, logger *log.Logger) *Client {
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}

	return &Client{
		baseURL:       cfg.BaseURL,
		apiToken:      cfg.APIToken,
		formationID:   cfg.FormationID,
		datasphereURL: cfg.DatasphereURL,
		httpClient: &http.Client{
			Timeout: cfg.Timeout,
		},
		logger: logger,
	}
}

// DataProduct represents a data product in SAP Business Data Cloud.
type DataProduct struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Category    string            `json:"category"`
	SystemType  string            `json:"system_type"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Status      string            `json:"status"`
	Metadata    map[string]any    `json:"metadata"`
}

// IntelligentApplication represents an intelligent application in SAP Business Data Cloud.
type IntelligentApplication struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Category    string            `json:"category"`
	Version     string            `json:"version"`
	Description string            `json:"description"`
	Status      string            `json:"status"`
	DataProducts []string         `json:"data_products"`
	Content     map[string]any    `json:"content"`
}

// Formation represents a SAP Business Data Cloud formation.
type Formation struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Components  []string          `json:"components"`
	DataSources []string          `json:"data_sources"`
	Status      string            `json:"status"`
}

// Schema represents schema information from SAP Datasphere or SAP HANA Data Lake.
type Schema struct {
	Database    string            `json:"database"`
	Schema      string            `json:"schema"`
	Tables      []TableInfo       `json:"tables"`
	Views       []ViewInfo        `json:"views"`
	Metadata    map[string]any    `json:"metadata"`
}

// TableInfo represents table information.
type TableInfo struct {
	Name        string            `json:"name"`
	Schema      string            `json:"schema"`
	Columns     []ColumnInfo      `json:"columns"`
	PrimaryKeys []string          `json:"primary_keys"`
	ForeignKeys []ForeignKeyInfo  `json:"foreign_keys"`
	Metadata    map[string]any    `json:"metadata"`
}

// ColumnInfo represents column information.
type ColumnInfo struct {
	Name        string            `json:"name"`
	DataType    string            `json:"data_type"`
	Nullable    bool              `json:"nullable"`
	Default     any               `json:"default,omitempty"`
	Comment     string            `json:"comment,omitempty"`
	Metadata    map[string]any    `json:"metadata,omitempty"`
}

// ViewInfo represents view information.
type ViewInfo struct {
	Name        string            `json:"name"`
	Schema      string            `json:"schema"`
	Definition  string            `json:"definition"`
	Columns     []ColumnInfo      `json:"columns"`
	Metadata    map[string]any    `json:"metadata"`
}

// ForeignKeyInfo represents foreign key information.
type ForeignKeyInfo struct {
	Name           string   `json:"name"`
	Column         string   `json:"column"`
	ReferencedTable string  `json:"referenced_table"`
	ReferencedColumn string `json:"referenced_column"`
}

// ListDataProducts lists all available data products in the formation.
func (c *Client) ListDataProducts(ctx context.Context) ([]DataProduct, error) {
	url := fmt.Sprintf("%s/api/v1/formations/%s/data-products", c.baseURL, c.formationID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var result struct {
		DataProducts []DataProduct `json:"data_products"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.DataProducts, nil
}

// GetDataProduct retrieves details of a specific data product.
func (c *Client) GetDataProduct(ctx context.Context, productID string) (*DataProduct, error) {
	url := fmt.Sprintf("%s/api/v1/formations/%s/data-products/%s", c.baseURL, c.formationID, productID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var product DataProduct
	if err := json.NewDecoder(resp.Body).Decode(&product); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &product, nil
}

// ListIntelligentApplications lists all available intelligent applications.
func (c *Client) ListIntelligentApplications(ctx context.Context) ([]IntelligentApplication, error) {
	url := fmt.Sprintf("%s/api/v1/formations/%s/intelligent-applications", c.baseURL, c.formationID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var result struct {
		Applications []IntelligentApplication `json:"intelligent_applications"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Applications, nil
}

// GetFormation retrieves formation details.
func (c *Client) GetFormation(ctx context.Context) (*Formation, error) {
	url := fmt.Sprintf("%s/api/v1/formations/%s", c.baseURL, c.formationID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var formation Formation
	if err := json.NewDecoder(resp.Body).Decode(&formation); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &formation, nil
}

// ExtractSchema extracts schema information from SAP Datasphere or SAP HANA Data Lake.
func (c *Client) ExtractSchema(ctx context.Context, spaceID string, database string) (*Schema, error) {
	// Extract from SAP Datasphere if spaceID is provided
	if spaceID != "" && c.datasphereURL != "" {
		return c.extractDatasphereSchema(ctx, spaceID)
	}
	
	// Otherwise extract from SAP HANA Data Lake (Foundation Service)
	return c.extractHanaDataLakeSchema(ctx, database)
}

// extractDatasphereSchema extracts schema from SAP Datasphere space.
func (c *Client) extractDatasphereSchema(ctx context.Context, spaceID string) (*Schema, error) {
	url := fmt.Sprintf("%s/api/v1/spaces/%s/schema", c.datasphereURL, spaceID)
	
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var schema Schema
	if err := json.NewDecoder(resp.Body).Decode(&schema); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &schema, nil
}

// extractHanaDataLakeSchema extracts schema from SAP HANA Data Lake (Foundation Service).
func (c *Client) extractHanaDataLakeSchema(ctx context.Context, database string) (*Schema, error) {
	// Connect to SAP HANA Data Lake via Foundation Service
	// This would use SAP HANA SQL interface to query metadata
	url := fmt.Sprintf("%s/api/v1/foundation-service/schema", c.baseURL)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	payload := map[string]any{
		"database": database,
	}

	body, _ := json.Marshal(payload)
	req.Body = io.NopCloser(strings.NewReader(string(body)))

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	var schema Schema
	if err := json.NewDecoder(resp.Body).Decode(&schema); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &schema, nil
}

// ActivateDataPackage activates a data package in the formation.
func (c *Client) ActivateDataPackage(ctx context.Context, packageID string) error {
	url := fmt.Sprintf("%s/api/v1/formations/%s/data-packages/%s/activate", c.baseURL, c.formationID, packageID)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	return nil
}

// InstallIntelligentApplication installs an intelligent application.
func (c *Client) InstallIntelligentApplication(ctx context.Context, appID string) error {
	url := fmt.Sprintf("%s/api/v1/formations/%s/intelligent-applications/%s/install", c.baseURL, c.formationID, appID)
	
	req, err := http.NewRequestWithContext(ctx, "POST", url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", c.apiToken))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API error: %d - %s", resp.StatusCode, string(body))
	}

	return nil
}

