package mx3

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client provides methods to interact with the MX3 API
type Client struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

// NewClient creates a new MX3 API client
func NewClient(baseURL, apiKey string) *Client {
	if baseURL == "" {
		baseURL = "https://api.mx3.com/v1" // Default MX3 API URL
	}

	return &Client{
		baseURL: baseURL,
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// doRequest performs an HTTP request to the MX3 API
func (c *Client) doRequest(ctx context.Context, method, path string, body io.Reader, result interface{}) error {
	url := fmt.Sprintf("%s%s", c.baseURL, path)
	req, err := http.NewRequestWithContext(ctx, method, url, body)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("API request failed: %s - %s", resp.Status, string(body))
	}

	if result != nil {
		if err := json.NewDecoder(resp.Body).Decode(result); err != nil {
			return fmt.Errorf("decode response: %w", err)
		}
	}

	return nil
}

// GetAnalysis retrieves analysis results from MX3 API
type AnalysisResult struct {
	ID          string                 `json:"id"`
	Status      string                 `json:"status"`
	Metrics     map[string]interface{} `json:"metrics"`
	CreatedAt   time.Time              `json:"created_at"`
	CompletedAt *time.Time             `json:"completed_at,omitempty"`
}

// GetAnalysis retrieves analysis results by ID
func (c *Client) GetAnalysis(ctx context.Context, analysisID string) (*AnalysisResult, error) {
	var result AnalysisResult
	path := fmt.Sprintf("/analyses/%s", analysisID)
	
	if err := c.doRequest(ctx, http.MethodGet, path, nil, &result); err != nil {
		return nil, fmt.Errorf("get analysis: %w", err)
	}

	return &result, nil
}

// CreateAnalysisRequest represents a request to create a new analysis
type CreateAnalysisRequest struct {
	ModelID     string            `json:"model_id"`
	DatasetID   string            `json:"dataset_id"`
	Parameters  map[string]string `json:"parameters,omitempty"`
	CallbackURL string            `json:"callback_url,omitempty"`
}

// CreateAnalysis starts a new analysis job
func (c *Client) CreateAnalysis(ctx context.Context, req CreateAnalysisRequest) (*AnalysisResult, error) {
	var result AnalysisResult
	path := "/analyses"
	
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	if err := c.doRequest(ctx, http.MethodPost, path, bytes.NewReader(body), &result); err != nil {
		return nil, fmt.Errorf("create analysis: %w", err)
	}

	return &result, nil
}

// ListAnalyses retrieves a list of analyses with optional filters
type ListAnalysesParams struct {
	Status    string    `json:"status,omitempty"`
	ModelID   string    `json:"model_id,omitempty"`
	StartTime time.Time `json:"start_time,omitempty"`
	EndTime   time.Time `json:"end_time,omitempty"`
	Limit     int       `json:"limit,omitempty"`
}

func (c *Client) ListAnalyses(ctx context.Context, params ListAnalysesParams) ([]AnalysisResult, error) {
	var results []AnalysisResult
	path := "/analyses"

	// Convert params to query string
	query := url.Values{}
	if params.Status != "" {
		query.Add("status", params.Status)
	}
	if params.ModelID != "" {
		query.Add("model_id", params.ModelID)
	}
	if !params.StartTime.IsZero() {
		query.Add("start_time", params.StartTime.Format(time.RFC3339))
	}
	if !params.EndTime.IsZero() {
		query.Add("end_time", params.EndTime.Format(time.RFC3339))
	}
	if params.Limit > 0 {
		query.Add("limit", strconv.Itoa(params.Limit))
	}

	if len(query) > 0 {
		path = fmt.Sprintf("%s?%s", path, query.Encode())
	}

	if err := c.doRequest(ctx, http.MethodGet, path, nil, &results); err != nil {
		return nil, fmt.Errorf("list analyses: %w", err)
	}

	return results, nil
}
