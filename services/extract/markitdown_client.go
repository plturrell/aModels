package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/plturrell/aModels/services/extract/httpclient"
)

// MarkItDownClient provides HTTP client for markitdown service integration.
type MarkItDownClient struct {
	baseURL          string
	httpClient       *httpclient.Client
	logger           *log.Logger
	enabled          bool
	fallbackToOCR    bool
	metricsCollector func(service, endpoint string, statusCode int, latency time.Duration, correlationID string)
}

// MarkItDownResponse represents the response from markitdown service.
type MarkItDownResponse struct {
	TextContent string            `json:"text_content"`
	Metadata    map[string]string `json:"metadata,omitempty"`
	Format      string            `json:"format,omitempty"`
	Error       string            `json:"error,omitempty"`
}

// MarkItDownRequest represents a request to markitdown service.
type MarkItDownRequest struct {
	FilePath    string `json:"file_path,omitempty"`
	FileContent []byte `json:"file_content,omitempty"`
	Format      string `json:"format,omitempty"` // Optional: specify output format
}

// NewMarkItDownClient creates a new markitdown service client.
func NewMarkItDownClient(
	baseURL string,
	logger *log.Logger,
	metricsCollector func(service, endpoint string, statusCode int, latency time.Duration, correlationID string),
) *MarkItDownClient {
	if baseURL == "" {
		baseURL = os.Getenv("MARKITDOWN_SERVICE_URL")
		if baseURL == "" {
			baseURL = "http://markitdown:8080"
		}
	}

	enabled := os.Getenv("MARKITDOWN_ENABLED")
	if enabled == "" {
		enabled = "true"
	}

	fallbackToOCR := os.Getenv("MARKITDOWN_FALLBACK_TO_OCR")
	if fallbackToOCR == "" {
		fallbackToOCR = "true"
	}

	timeoutStr := os.Getenv("MARKITDOWN_TIMEOUT")
	timeout := 60 * time.Second
	if timeoutStr != "" {
		if parsed, err := time.ParseDuration(timeoutStr); err == nil {
			timeout = parsed
		}
	}

	var client *httpclient.Client
	if enabled == "true" {
		client = httpclient.NewClient(httpclient.ClientConfig{
			Timeout:         timeout,
			MaxRetries:      3,
			InitialBackoff: 1 * time.Second,
			MaxBackoff:      10 * time.Second,
			BaseURL:         baseURL,
			HealthCheckPath: "/healthz",
			Logger:          logger,
			MetricsCollector: func(service, endpoint string, statusCode int, latency time.Duration, correlationID string) {
				if metricsCollector != nil {
					metricsCollector(service, endpoint, statusCode, latency, correlationID)
				}
			},
		})
	}

	return &MarkItDownClient{
		baseURL:          baseURL,
		httpClient:       client,
		logger:           logger,
		enabled:          enabled == "true",
		fallbackToOCR:    fallbackToOCR == "true",
		metricsCollector: metricsCollector,
	}
}

// ConvertFile converts a file to markdown using markitdown service.
func (c *MarkItDownClient) ConvertFile(ctx context.Context, filePath string) (*MarkItDownResponse, error) {
	if !c.enabled {
		return nil, fmt.Errorf("markitdown service not enabled")
	}

	if c.httpClient == nil {
		return nil, fmt.Errorf("markitdown HTTP client not initialized")
	}

	// Read file
	fileData, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	return c.ConvertBytes(ctx, fileData, filepath.Ext(filePath))
}

// ConvertBytes converts file bytes to markdown using markitdown service.
func (c *MarkItDownClient) ConvertBytes(ctx context.Context, fileData []byte, fileExtension string) (*MarkItDownResponse, error) {
	if !c.enabled {
		return nil, fmt.Errorf("markitdown service not enabled")
	}

	if c.httpClient == nil {
		return nil, fmt.Errorf("markitdown HTTP client not initialized")
	}

	// Create multipart form request
	var requestBody bytes.Buffer
	writer := multipart.NewWriter(&requestBody)

	// Add file
	part, err := writer.CreateFormFile("file", "document"+fileExtension)
	if err != nil {
		return nil, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := part.Write(fileData); err != nil {
		return nil, fmt.Errorf("failed to write file data: %w", err)
	}

	// Add format (optional)
	if err := writer.WriteField("format", "markdown"); err != nil {
		return nil, fmt.Errorf("failed to write format field: %w", err)
	}

	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("failed to close multipart writer: %w", err)
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/convert", &requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	// Add correlation ID - extract from context or generate new
	var correlationID string
	if id, ok := ctx.Value(httpclient.CorrelationIDKey).(string); ok && id != "" {
		correlationID = id
	} else {
		correlationID = fmt.Sprintf("%d", time.Now().UnixNano())
	}
	req.Header.Set("X-Request-ID", correlationID)

	// Validate response structure
	validator := func(data map[string]interface{}) error {
		if _, ok := data["text_content"]; !ok {
			return fmt.Errorf("response missing 'text_content' field")
		}
		return nil
	}

	var responseData map[string]interface{}
	err = c.httpClient.DoJSON(req, &responseData, validator)
	if err != nil {
		return nil, fmt.Errorf("markitdown conversion failed: %w", err)
	}

	// Parse response
	var response MarkItDownResponse
	if textContent, ok := responseData["text_content"].(string); ok {
		response.TextContent = textContent
	}
	if metadata, ok := responseData["metadata"].(map[string]interface{}); ok {
		response.Metadata = make(map[string]string)
		for k, v := range metadata {
			if str, ok := v.(string); ok {
				response.Metadata[k] = str
			}
		}
	}
	if format, ok := responseData["format"].(string); ok {
		response.Format = format
	}
	if errMsg, ok := responseData["error"].(string); ok {
		response.Error = errMsg
	}

	if response.Error != "" {
		return nil, fmt.Errorf("markitdown error: %s", response.Error)
	}

	if c.logger != nil {
		c.logger.Printf("[%s] MarkItDown conversion successful (format: %s, length: %d)", 
			correlationID, response.Format, len(response.TextContent))
	}

	return &response, nil
}

// IsFormatSupported checks if a file format is supported by markitdown.
func (c *MarkItDownClient) IsFormatSupported(fileExtension string) bool {
	if !c.enabled {
		return false
	}

	supportedFormats := map[string]bool{
		".pdf":  true,
		".docx": true,
		".doc":  true,
		".xlsx": true,
		".xls":  true,
		".pptx": true,
		".ppt":  true,
		".html": true,
		".htm":  true,
		".csv":  true,
		".json": true,
		".xml":  true,
		".epub": true,
		".jpg":  true,
		".jpeg": true,
		".png":  true,
		".gif":  true,
		".bmp":  true,
		".tiff": true,
		".wav":  true,
		".mp3":  true,
		".m4a":  true,
		".zip":  true,
	}

	return supportedFormats[fileExtension]
}

// HealthCheck checks if markitdown service is healthy.
func (c *MarkItDownClient) HealthCheck(ctx context.Context) (bool, error) {
	if !c.enabled || c.httpClient == nil {
		return false, nil
	}

	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/healthz", nil)
	if err != nil {
		return false, err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	return resp.StatusCode == 200, nil
}

