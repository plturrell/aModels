package vision

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// DeepSeekClient provides a thin wrapper around the DeepSeek OCR HTTP API.
type DeepSeekClient struct {
	endpoint   string
	apiKey     string
	httpClient *http.Client
}

// DeepSeekConfig configures the OCR client.
type DeepSeekConfig struct {
	Endpoint string
	APIKey   string
	Timeout  time.Duration
}

// NewDeepSeekClient constructs a new client using the supplied configuration.
func NewDeepSeekClient(cfg DeepSeekConfig) *DeepSeekClient {
	timeout := cfg.Timeout
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &DeepSeekClient{
		endpoint:   strings.TrimSpace(cfg.Endpoint),
		apiKey:     strings.TrimSpace(cfg.APIKey),
		httpClient: &http.Client{Timeout: timeout},
	}
}

// ExtractText submits an image (raw bytes) to the DeepSeek OCR backend and returns the textual result.
func (c *DeepSeekClient) ExtractText(ctx context.Context, image []byte, prompt, modelVariant string) (string, error) {
	if c == nil {
		return "", errors.New("deepseek client is nil")
	}
	if len(image) == 0 {
		return "", errors.New("image payload is empty")
	}
	if c.endpoint == "" {
		return "", errors.New("deepseek endpoint not configured")
	}

	reqBody := map[string]any{
		"prompt": prompt,
		"image":  base64.StdEncoding.EncodeToString(image),
	}
	if modelVariant != "" {
		reqBody["model"] = modelVariant
	}

	payload, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	request, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	request.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	resp, err := c.httpClient.Do(request)
	if err != nil {
		return "", fmt.Errorf("call deepseek ocr: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		var buf bytes.Buffer
		if _, err := buf.ReadFrom(resp.Body); err == nil {
			return "", fmt.Errorf("deepseek ocr returned status %d: %s", resp.StatusCode, strings.TrimSpace(buf.String()))
		}
		return "", fmt.Errorf("deepseek ocr returned status %d", resp.StatusCode)
	}

	var parsed map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	if text, ok := parsed["text"].(string); ok && text != "" {
		return text, nil
	}
	if data, ok := parsed["data"].(map[string]any); ok {
		if text, ok := data["text"].(string); ok && text != "" {
			return text, nil
		}
	}
	if outputs, ok := parsed["outputs"].([]any); ok && len(outputs) > 0 {
		if first, ok := outputs[0].(map[string]any); ok {
			if text, ok := first["text"].(string); ok && text != "" {
				return text, nil
			}
		}
	}

	return "", errors.New("deepseek ocr response missing text field")
}
