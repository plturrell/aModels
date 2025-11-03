package transformers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// Client wraps an OpenAI-compatible chat completion endpoint.
type Client struct {
	endpoint   string
	modelName  string
	httpClient *http.Client
}

func NewClient(endpoint, modelName string, timeout time.Duration) *Client {
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	return &Client{
		endpoint:  endpoint,
		modelName: modelName,
		httpClient: &http.Client{
			Timeout: timeout,
		},
	}
}

type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []ChatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature,omitempty"`
	TopP        float64       `json:"top_p,omitempty"`
}

type chatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		TotalTokens int `json:"total_tokens"`
	} `json:"usage"`
}

// Generate sends the messages to the configured chat completions endpoint and returns the text + token usage.
func (c *Client) Generate(ctx context.Context, messages []ChatMessage, maxTokens int, temperature, topP float64) (string, int, error) {
	if c == nil {
		return "", 0, fmt.Errorf("transformers client is nil")
	}
	if len(messages) == 0 {
		return "", 0, fmt.Errorf("messages cannot be empty")
	}
	if maxTokens <= 0 {
		maxTokens = 128
	}
	if temperature < 0 {
		temperature = 0
	}
	if topP <= 0 || topP > 1 {
		topP = 1
	}

	payload := chatCompletionRequest{
		Model:       c.modelName,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: temperature,
		TopP:        topP,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", 0, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return "", 0, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", 0, fmt.Errorf("transformers backend status %d: %s", resp.StatusCode, data)
	}

	var ccResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&ccResp); err != nil {
		return "", 0, fmt.Errorf("decode response: %w", err)
	}
	if len(ccResp.Choices) == 0 {
		return "", 0, fmt.Errorf("transformers backend returned no choices")
	}

	content := ccResp.Choices[0].Message.Content
	return content, ccResp.Usage.TotalTokens, nil
}
