// Package localai implements a LangChainGo LLM adapter for agenticAiETH LocalAI server
package localai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/callbacks"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/llms"
)

var (
	_ llms.Model = (*LLM)(nil)
)

// LLM is a LocalAI LLM implementation.
type LLM struct {
	CallbacksHandler callbacks.Handler
	client           *http.Client
	baseURL          string
	model            string
	temperature      float64
	maxTokens        int
	domains          []string
	autoRouting      bool
}

// Option is a function that configures a LocalAI LLM.
type Option func(*LLM)

// WithBaseURL sets the base URL for the LocalAI server.
func WithBaseURL(url string) Option {
	return func(l *LLM) {
		l.baseURL = url
	}
}

// WithModel sets the model name/domain to use.
func WithModel(model string) Option {
	return func(l *LLM) {
		l.model = model
	}
}

// WithTemperature sets the sampling temperature.
func WithTemperature(temp float64) Option {
	return func(l *LLM) {
		l.temperature = temp
	}
}

// WithMaxTokens sets the maximum number of tokens to generate.
func WithMaxTokens(tokens int) Option {
	return func(l *LLM) {
		l.maxTokens = tokens
	}
}

// WithHTTPClient sets a custom HTTP client.
func WithHTTPClient(client *http.Client) Option {
	return func(l *LLM) {
		l.client = client
	}
}

// WithDomains sets the available domains for routing.
func WithDomains(domains []string) Option {
	return func(l *LLM) {
		l.domains = domains
	}
}

// WithAutoRouting enables automatic domain routing.
func WithAutoRouting(enabled bool) Option {
	return func(l *LLM) {
		l.autoRouting = enabled
	}
}

// New creates a new LocalAI LLM instance.
func New(opts ...Option) (*LLM, error) {
	llm := &LLM{
		baseURL:     "http://localhost:8080",
		model:       "auto",
		temperature: 0.7,
		maxTokens:   500,
		client:      &http.Client{},
		autoRouting: true,
	}

	for _, opt := range opts {
		opt(llm)
	}

	return llm, nil
}

// DomainInfo represents information about an available domain
type DomainInfo struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Models      []string `json:"models"`
	Enabled     bool     `json:"enabled"`
}

// ModelRegistry represents the available domains and models
type ModelRegistry struct {
	Domains []DomainInfo `json:"domains"`
}

// GetModelRegistry retrieves the available domains and models from the LocalAI server
func (l *LLM) GetModelRegistry(ctx context.Context) (*ModelRegistry, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", l.baseURL+"/v1/domains", nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("do request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status %d: %s", resp.StatusCode, string(body))
	}

	var registry ModelRegistry
	if err := json.NewDecoder(resp.Body).Decode(&registry); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &registry, nil
}

// Call implements the llms.Model interface.
func (l *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, l, prompt, options...)
}

// GenerateContent implements the llms.Model interface.
func (l *LLM) GenerateContent(ctx context.Context, messages []llms.MessageContent, options ...llms.CallOption) (*llms.ContentResponse, error) {
	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentStart(ctx, messages)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	// Convert messages to LocalAI chat format
	chatMessages := make([]chatMessage, 0, len(messages))
	for _, msg := range messages {
		role := "user"
		switch msg.Role {
		case llms.ChatMessageTypeSystem:
			role = "system"
		case llms.ChatMessageTypeAI:
			role = "assistant"
		case llms.ChatMessageTypeHuman:
			role = "user"
		}

		var content string
		for _, part := range msg.Parts {
			if textPart, ok := part.(llms.TextContent); ok {
				content += textPart.Text
			}
		}

		chatMessages = append(chatMessages, chatMessage{
			Role:    role,
			Content: content,
		})
	}

	// Build request
	reqBody := chatCompletionRequest{
		Model:       l.model,
		Messages:    chatMessages,
		Temperature: l.temperature,
		MaxTokens:   l.maxTokens,
		Domains:     l.domains,
	}

	if opts.Temperature > 0 {
		reqBody.Temperature = opts.Temperature
	}
	if opts.MaxTokens > 0 {
		reqBody.MaxTokens = opts.MaxTokens
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", l.baseURL+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API error (status %d): %s", resp.StatusCode, string(body))
	}

	var chatResp chatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in response")
	}

	text := chatResp.Choices[0].Message.Content

	response := &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content: text,
				GenerationInfo: map[string]any{
					"model":         chatResp.Model,
					"finish_reason": chatResp.Choices[0].FinishReason,
				},
			},
		},
	}

	if l.CallbacksHandler != nil {
		l.CallbacksHandler.HandleLLMGenerateContentEnd(ctx, response)
	}

	return response, nil
}

// chatMessage represents a message in the chat completion API.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatCompletionRequest represents a request to the chat completions endpoint.
type chatCompletionRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature,omitempty"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Domains     []string      `json:"domains,omitempty"`
}

// chatCompletionResponse represents a response from the chat completions endpoint.
type chatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int         `json:"index"`
		Message      chatMessage `json:"message"`
		FinishReason string      `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}
