// Copyright 2025 AgenticAI ETH Contributors
// SPDX-License-Identifier: MIT

package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
    "time"
    "log"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_LocalAI/pkg/types"
)

// LocalAIClient provides a client for interacting with LocalAI endpoints
type LocalAIClient struct {
	Endpoint   string
	ModelName  string
	APIKey     string
	HTTPClient *http.Client
	Timeout    time.Duration
	MaxRetries int
}

// NewLocalAIClient creates a new LocalAI client
func NewLocalAIClient(endpoint, modelName, apiKey string) *LocalAIClient {
	return &LocalAIClient{
		Endpoint:   endpoint,
		ModelName:  modelName,
		APIKey:     apiKey,
		HTTPClient: &http.Client{},
		Timeout:    60 * time.Second,
		MaxRetries: 3,
	}
}

// EmbedText requests text embeddings from LocalAI using an OpenAI-compatible API.
// embedModel can differ from the generation model (e.g., "all-minilm-l6-v2").
func (c *LocalAIClient) EmbedText(ctx context.Context, embedModel string, input string) ([]float64, error) {
	if embedModel == "" {
		embedModel = c.ModelName
	}
	reqBody := map[string]interface{}{
		"model": embedModel,
		"input": input,
	}
	b, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal embeddings request: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, "POST", c.Endpoint+"/v1/embeddings", bytes.NewBuffer(b))
	if err != nil {
		return nil, fmt.Errorf("create embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}
	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("embeddings request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embeddings status %d: %s", resp.StatusCode, string(body))
	}
	var apiResp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&apiResp); err != nil {
		return nil, fmt.Errorf("decode embeddings: %w", err)
	}
	if len(apiResp.Data) == 0 {
		return nil, fmt.Errorf("no embeddings returned")
	}
	return apiResp.Data[0].Embedding, nil
}

// Generate sends a generation request to LocalAI
func (c *LocalAIClient) Generate(ctx context.Context, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	// Construct OpenAI-compatible request
	requestBody := map[string]interface{}{
		"model":             c.ModelName,
		"prompt":            req.Prompt,
		"temperature":       req.Temperature,
		"max_tokens":        req.MaxTokens,
		"top_p":             req.TopP,
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
	}

	if len(req.StopSequences) > 0 {
		requestBody["stop"] = req.StopSequences
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.Endpoint+"/v1/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	// Execute request with retries
	var resp *http.Response
	var lastErr error

	for attempt := 0; attempt < c.MaxRetries; attempt++ {
		resp, lastErr = c.HTTPClient.Do(httpReq)
		if lastErr == nil && resp.StatusCode == http.StatusOK {
			break
		}

		if resp != nil {
			resp.Body.Close()
		}

		if attempt < c.MaxRetries-1 {
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if lastErr != nil {
		return nil, fmt.Errorf("request failed after %d attempts: %w", c.MaxRetries, lastErr)
	}

	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResponse struct {
		Choices []struct {
			Text         string `json:"text"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(apiResponse.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from API")
	}

    log.Printf("LocalAI generation completed model=%s tokens=%d finish_reason=%s",
        c.ModelName, apiResponse.Usage.TotalTokens, apiResponse.Choices[0].FinishReason)

	return &types.GenerateResponse{
		Text:         apiResponse.Choices[0].Text,
		TokensUsed:   apiResponse.Usage.TotalTokens,
		FinishReason: apiResponse.Choices[0].FinishReason,
		Metadata: map[string]interface{}{
			"model":    c.ModelName,
			"endpoint": c.Endpoint,
		},
	}, nil
}

// GenerateChat sends a chat completion request to LocalAI
func (c *LocalAIClient) GenerateChat(ctx context.Context, messages []types.ChatMessage, req *types.GenerateRequest) (*types.GenerateResponse, error) {
	// Construct OpenAI-compatible chat request
	requestBody := map[string]interface{}{
		"model":             c.ModelName,
		"messages":          messages,
		"temperature":       req.Temperature,
		"max_tokens":        req.MaxTokens,
		"top_p":             req.TopP,
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
	}

	if len(req.StopSequences) > 0 {
		requestBody["stop"] = req.StopSequences
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, "POST", c.Endpoint+"/v1/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	// Execute request
	resp, err := c.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	// Parse response
	var apiResponse struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			TotalTokens int `json:"total_tokens"`
		} `json:"usage"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if len(apiResponse.Choices) == 0 {
		return nil, fmt.Errorf("no choices returned from API")
	}

	return &types.GenerateResponse{
		Text:         apiResponse.Choices[0].Message.Content,
		TokensUsed:   apiResponse.Usage.TotalTokens,
		FinishReason: apiResponse.Choices[0].FinishReason,
		Metadata: map[string]interface{}{
			"model":    c.ModelName,
			"endpoint": c.Endpoint,
		},
	}, nil
}

// HealthCheck verifies the LocalAI endpoint is accessible
func (c *LocalAIClient) HealthCheck(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.Endpoint+"/health", nil)
	if err != nil {
		return fmt.Errorf("failed to create health check request: %w", err)
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health check returned status %d", resp.StatusCode)
	}

	return nil
}

// GetHealthStatus returns the current health status
func (c *LocalAIClient) GetHealthStatus() string {
	// Simple implementation - could be enhanced with actual health tracking
	return "healthy"
}

// GetMetrics returns current metrics
func (c *LocalAIClient) GetMetrics() map[string]interface{} {
	return map[string]interface{}{
		"endpoint":    c.Endpoint,
		"model":       c.ModelName,
		"timeout":     c.Timeout,
		"max_retries": c.MaxRetries,
	}
}

// Close closes the client
func (c *LocalAIClient) Close() error {
	// No-op for basic client
	return nil
}

// GetModels retrieves the list of available models from LocalAI
func (c *LocalAIClient) GetModels(ctx context.Context) ([]string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.Endpoint+"/v1/models", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create models request: %w", err)
	}

	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	resp, err := c.HTTPClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("models request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("models request returned status %d", resp.StatusCode)
	}

	var apiResponse struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&apiResponse); err != nil {
		return nil, fmt.Errorf("failed to decode models response: %w", err)
	}

	models := make([]string, len(apiResponse.Data))
	for i, model := range apiResponse.Data {
		models[i] = model.ID
	}

	return models, nil
}
