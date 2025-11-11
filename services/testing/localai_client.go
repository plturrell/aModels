package testing

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/pkg/localai"
)

// LocalAIClient wraps the LocalAI client for the testing service.
type LocalAIClient struct {
	client    *localai.Client
	logger    *log.Logger
	model     string
	enabled   bool
	timeout   time.Duration
	maxRetries int
}

// NewLocalAIClient creates a new LocalAI client wrapper.
func NewLocalAIClient(baseURL, model string, enabled bool, timeout time.Duration, maxRetries int, logger *log.Logger) *LocalAIClient {
	if !enabled || baseURL == "" {
		return &LocalAIClient{
			enabled: false,
			logger:   logger,
		}
	}

	client := localai.NewClient(baseURL)
	return &LocalAIClient{
		client:     client,
		logger:     logger,
		model:      model,
		enabled:    true,
		timeout:    timeout,
		maxRetries: maxRetries,
	}
}

// IsEnabled returns whether LocalAI is enabled.
func (c *LocalAIClient) IsEnabled() bool {
	return c.enabled && c.client != nil
}

// GenerateIntelligentValue generates a value using LocalAI based on column context.
func (c *LocalAIClient) GenerateIntelligentValue(ctx context.Context, tableName, columnName, columnType string, contextInfo map[string]any) (string, error) {
	if !c.IsEnabled() {
		return "", fmt.Errorf("LocalAI is not enabled")
	}

	// Build prompt for value generation
	prompt := c.buildValueGenerationPrompt(tableName, columnName, columnType, contextInfo)

	// Create chat request
	req := &localai.ChatRequest{
		Model:       c.model,
		Messages: []localai.Message{
			{
				Role:    "system",
				Content: "You are a data generation assistant. Generate realistic test data values based on the context provided.",
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   50,
		Temperature: 0.7,
	}

	// Create context with timeout
	ctxWithTimeout, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	// Call LocalAI with retry logic
	var resp *localai.ChatResponse
	var err error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		resp, err = c.client.ChatCompletion(ctxWithTimeout, req)
		if err == nil {
			break
		}
		if attempt < c.maxRetries-1 {
			c.logger.Printf("LocalAI request failed (attempt %d/%d): %v, retrying...", attempt+1, c.maxRetries, err)
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if err != nil {
		return "", fmt.Errorf("LocalAI chat completion failed after %d attempts: %w", c.maxRetries, err)
	}

	if resp == nil || len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty response from LocalAI")
	}

	value := resp.Choices[0].Message.Content
	return value, nil
}

// GenerateQualityRules generates quality rules using LocalAI based on table schema.
func (c *LocalAIClient) GenerateQualityRules(ctx context.Context, schema *TableSchema) ([]QualityRule, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("LocalAI is not enabled")
	}

	// Build prompt for quality rule generation
	prompt := c.buildQualityRulePrompt(schema)

	req := &localai.ChatRequest{
		Model: c.model,
		Messages: []localai.Message{
			{
				Role:    "system",
				Content: "You are a data quality expert. Generate data quality rules in JSON format based on table schemas.",
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   500,
		Temperature: 0.3,
	}

	ctxWithTimeout, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	var resp *localai.ChatResponse
	var err error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		resp, err = c.client.ChatCompletion(ctxWithTimeout, req)
		if err == nil {
			break
		}
		if attempt < c.maxRetries-1 {
			c.logger.Printf("LocalAI quality rule generation failed (attempt %d/%d): %v, retrying...", attempt+1, c.maxRetries, err)
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("LocalAI quality rule generation failed: %w", err)
	}

	if resp == nil || len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from LocalAI")
	}

	// Parse response (simplified - would need proper JSON parsing)
	// For now, return empty rules and log the response
	c.logger.Printf("LocalAI generated quality rules suggestion: %s", resp.Choices[0].Message.Content)
	
	// Return empty rules for now - would need to parse JSON response
	// This is a placeholder for the actual implementation
	return []QualityRule{}, nil
}

// LearnPatterns analyzes data patterns using LocalAI.
func (c *LocalAIClient) LearnPatterns(ctx context.Context, tableName string, sampleData []map[string]any) (map[string]*ColumnPattern, error) {
	if !c.IsEnabled() {
		return nil, fmt.Errorf("LocalAI is not enabled")
	}

	// Build prompt for pattern learning
	prompt := c.buildPatternLearningPrompt(tableName, sampleData)

	req := &localai.ChatRequest{
		Model: c.model,
		Messages: []localai.Message{
			{
				Role:    "system",
				Content: "You are a data pattern analysis expert. Analyze data samples and identify patterns, distributions, and common values.",
			},
			{
				Role:    "user",
				Content: prompt,
			},
		},
		MaxTokens:   1000,
		Temperature: 0.2,
	}

	ctxWithTimeout, cancel := context.WithTimeout(ctx, c.timeout)
	defer cancel()

	var resp *localai.ChatResponse
	var err error
	for attempt := 0; attempt < c.maxRetries; attempt++ {
		resp, err = c.client.ChatCompletion(ctxWithTimeout, req)
		if err == nil {
			break
		}
		if attempt < c.maxRetries-1 {
			c.logger.Printf("LocalAI pattern learning failed (attempt %d/%d): %v, retrying...", attempt+1, c.maxRetries, err)
			time.Sleep(time.Duration(attempt+1) * time.Second)
		}
	}

	if err != nil {
		return nil, fmt.Errorf("LocalAI pattern learning failed: %w", err)
	}

	if resp == nil || len(resp.Choices) == 0 {
		return nil, fmt.Errorf("empty response from LocalAI")
	}

	c.logger.Printf("LocalAI pattern analysis result: %s", resp.Choices[0].Message.Content)
	
	// Return empty patterns for now - would need to parse JSON response
	return make(map[string]*ColumnPattern), nil
}

// buildValueGenerationPrompt builds a prompt for value generation.
func (c *LocalAIClient) buildValueGenerationPrompt(tableName, columnName, columnType string, contextInfo map[string]any) string {
	prompt := fmt.Sprintf("Generate a realistic test data value for:\n")
	prompt += fmt.Sprintf("- Table: %s\n", tableName)
	prompt += fmt.Sprintf("- Column: %s\n", columnName)
	prompt += fmt.Sprintf("- Type: %s\n", columnType)
	
	if len(contextInfo) > 0 {
		prompt += "\nContext:\n"
		for key, value := range contextInfo {
			prompt += fmt.Sprintf("- %s: %v\n", key, value)
		}
	}
	
	prompt += "\nGenerate only the value, no explanation."
	return prompt
}

// buildQualityRulePrompt builds a prompt for quality rule generation.
func (c *LocalAIClient) buildQualityRulePrompt(schema *TableSchema) string {
	prompt := fmt.Sprintf("Generate data quality rules for table: %s\n\n", schema.Name)
	prompt += fmt.Sprintf("Table Type: %s\n", schema.Type)
	prompt += "Columns:\n"
	
	for _, col := range schema.Columns {
		prompt += fmt.Sprintf("- %s (%s)", col.Name, col.Type)
		if !col.Nullable {
			prompt += " [NOT NULL]"
		}
		if col.IsPrimaryKey {
			prompt += " [PRIMARY KEY]"
		}
		if col.IsForeignKey {
			prompt += " [FOREIGN KEY]"
		}
		prompt += "\n"
	}
	
	if len(schema.ForeignKeys) > 0 {
		prompt += "\nForeign Keys:\n"
		for _, fk := range schema.ForeignKeys {
			prompt += fmt.Sprintf("- %s -> %s.%s\n", fk.Column, fk.ReferencedTable, fk.ReferencedColumn)
		}
	}
	
	prompt += "\nGenerate quality rules in JSON format with fields: name, type, rule, severity"
	return prompt
}

// buildPatternLearningPrompt builds a prompt for pattern learning.
func (c *LocalAIClient) buildPatternLearningPrompt(tableName string, sampleData []map[string]any) string {
	prompt := fmt.Sprintf("Analyze data patterns for table: %s\n\n", tableName)
	prompt += fmt.Sprintf("Sample data (%d rows):\n", len(sampleData))
	
	// Include first few rows as examples
	maxRows := 10
	if len(sampleData) < maxRows {
		maxRows = len(sampleData)
	}
	
	for i := 0; i < maxRows; i++ {
		prompt += fmt.Sprintf("Row %d: %v\n", i+1, sampleData[i])
	}
	
	prompt += "\nIdentify patterns, distributions, common values, and enum-like columns."
	return prompt
}

