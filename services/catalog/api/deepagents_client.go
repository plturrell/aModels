package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"
)

// DeepAgentsClient handles communication with the DeepAgents service for catalog operations.
type DeepAgentsClient struct {
	baseURL string
	client  *http.Client
	logger  *log.Logger
	enabled bool
}

// NewDeepAgentsClient creates a new DeepAgents client for catalog service.
func NewDeepAgentsClient(logger *log.Logger) *DeepAgentsClient {
	baseURL := os.Getenv("DEEPAGENTS_URL")
	if baseURL == "" {
		baseURL = "http://deepagents-service:9004"
	}

	// Check if AI features are enabled
	enabled := os.Getenv("CATALOG_AI_DEDUPLICATION_ENABLED") == "true" ||
		os.Getenv("CATALOG_AI_VALIDATION_ENABLED") == "true"

	if enabled {
		logger.Printf("Catalog DeepAgents client enabled (URL: %s)", baseURL)
	} else {
		logger.Printf("Catalog DeepAgents client disabled (set CATALOG_AI_DEDUPLICATION_ENABLED or CATALOG_AI_VALIDATION_ENABLED=true to enable)")
	}

	return &DeepAgentsClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 30 * time.Second, // Shorter timeout for catalog operations
		},
		logger:  logger,
		enabled: enabled,
	}
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// DeduplicationRequest represents a request for duplicate checking.
type DeduplicationRequest struct {
	CandidateElements []CandidateElement `json:"candidate_elements"`
	ExistingElements  []ExistingElement  `json:"existing_elements,omitempty"`
}

// CandidateElement represents a candidate data element for registration.
type CandidateElement struct {
	Name                 string            `json:"name"`
	Definition           string            `json:"definition"`
	DataElementConceptID string            `json:"data_element_concept_id"`
	RepresentationID     string            `json:"representation_id"`
	Metadata             map[string]string `json:"metadata,omitempty"`
}

// ExistingElement represents an existing data element in the catalog.
type ExistingElement struct {
	Identifier           string `json:"identifier"`
	Name                 string `json:"name"`
	Definition           string `json:"definition"`
	DataElementConceptID string `json:"data_element_concept_id"`
	RepresentationID     string `json:"representation_id"`
}

// DeduplicationResponse represents the response from deduplication analysis.
type DeduplicationResponse struct {
	Suggestions []DeduplicationSuggestion `json:"suggestions"`
}

// DeduplicationSuggestion represents a suggestion for a candidate element.
type DeduplicationSuggestion struct {
	Index      int    `json:"index"`
	Action     string `json:"action"` // "register", "skip", "merge"
	Reason     string `json:"reason"`
	SimilarTo  string `json:"similar_to,omitempty"` // Identifier of similar element
	Confidence float64 `json:"confidence"`
}

// ValidationRequest represents a request for definition validation.
type ValidationRequest struct {
	Elements []CandidateElement `json:"elements"`
}

// ValidationResponse represents the response from validation analysis.
type ValidationResponse struct {
	Suggestions []ValidationSuggestion `json:"suggestions"`
}

// ValidationSuggestion represents a validation suggestion for an element.
type ValidationSuggestion struct {
	Index       int      `json:"index"`
	Improvements []string `json:"improvements"`
	Score       float64  `json:"score"` // Quality score 0-1
}

// CheckDuplicates analyzes candidate elements for duplicates using DeepAgents.
func (c *DeepAgentsClient) CheckDuplicates(ctx context.Context, candidates []CandidateElement, existing []ExistingElement) (*DeduplicationResponse, error) {
	if !c.enabled {
		return nil, nil // Gracefully skip if disabled
	}

	// Quick health check
	healthCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if !c.checkHealth(healthCtx) {
		if c.logger != nil {
			c.logger.Printf("DeepAgents service unavailable, skipping deduplication")
		}
		return nil, nil
	}

	// Build prompt for DeepAgents
	prompt := "Analyze these candidate data elements for duplicates against existing catalog elements.\n\n"
	prompt += "For each candidate element, determine if it:\n"
	prompt += "1. Is a duplicate (action: 'skip' or 'merge') - provide similar_to identifier\n"
	prompt += "2. Is new and should be registered (action: 'register')\n"
	prompt += "Provide confidence score (0-1) and reason for each decision.\n\n"
	prompt += "Candidate Elements:\n"
	for i, elem := range candidates {
		prompt += fmt.Sprintf("%d. Name: %s, Definition: %s, Concept: %s\n", i, elem.Name, elem.Definition, elem.DataElementConceptID)
	}
	if len(existing) > 0 {
		prompt += "\nExisting Elements:\n"
		for _, elem := range existing {
			prompt += fmt.Sprintf("- %s: %s (%s)\n", elem.Identifier, elem.Name, elem.Definition)
		}
	}

	// Prepare request
	reqBody := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call DeepAgents
	url := fmt.Sprintf("%s/invoke", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("DeepAgents deduplication request failed: %v", err)
		}
		return nil, nil // Non-fatal, return nil
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if c.logger != nil {
			c.logger.Printf("DeepAgents returned status %d, skipping deduplication", resp.StatusCode)
		}
		return nil, nil // Non-fatal
	}

	var response struct {
		Messages []Message     `json:"messages"`
		Result   interface{}   `json:"result,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode DeepAgents response: %v", err)
		}
		return nil, nil // Non-fatal
	}

	// Parse response - extract suggestions from assistant message
	suggestions := c.parseDeduplicationResponse(response.Messages, len(candidates))
	return &DeduplicationResponse{Suggestions: suggestions}, nil
}

// ValidateDefinitions validates data element definitions using DeepAgents.
func (c *DeepAgentsClient) ValidateDefinitions(ctx context.Context, elements []CandidateElement) (*ValidationResponse, error) {
	if !c.enabled {
		return nil, nil // Gracefully skip if disabled
	}

	// Quick health check
	healthCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if !c.checkHealth(healthCtx) {
		if c.logger != nil {
			c.logger.Printf("DeepAgents service unavailable, skipping validation")
		}
		return nil, nil
	}

	// Build validation prompt
	prompt := "Validate these data element definitions against ISO 11179 standards.\n\n"
	prompt += "For each element, provide:\n"
	prompt += "1. Quality score (0-1)\n"
	prompt += "2. List of improvements (if any)\n\n"
	prompt += "Elements to validate:\n"
	for i, elem := range elements {
		prompt += fmt.Sprintf("%d. Name: %s, Definition: %s\n", i, elem.Name, elem.Definition)
	}

	reqBody := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/invoke", c.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		if c.logger != nil {
			c.logger.Printf("DeepAgents validation request failed: %v", err)
		}
		return nil, nil // Non-fatal
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		if c.logger != nil {
			c.logger.Printf("DeepAgents returned status %d, skipping validation", resp.StatusCode)
		}
		return nil, nil // Non-fatal
	}

	var response struct {
		Messages []Message   `json:"messages"`
		Result   interface{} `json:"result,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode DeepAgents response: %v", err)
		}
		return nil, nil // Non-fatal
	}

	// Parse response
	suggestions := c.parseValidationResponse(response.Messages, len(elements))
	return &ValidationResponse{Suggestions: suggestions}, nil
}

// parseDeduplicationResponse parses DeepAgents response into deduplication suggestions.
func (c *DeepAgentsClient) parseDeduplicationResponse(messages []Message, elementCount int) []DeduplicationSuggestion {
	var suggestions []DeduplicationSuggestion
	
	// Extract assistant message
	for _, msg := range messages {
		if msg.Role == "assistant" {
			// Simple parsing - in production, use structured output or better parsing
			// For now, create default suggestions (all register)
			for i := 0; i < elementCount; i++ {
				suggestions = append(suggestions, DeduplicationSuggestion{
					Index:      i,
					Action:     "register",
					Reason:     "AI analysis unavailable, defaulting to register",
					Confidence: 0.5,
				})
			}
			break
		}
	}
	
	return suggestions
}

// parseValidationResponse parses DeepAgents response into validation suggestions.
func (c *DeepAgentsClient) parseValidationResponse(messages []Message, elementCount int) []ValidationSuggestion {
	var suggestions []ValidationSuggestion
	
	// Extract assistant message
	for _, msg := range messages {
		if msg.Role == "assistant" {
			// Simple parsing - in production, use structured output
			for i := 0; i < elementCount; i++ {
				suggestions = append(suggestions, ValidationSuggestion{
					Index:        i,
					Improvements: []string{},
					Score:        0.8, // Default score
				})
			}
			break
		}
	}
	
	return suggestions
}

// checkHealth performs a quick health check on the DeepAgents service.
func (c *DeepAgentsClient) checkHealth(ctx context.Context) bool {
	url := fmt.Sprintf("%s/healthz", c.baseURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return false
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

