package api

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
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
	cache   CacheInterface
	cacheTTL time.Duration
}

// NewDeepAgentsClient creates a new DeepAgents client for catalog service.
func NewDeepAgentsClient(logger *log.Logger) *DeepAgentsClient {
	baseURL := os.Getenv("DEEPAGENTS_URL")
	if baseURL == "" {
		baseURL = "http://deepagents-service:9004"
	}

	// AI features are enabled by default (opt-out instead of opt-in)
	// Set CATALOG_AI_DISABLED=true to disable all AI features
	disabled := os.Getenv("CATALOG_AI_DISABLED") == "true"
	enabled := !disabled

	if enabled {
		logger.Printf("Catalog DeepAgents client enabled (URL: %s)", baseURL)
	} else {
		logger.Printf("Catalog DeepAgents client disabled (CATALOG_AI_DISABLED=true)")
	}

	cacheTTL := 5 * time.Minute // Default cache TTL
	if ttlStr := os.Getenv("CATALOG_AI_CACHE_TTL"); ttlStr != "" {
		if parsed, err := time.ParseDuration(ttlStr); err == nil {
			cacheTTL = parsed
		}
	}

	return &DeepAgentsClient{
		baseURL: baseURL,
		client: &http.Client{
			Timeout: 30 * time.Second, // Shorter timeout for catalog operations
		},
		logger:  logger,
		enabled: enabled,
		cacheTTL: cacheTTL,
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

// SetCache sets the cache for the client.
func (c *DeepAgentsClient) SetCache(cache CacheInterface) {
	c.cache = cache
}

// generateCacheKey generates a cache key from candidates and existing elements.
func (c *DeepAgentsClient) generateCacheKey(prefix string, candidates []CandidateElement, existing []ExistingElement) string {
	// Create a hash of the input data
	hash := sha256.New()
	
	// Include candidates
	for _, cand := range candidates {
		hash.Write([]byte(cand.Name))
		hash.Write([]byte(cand.Definition))
		hash.Write([]byte(cand.DataElementConceptID))
		hash.Write([]byte(cand.RepresentationID))
	}
	
	// Include existing elements if provided
	if existing != nil {
		for _, exist := range existing {
			hash.Write([]byte(exist.Identifier))
			hash.Write([]byte(exist.Name))
			hash.Write([]byte(exist.Definition))
		}
	}
	
	hashSum := hash.Sum(nil)
	return fmt.Sprintf("deepagents:%s:%s", prefix, hex.EncodeToString(hashSum[:16]))
}

// CheckDuplicates analyzes candidate elements for duplicates using DeepAgents.
func (c *DeepAgentsClient) CheckDuplicates(ctx context.Context, candidates []CandidateElement, existing []ExistingElement) (*DeduplicationResponse, error) {
	if !c.enabled {
		return nil, nil // Gracefully skip if disabled
	}

	// Check cache first
	if c.cache != nil {
		cacheKey := c.generateCacheKey("deduplication", candidates, existing)
		var cachedResponse DeduplicationResponse
		if err := c.cache.Get(ctx, cacheKey, &cachedResponse); err == nil {
			if c.logger != nil {
				c.logger.Printf("Using cached deduplication result for %d candidates", len(candidates))
			}
			return &cachedResponse, nil
		}
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

	// Build prompt for DeepAgents - instruct to use check_duplicates tool
	prompt := "Use the check_duplicates tool to analyze these candidate data elements for duplicates.\n\n"
	prompt += "Candidate Elements:\n"
	candidateElementsJSON, _ := json.Marshal(candidates)
	prompt += fmt.Sprintf("Candidate elements JSON: %s\n", string(candidateElementsJSON))
	if len(existing) > 0 {
		existingElementsJSON, _ := json.Marshal(existing)
		prompt += fmt.Sprintf("\nExisting elements JSON: %s\n", string(existingElementsJSON))
	}
	prompt += "\nPlease use the check_duplicates tool with the candidate_elements and existing_elements parameters."

	// Define JSON schema for structured output
	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"suggestions": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"index":      map[string]interface{}{"type": "integer"},
						"action":     map[string]interface{}{"type": "string", "enum": []string{"register", "skip", "merge"}},
						"reason":     map[string]interface{}{"type": "string"},
						"similar_to": map[string]interface{}{"type": "string"},
						"confidence": map[string]interface{}{"type": "number", "minimum": 0, "maximum": 1},
					},
					"required": []string{"index", "action", "reason", "confidence"},
				},
			},
		},
		"required": []string{"suggestions"},
	}

	// Prepare structured request
	reqBody := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
		"response_format": map[string]interface{}{
			"type":       "json_schema",
			"json_schema": jsonSchema,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Call DeepAgents structured endpoint
	url := fmt.Sprintf("%s/invoke/structured", c.baseURL)
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
		Messages         []Message              `json:"messages"`
		StructuredOutput map[string]interface{} `json:"structured_output"`
		ValidationErrors []string               `json:"validation_errors,omitempty"`
		Result           interface{}            `json:"result,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode DeepAgents response: %v", err)
		}
		return nil, nil // Non-fatal
	}

	// Extract structured output
	var result *DeduplicationResponse
	if response.StructuredOutput != nil {
		if suggestionsData, ok := response.StructuredOutput["suggestions"].([]interface{}); ok {
			var suggestions []DeduplicationSuggestion
			for _, s := range suggestionsData {
				if sMap, ok := s.(map[string]interface{}); ok {
					suggestion := DeduplicationSuggestion{
						Index:      int(sMap["index"].(float64)),
						Action:     sMap["action"].(string),
						Reason:     sMap["reason"].(string),
						Confidence: sMap["confidence"].(float64),
					}
					if similarTo, ok := sMap["similar_to"].(string); ok && similarTo != "" {
						suggestion.SimilarTo = similarTo
					}
					suggestions = append(suggestions, suggestion)
				}
			}
			result = &DeduplicationResponse{Suggestions: suggestions}
		}
	}

	// Fallback: if structured output not available, return empty suggestions
	if result == nil {
		if c.logger != nil {
			c.logger.Printf("No structured output in DeepAgents response, returning empty suggestions")
		}
		result = &DeduplicationResponse{Suggestions: []DeduplicationSuggestion{}}
	}

	// Cache the result
	if c.cache != nil && result != nil {
		cacheKey := c.generateCacheKey("deduplication", candidates, existing)
		if err := c.cache.Set(ctx, cacheKey, result, c.cacheTTL); err != nil && c.logger != nil {
			c.logger.Printf("Failed to cache deduplication result: %v", err)
		}
	}

	return result, nil
}

// ValidateDefinitions validates data element definitions using DeepAgents.
func (c *DeepAgentsClient) ValidateDefinitions(ctx context.Context, elements []CandidateElement) (*ValidationResponse, error) {
	if !c.enabled {
		return nil, nil // Gracefully skip if disabled
	}

	// Check cache first
	if c.cache != nil {
		cacheKey := c.generateCacheKey("validation", elements, nil)
		var cachedResponse ValidationResponse
		if err := c.cache.Get(ctx, cacheKey, &cachedResponse); err == nil {
			if c.logger != nil {
				c.logger.Printf("Using cached validation result for %d elements", len(elements))
			}
			return &cachedResponse, nil
		}
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

	// Build validation prompt - instruct to use validate_definition tool
	prompt := "Use the validate_definition tool to validate these data element definitions against ISO 11179 standards.\n\n"
	elementsJSON, _ := json.Marshal(elements)
	prompt += fmt.Sprintf("Elements to validate (JSON): %s\n", string(elementsJSON))
	prompt += "\nFor each element, please use the validate_definition tool with the name and definition parameters."

	// Define JSON schema for structured output
	jsonSchema := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"suggestions": map[string]interface{}{
				"type": "array",
				"items": map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"index": map[string]interface{}{
							"type": "integer",
						},
						"score": map[string]interface{}{
							"type":    "number",
							"minimum": 0,
							"maximum": 1,
						},
						"improvements": map[string]interface{}{
							"type": "array",
							"items": map[string]interface{}{
								"type": "string",
							},
						},
					},
					"required": []string{"index", "score"},
				},
			},
		},
		"required": []string{"suggestions"},
	}

	reqBody := map[string]interface{}{
		"messages": []Message{
			{Role: "user", Content: prompt},
		},
		"response_format": map[string]interface{}{
			"type":       "json_schema",
			"json_schema": jsonSchema,
		},
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/invoke/structured", c.baseURL)
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
		Messages         []Message              `json:"messages"`
		StructuredOutput map[string]interface{} `json:"structured_output"`
		ValidationErrors []string               `json:"validation_errors,omitempty"`
		Result           interface{}            `json:"result,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&response); err != nil {
		if c.logger != nil {
			c.logger.Printf("Failed to decode DeepAgents response: %v", err)
		}
		return nil, nil // Non-fatal
	}

	// Extract structured output
	var result *ValidationResponse
	if response.StructuredOutput != nil {
		if suggestionsData, ok := response.StructuredOutput["suggestions"].([]interface{}); ok {
			var suggestions []ValidationSuggestion
			for _, s := range suggestionsData {
				if sMap, ok := s.(map[string]interface{}); ok {
					suggestion := ValidationSuggestion{
						Index:  int(sMap["index"].(float64)),
						Score:  sMap["score"].(float64),
						Improvements: []string{},
					}
					if improvements, ok := sMap["improvements"].([]interface{}); ok {
						for _, imp := range improvements {
							if impStr, ok := imp.(string); ok {
								suggestion.Improvements = append(suggestion.Improvements, impStr)
							}
						}
					}
					suggestions = append(suggestions, suggestion)
				}
			}
			result = &ValidationResponse{Suggestions: suggestions}
		}
	}

	// Fallback: if structured output not available, return empty suggestions
	if result == nil {
		if c.logger != nil {
			c.logger.Printf("No structured output in DeepAgents response, returning empty suggestions")
		}
		result = &ValidationResponse{Suggestions: []ValidationSuggestion{}}
	}

	// Cache the result
	if c.cache != nil && result != nil {
		cacheKey := c.generateCacheKey("validation", elements, nil)
		if err := c.cache.Set(ctx, cacheKey, result, c.cacheTTL); err != nil && c.logger != nil {
			c.logger.Printf("Failed to cache validation result: %v", err)
		}
	}

	return result, nil
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

