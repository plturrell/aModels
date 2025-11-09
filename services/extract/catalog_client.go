package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"sync"
	"time"
)

// CatalogClient provides HTTP client for catalog service integration.
type CatalogClient struct {
	baseURL          string
	httpClient       *http.Client
	logger           *log.Logger
	enabled          bool
	maxRetries       int
	retryDelay       time.Duration
	circuitBreaker   *CircuitBreaker
	deepAgentsClient *DeepAgentsClient
	aiEnrichmentEnabled bool
}

// CircuitBreaker implements circuit breaker pattern for resilient service calls.
type CircuitBreaker struct {
	failureCount    int
	lastFailureTime time.Time
	state           string // "closed", "open", "half-open"
	threshold       int
	timeout         time.Duration
	mu              sync.RWMutex
	logger          *log.Logger
}

const (
	circuitStateClosed   = "closed"
	circuitStateOpen     = "open"
	circuitStateHalfOpen = "half-open"
)

// NewCircuitBreaker creates a new circuit breaker.
func NewCircuitBreaker(threshold int, timeout time.Duration, logger *log.Logger) *CircuitBreaker {
	return &CircuitBreaker{
		state:     circuitStateClosed,
		threshold: threshold,
		timeout:   timeout,
		logger:    logger,
	}
}

// Call executes a function with circuit breaker protection.
func (cb *CircuitBreaker) Call(fn func() error) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := time.Now()

	// Check if circuit should transition from open to half-open
	if cb.state == circuitStateOpen {
		if now.Sub(cb.lastFailureTime) >= cb.timeout {
			cb.state = circuitStateHalfOpen
			cb.failureCount = 0
			if cb.logger != nil {
				cb.logger.Printf("Circuit breaker transitioning to half-open")
			}
		} else {
			return fmt.Errorf("circuit breaker is open (too many failures)")
		}
	}

	// Execute the function
	err := fn()

	if err != nil {
		cb.failureCount++
		cb.lastFailureTime = now

		// Transition to open if threshold exceeded
		if cb.failureCount >= cb.threshold {
			if cb.state != circuitStateOpen {
				cb.state = circuitStateOpen
				if cb.logger != nil {
					cb.logger.Printf("Circuit breaker opened after %d failures", cb.failureCount)
				}
			}
		} else if cb.state == circuitStateHalfOpen {
			// Half-open failed, go back to open
			cb.state = circuitStateOpen
			if cb.logger != nil {
				cb.logger.Printf("Circuit breaker returned to open state after half-open failure")
			}
		}
		return err
	}

	// Success - reset failure count and close circuit if needed
	if cb.state == circuitStateHalfOpen {
		cb.state = circuitStateClosed
		if cb.logger != nil {
			cb.logger.Printf("Circuit breaker closed after successful call")
		}
	}
	cb.failureCount = 0

	return nil
}

// State returns the current circuit breaker state.
func (cb *CircuitBreaker) State() string {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.state
}

// Note: DeepAgentsClient type is defined in deepagents.go
// We reference it here for the catalog client integration

// NewCatalogClient creates a new catalog service client.
func NewCatalogClient(baseURL string, logger *log.Logger) *CatalogClient {
	if baseURL == "" {
		return &CatalogClient{
			enabled: false,
			logger:  logger,
		}
	}

	client := &CatalogClient{
		baseURL:       baseURL,
		httpClient:    &http.Client{Timeout: 30 * time.Second},
		logger:        logger,
		enabled:       true,
		maxRetries:    3,
		retryDelay:    1 * time.Second,
		circuitBreaker: NewCircuitBreaker(5, 30*time.Second, logger), // 5 failures, 30s timeout
		aiEnrichmentEnabled: os.Getenv("EXTRACT_AI_ENRICHMENT_ENABLED") == "true",
	}

	// Initialize DeepAgents client if AI enrichment is enabled
	// Note: We use the DeepAgentsClient from deepagents.go
	if client.aiEnrichmentEnabled {
		// Use the existing DeepAgents client from extract service
		client.deepAgentsClient = NewDeepAgentsClient(logger)
		if logger != nil {
			logger.Printf("AI metadata enrichment enabled for catalog client")
		}
	}

	return client
}

// DataElementRequest represents a data element to register in the catalog.
type DataElementRequest struct {
	Name                 string            `json:"name"`
	DataElementConceptID string            `json:"data_element_concept_id"`
	RepresentationID     string            `json:"representation_id"`
	Definition           string            `json:"definition"`
	Identifier           string            `json:"identifier,omitempty"`
	Metadata             map[string]string `json:"metadata,omitempty"`
}

// registerWithRetry performs HTTP request with retry logic and exponential backoff.
func (c *CatalogClient) registerWithRetry(ctx context.Context, url string, jsonData []byte) error {
	if !c.enabled {
		return nil // Silently skip if catalog service not configured
	}

	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			// Exponential backoff: 1s, 2s, 4s
			delay := time.Duration(math.Pow(2, float64(attempt-1))) * c.retryDelay
			if c.logger != nil {
				c.logger.Printf("Retrying catalog service request (attempt %d/%d, delay %v)", attempt+1, c.maxRetries+1, delay)
			}
			
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(delay):
			}
		}

		// Use circuit breaker
		err := c.circuitBreaker.Call(func() error {
			req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
			if err != nil {
				return fmt.Errorf("failed to create request: %w", err)
			}
			req.Header.Set("Content-Type", "application/json")

			resp, err := c.httpClient.Do(req)
			if err != nil {
				return fmt.Errorf("catalog service request failed: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
				// Read response body for error details
				bodyBytes, readErr := io.ReadAll(resp.Body)
				errorBody := string(bodyBytes)
				if readErr != nil {
					errorBody = fmt.Sprintf("(failed to read response body: %v)", readErr)
				}
				
				// Don't retry on 4xx errors (client errors)
				if resp.StatusCode >= 400 && resp.StatusCode < 500 {
					return fmt.Errorf("catalog service returned client error %d: %s", resp.StatusCode, errorBody)
				}
				
				return fmt.Errorf("catalog service returned status %d: %s", resp.StatusCode, errorBody)
			}

			return nil
		})

		if err == nil {
			if attempt > 0 && c.logger != nil {
				c.logger.Printf("Catalog service request succeeded after %d attempts", attempt+1)
			}
			return nil
		}

		lastErr = err
		
		// Check if circuit breaker is open
		if err.Error() == "circuit breaker is open (too many failures)" {
			if c.logger != nil {
				c.logger.Printf("Catalog service circuit breaker is OPEN, skipping request")
			}
			return err
		}
		
		// Check if error message indicates client error (4xx)
		errorMsg := err.Error()
		if contains(errorMsg, "client error") || contains(errorMsg, "status 4") {
			if c.logger != nil {
				c.logger.Printf("Catalog service returned client error, not retrying: %v", err)
			}
			return err
		}
	}

	if c.logger != nil {
		c.logger.Printf("Warning: Failed to register in catalog service after %d attempts: %v", c.maxRetries+1, lastErr)
	}
	return fmt.Errorf("failed after %d attempts: %w", c.maxRetries+1, lastErr)
}

// contains checks if a string contains a substring (simple implementation).
func contains(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// RegisterDataElement registers a single data element in the catalog service.
func (c *CatalogClient) RegisterDataElement(ctx context.Context, element DataElementRequest) error {
	if !c.enabled {
		return nil // Silently skip if catalog service not configured
	}

	jsonData, err := json.Marshal(element)
	if err != nil {
		return fmt.Errorf("failed to marshal data element: %w", err)
	}

	url := fmt.Sprintf("%s/catalog/data-elements", c.baseURL)
	err = c.registerWithRetry(ctx, url, jsonData)
	if err != nil {
		// Log but don't fail extraction if catalog registration fails
		if c.logger != nil {
			state := c.circuitBreaker.State()
			if state == circuitStateOpen {
				c.logger.Printf("Warning: Catalog service circuit breaker is OPEN, skipping registration: %s", element.Identifier)
			} else {
				c.logger.Printf("Warning: Failed to register data element in catalog: %v (element: %s)", err, element.Identifier)
			}
		}
		// Return nil to allow extraction to continue
		return nil
	}

	if c.logger != nil {
		c.logger.Printf("Registered data element in catalog: %s", element.Identifier)
	}

	return nil
}

// RegisterDataElementsBulk registers multiple data elements in the catalog service.
func (c *CatalogClient) RegisterDataElementsBulk(ctx context.Context, elements []DataElementRequest) error {
	if !c.enabled {
		return nil // Silently skip if catalog service not configured
	}

	if len(elements) == 0 {
		return nil
	}

	jsonData, err := json.Marshal(elements)
	if err != nil {
		return fmt.Errorf("failed to marshal data elements: %w", err)
	}

	url := fmt.Sprintf("%s/catalog/data-elements/bulk", c.baseURL)
	err = c.registerWithRetry(ctx, url, jsonData)
	if err != nil {
		// Log but don't fail extraction if catalog registration fails
		if c.logger != nil {
			state := c.circuitBreaker.State()
			if state == circuitStateOpen {
				c.logger.Printf("Warning: Catalog service circuit breaker is OPEN, skipping bulk registration of %d elements", len(elements))
			} else {
				c.logger.Printf("Warning: Failed to register %d data elements in catalog: %v", len(elements), err)
			}
		}
		// Return nil to allow extraction to continue
		return nil
	}

	if c.logger != nil {
		c.logger.Printf("Registered %d data elements in catalog", len(elements))
	}

	return nil
}

// enrichMetadataWithAI enriches node metadata using DeepAgents.
func (c *CatalogClient) enrichMetadataWithAI(ctx context.Context, node Node, projectID, systemID string) (*DataElementRequest, error) {
	if c.deepAgentsClient == nil || !c.aiEnrichmentEnabled {
		return nil, nil // Not enabled, return nil to use basic conversion
	}

	// Build context for AI analysis
	contextStr := fmt.Sprintf("Node: %s (Type: %s, Label: %s)", node.ID, node.Type, node.Label)
	if node.Props != nil {
		contextStr += "\nProperties: "
		for k, v := range node.Props {
			contextStr += fmt.Sprintf("%s=%v; ", k, v)
		}
	}

	// Call DeepAgents for analysis
	analysis, err := c.deepAgentsClient.AnalyzeKnowledgeGraph(ctx, contextStr, projectID, systemID)
	if err != nil || analysis == nil {
		// Non-fatal - return nil to use basic conversion
		if c.logger != nil {
			c.logger.Printf("AI enrichment failed for node %s: %v", node.ID, err)
		}
		return nil, nil
	}

	// For now, return nil to fall back to basic conversion
	// In a full implementation, we'd parse the analysis result to enhance definition, concept, etc.
	return nil, nil
}

// ConvertNodeToDataElementWithAI converts with optional AI enrichment.
func (c *CatalogClient) ConvertNodeToDataElementWithAI(ctx context.Context, node Node, projectID, systemID string) DataElementRequest {
	// Try AI enrichment first
	if c.aiEnrichmentEnabled && c.deepAgentsClient != nil {
		enriched, err := c.enrichMetadataWithAI(ctx, node, projectID, systemID)
		if err == nil && enriched != nil {
			enriched.Metadata["ai_enriched"] = "true"
			return *enriched
		}
		// Fall through to basic conversion if enrichment failed
	}

	// Basic conversion
	element := ConvertNodeToDataElement(node, projectID, systemID)
	if c.aiEnrichmentEnabled {
		element.Metadata["ai_enrichment_attempted"] = "true"
	}
	return element
}

// ConvertNodeToDataElement converts an extracted node to a catalog data element request.
func ConvertNodeToDataElement(node Node, projectID, systemID string) DataElementRequest {
	// Generate identifier from node ID
	identifier := fmt.Sprintf("http://amodels.org/catalog/data-element/%s", node.ID)

	// Create concept ID
	conceptID := fmt.Sprintf("http://amodels.org/catalog/concept/%s", node.Type)

	// Create representation ID based on node type
	representationID := fmt.Sprintf("http://amodels.org/catalog/representation/%s", node.Type)

	// Build definition
	definition := node.Label
	if node.Props != nil {
		if desc, ok := node.Props["description"].(string); ok && desc != "" {
			definition = desc
		}
	}

	// Build metadata
	metadata := make(map[string]string)
	metadata["source"] = "extract-service"
	metadata["node_type"] = node.Type
	if projectID != "" {
		metadata["project_id"] = projectID
	}
	if systemID != "" {
		metadata["system_id"] = systemID
	}

	// Add node properties as metadata
	if node.Props != nil {
		for k, v := range node.Props {
			if k == "description" {
				continue // Already in definition
			}
			// Convert value to string
			if strVal, ok := v.(string); ok {
				metadata[k] = strVal
			} else {
				metadata[k] = fmt.Sprintf("%v", v)
			}
		}
	}

	return DataElementRequest{
		Name:                 node.Label,
		DataElementConceptID: conceptID,
		RepresentationID:     representationID,
		Definition:           definition,
		Identifier:           identifier,
		Metadata:             metadata,
	}
}
