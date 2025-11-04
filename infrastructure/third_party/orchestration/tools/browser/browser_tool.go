package browser

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// BrowserTool implements the Tool interface for browser automation
type BrowserTool struct {
	baseURL    string
	httpClient *http.Client
}

// BrowserRequest represents a request to the browser service
type BrowserRequest struct {
	Action     string                 `json:"action"`
	SessionID  string                 `json:"session_id,omitempty"`
	URL        string                 `json:"url,omitempty"`
	Selector   string                 `json:"selector,omitempty"`
	Text       string                 `json:"text,omitempty"`
	Selectors  map[string]string      `json:"selectors,omitempty"`
	Prompt     string                 `json:"prompt,omitempty"`
	Timeout    int                    `json:"timeout,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
}

// BrowserResponse represents a response from the browser service
type BrowserResponse struct {
	Success bool                   `json:"success"`
	Data    map[string]interface{} `json:"data,omitempty"`
	Message string                 `json:"message,omitempty"`
	Error   string                 `json:"error,omitempty"`
}

// NewBrowserTool creates a new browser tool
func NewBrowserTool(baseURL string) *BrowserTool {
	return &BrowserTool{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Name returns the tool name
func (t *BrowserTool) Name() string {
	return "browser"
}

// Description returns the tool description
func (t *BrowserTool) Description() string {
	return "Browser automation tool for web scraping, navigation, and data extraction with privacy controls"
}

// Args returns the tool arguments schema
func (t *BrowserTool) Args() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"action": map[string]interface{}{
				"type":        "string",
				"description": "Action to perform: navigate, extract, click, type, analyze, screenshot",
				"enum":        []string{"navigate", "extract", "click", "type", "analyze", "screenshot"},
			},
			"url": map[string]interface{}{
				"type":        "string",
				"description": "URL to navigate to (required for navigate action)",
			},
			"selector": map[string]interface{}{
				"type":        "string",
				"description": "CSS selector for element (required for extract, click, type actions)",
			},
			"text": map[string]interface{}{
				"type":        "string",
				"description": "Text to type (required for type action)",
			},
			"selectors": map[string]interface{}{
				"type":        "object",
				"description": "Map of selectors for extraction (required for extract action)",
			},
			"prompt": map[string]interface{}{
				"type":        "string",
				"description": "AI prompt for analysis or extraction",
			},
			"timeout": map[string]interface{}{
				"type":        "integer",
				"description": "Timeout in seconds (optional)",
			},
		},
		"required": []string{"action"},
	}
}

// Call executes the browser tool
func (t *BrowserTool) Call(ctx context.Context, input map[string]interface{}) (interface{}, error) {
	action, ok := input["action"].(string)
	if !ok {
		return nil, fmt.Errorf("action is required")
	}

	// Create request
	req := BrowserRequest{
		Action: action,
	}

	// Extract parameters based on action
	if url, ok := input["url"].(string); ok {
		req.URL = url
	}
	if selector, ok := input["selector"].(string); ok {
		req.Selector = selector
	}
	if text, ok := input["text"].(string); ok {
		req.Text = text
	}
	if selectors, ok := input["selectors"].(map[string]interface{}); ok {
		req.Selectors = make(map[string]string)
		for k, v := range selectors {
			if str, ok := v.(string); ok {
				req.Selectors[k] = str
			}
		}
	}
	if prompt, ok := input["prompt"].(string); ok {
		req.Prompt = prompt
	}
	if timeout, ok := input["timeout"].(float64); ok {
		req.Timeout = int(timeout)
	}

	// Execute the request
	return t.executeRequest(ctx, req)
}

// executeRequest executes a browser request
func (t *BrowserTool) executeRequest(ctx context.Context, req BrowserRequest) (interface{}, error) {
	var endpoint string
	var method string

	// Determine endpoint and method based on action
	switch req.Action {
	case "navigate":
		endpoint = "/api/v1/navigate"
		method = "POST"
	case "extract":
		endpoint = "/api/v1/extract"
		method = "POST"
	case "analyze":
		endpoint = "/api/v1/analyze"
		method = "POST"
	default:
		return nil, fmt.Errorf("unsupported action: %s", req.Action)
	}

	// Prepare request body
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	// Create HTTP request
	httpReq, err := http.NewRequestWithContext(ctx, method, t.baseURL+endpoint, bytes.NewBuffer(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	// Make request
	resp, err := t.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	// Check status code
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("browser service returned status %d", resp.StatusCode)
	}

	// Parse response
	var browserResp BrowserResponse
	if err := json.NewDecoder(resp.Body).Decode(&browserResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if !browserResp.Success {
		return nil, fmt.Errorf("browser operation failed: %s", browserResp.Error)
	}

	return browserResp.Data, nil
}

// Navigate navigates to a URL
func (t *BrowserTool) Navigate(ctx context.Context, url string) (map[string]interface{}, error) {
	req := BrowserRequest{
		Action: "navigate",
		URL:    url,
	}

	result, err := t.executeRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	return result.(map[string]interface{}), nil
}

// Extract extracts data from the current page
func (t *BrowserTool) Extract(ctx context.Context, selectors map[string]string) (map[string]interface{}, error) {
	req := BrowserRequest{
		Action:    "extract",
		Selectors: selectors,
	}

	result, err := t.executeRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	return result.(map[string]interface{}), nil
}

// Click clicks on an element
func (t *BrowserTool) Click(ctx context.Context, selector string) error {
	req := BrowserRequest{
		Action:   "click",
		Selector: selector,
	}

	_, err := t.executeRequest(ctx, req)
	return err
}

// Type types text into an element
func (t *BrowserTool) Type(ctx context.Context, selector, text string) error {
	req := BrowserRequest{
		Action:   "type",
		Selector: selector,
		Text:     text,
	}

	_, err := t.executeRequest(ctx, req)
	return err
}

// Analyze analyzes the current page using AI
func (t *BrowserTool) Analyze(ctx context.Context) (map[string]interface{}, error) {
	req := BrowserRequest{
		Action: "analyze",
	}

	result, err := t.executeRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	return result.(map[string]interface{}), nil
}

// ExtractWithAI extracts data using AI with a custom prompt
func (t *BrowserTool) ExtractWithAI(ctx context.Context, prompt string) (interface{}, error) {
	req := BrowserRequest{
		Action: "extract",
		Prompt: prompt,
	}

	result, err := t.executeRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// BrowserWorkflow represents a multi-step browser workflow
type BrowserWorkflow struct {
	tool   *BrowserTool
	steps  []WorkflowStep
	ctx    context.Context
	cancel context.CancelFunc
}

// WorkflowStep represents a single step in a browser workflow
type WorkflowStep struct {
	ID          string                 `json:"id"`
	Action      string                 `json:"action"`
	Parameters  map[string]interface{} `json:"parameters"`
	Condition   string                 `json:"condition,omitempty"`
	WaitAfter   time.Duration          `json:"wait_after,omitempty"`
	RetryCount  int                    `json:"retry_count,omitempty"`
	Description string                 `json:"description,omitempty"`
}

// NewBrowserWorkflow creates a new browser workflow
func NewBrowserWorkflow(tool *BrowserTool) *BrowserWorkflow {
	ctx, cancel := context.WithCancel(context.Background())
	return &BrowserWorkflow{
		tool:   tool,
		steps:  make([]WorkflowStep, 0),
		ctx:    ctx,
		cancel: cancel,
	}
}

// AddStep adds a step to the workflow
func (w *BrowserWorkflow) AddStep(step WorkflowStep) {
	w.steps = append(w.steps, step)
}

// Execute executes the workflow
func (w *BrowserWorkflow) Execute() ([]map[string]interface{}, error) {
	results := make([]map[string]interface{}, 0, len(w.steps))

	for i, step := range w.steps {
		// Execute step
		result, err := w.executeStep(step)
		if err != nil {
			return results, fmt.Errorf("failed to execute step %d (%s): %w", i+1, step.ID, err)
		}

		results = append(results, map[string]interface{}{
			"step_id":    step.ID,
			"step_index": i + 1,
			"result":     result,
			"success":    true,
		})

		// Wait after step if specified
		if step.WaitAfter > 0 {
			time.Sleep(step.WaitAfter)
		}
	}

	return results, nil
}

// executeStep executes a single workflow step
func (w *BrowserWorkflow) executeStep(step WorkflowStep) (interface{}, error) {
	// Execute step using the tool
	result, err := w.tool.Call(w.ctx, step.Parameters)
	if err != nil {
		// Retry if specified
		if step.RetryCount > 0 {
			for i := 0; i < step.RetryCount; i++ {
				time.Sleep(time.Second) // Wait before retry
				result, err = w.tool.Call(w.ctx, step.Parameters)
				if err == nil {
					break
				}
			}
		}
	}

	return result, err
}

// Cancel cancels the workflow execution
func (w *BrowserWorkflow) Cancel() {
	w.cancel()
}

// BrowserChain represents a browser operation chain
type BrowserChain struct {
	tool    *BrowserTool
	steps   []map[string]interface{}
	context map[string]interface{}
}

// NewBrowserChain creates a new browser chain
func NewBrowserChain(tool *BrowserTool) *BrowserChain {
	return &BrowserChain{
		tool:    tool,
		steps:   make([]map[string]interface{}, 0),
		context: make(map[string]interface{}),
	}
}

// AddStep adds a step to the chain
func (c *BrowserChain) AddStep(step map[string]interface{}) {
	c.steps = append(c.steps, step)
}

// Execute executes the chain
func (c *BrowserChain) Execute(ctx context.Context) ([]interface{}, error) {
	results := make([]interface{}, 0, len(c.steps))

	for i, step := range c.steps {
		// Merge context into step parameters
		stepWithContext := make(map[string]interface{})
		for k, v := range step {
			stepWithContext[k] = v
		}
		for k, v := range c.context {
			if _, exists := stepWithContext[k]; !exists {
				stepWithContext[k] = v
			}
		}

		// Execute step
		result, err := c.tool.Call(ctx, stepWithContext)
		if err != nil {
			return results, fmt.Errorf("failed to execute chain step %d: %w", i+1, err)
		}

		results = append(results, result)

		// Update context with result
		if resultMap, ok := result.(map[string]interface{}); ok {
			for k, v := range resultMap {
				c.context[k] = v
			}
		}
	}

	return results, nil
}

// GetContext returns the current chain context
func (c *BrowserChain) GetContext() map[string]interface{} {
	return c.context
}

// SetContext sets the chain context
func (c *BrowserChain) SetContext(context map[string]interface{}) {
	c.context = context
}

// ClearContext clears the chain context
func (c *BrowserChain) ClearContext() {
	c.context = make(map[string]interface{})
}
