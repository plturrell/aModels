package ai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// CWMClient wraps LocalAI API calls to CWM domain
type CWMClient struct {
	baseURL    string
	httpClient *http.Client
	model      string // "0xC0DE-CodeWorldModelAgent" or "cwm"
}

// NewCWMClient creates a new CWM client
func NewCWMClient(localAIURL string) *CWMClient {
	return &CWMClient{
		baseURL: localAIURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
		model: "0xC0DE-CodeWorldModelAgent", // CWM domain in LocalAI
	}
}

// SemanticAnalysis represents semantic understanding of code
type SemanticAnalysis struct {
	Intent          string            `json:"intent"`
	Transformations []string          `json:"transformations"`
	Dependencies    []string          `json:"dependencies"`
	DataFlow        []DataFlowStep    `json:"data_flow"`
	PerformanceNotes []string         `json:"performance_notes"`
	SecurityNotes   []string          `json:"security_notes"`
}

// DataFlowStep represents a step in data flow
type DataFlowStep struct {
	From   string `json:"from"`
	To     string `json:"to"`
	Action string `json:"action"`
}

// Relationship represents an implicit relationship between entities
type Relationship struct {
	Source     string  `json:"source"`
	Target     string  `json:"target"`
	Type       string  `json:"type"`
	Confidence float64 `json:"confidence"`
	Reason     string  `json:"reason"`
}

// Call sends a prompt to CWM and returns the response
func (c *CWMClient) Call(ctx context.Context, prompt string) (string, error) {
	payload := map[string]interface{}{
		"model": c.model,
		"messages": []map[string]interface{}{
			{
				"role":    "system",
				"content": "You are a code analysis assistant. Analyze code and extract semantic information, relationships, and insights.",
			},
			{
				"role":    "user",
				"content": prompt,
			},
		},
		"temperature": 0.2, // Low temperature for deterministic analysis
		"max_tokens":   2000,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.baseURL+"/v1/chat/completions",
		bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("call CWM: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("CWM returned status %d: %s", resp.StatusCode, string(body))
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("decode response: %w", err)
	}

	choices, ok := result["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return "", fmt.Errorf("invalid response format")
	}

	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid choice format")
	}

	message, ok := choice["message"].(map[string]interface{})
	if !ok {
		return "", fmt.Errorf("invalid message format")
	}

	content, ok := message["content"].(string)
	if !ok {
		return "", fmt.Errorf("invalid content format")
	}

	return content, nil
}

// AnalyzeCodeSemantics uses CWM to understand code semantics
func (c *CWMClient) AnalyzeCodeSemantics(ctx context.Context, code string, codeType string) (*SemanticAnalysis, error) {
	prompt := fmt.Sprintf(`
Analyze this %s code and extract semantic information:

%s

Return a JSON object with:
1. "intent": Business logic intent
2. "transformations": Data transformations performed
3. "dependencies": Code dependencies (functions, classes, modules)
4. "data_flow": How data flows through the code
5. "performance_notes": Performance implications
6. "security_notes": Security considerations

Return ONLY valid JSON, no markdown formatting.
`, codeType, code)

	response, err := c.Call(ctx, prompt)
	if err != nil {
		return nil, err
	}

	var analysis SemanticAnalysis
	if err := json.Unmarshal([]byte(response), &analysis); err != nil {
		return nil, fmt.Errorf("parse semantic analysis: %w", err)
	}

	return &analysis, nil
}

// DiscoverImplicitRelationships uses CWM to find relationships between entities
func (c *CWMClient) DiscoverImplicitRelationships(ctx context.Context, entitiesJSON string) ([]Relationship, error) {
	prompt := fmt.Sprintf(`
Given these code entities:
%s

Identify implicit relationships such as:
- Data flow dependencies
- Logical dependencies  
- Business process connections
- Temporal dependencies
- Call/import relationships

Return a JSON array of relationships, each with:
- "source": source entity ID
- "target": target entity ID
- "type": relationship type (DEPENDS_ON, CALLS, IMPORTS, etc.)
- "confidence": confidence score (0-1)
- "reason": explanation

Return ONLY valid JSON array, no markdown formatting.
`, entitiesJSON)

	response, err := c.Call(ctx, prompt)
	if err != nil {
		return nil, err
	}

	var relationships []Relationship
	if err := json.Unmarshal([]byte(response), &relationships); err != nil {
		return nil, fmt.Errorf("parse relationships: %w", err)
	}

	return relationships, nil
}

// GenerateDocumentation uses CWM to generate code documentation
func (c *CWMClient) GenerateDocumentation(ctx context.Context, code string, codeType string) (string, error) {
	prompt := fmt.Sprintf(`
Generate comprehensive documentation for this %s code:

%s

Include:
1. Purpose and functionality
2. Parameters and return values
3. Usage examples
4. Dependencies
5. Notes and warnings

Format as clear, well-structured documentation.
`, codeType, code)

	return c.Call(ctx, prompt)
}

