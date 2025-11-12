package boolq

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// SAPAIClient wraps SAP AI Core for model inference via direct HTTP calls.
type SAPAIClient struct {
	DeploymentID string
	ModelName    string // "gemmavault" or "phimini"
	APIEndpoint  string
	AuthURL      string
	ClientID     string
	ClientSecret string

	token       string
	tokenExpiry time.Time
	mu          sync.RWMutex
}

// AuthResponse is the structure of the token response from the auth server.
type AuthResponse struct {
	AccessToken string `json:"access_token"`
	ExpiresIn   int    `json:"expires_in"`
}

// NewSAPAIClient creates a client for SAP AI Core.
func NewSAPAIClient(deploymentID, modelName string) *SAPAIClient {
	return &SAPAIClient{
		DeploymentID: deploymentID,
		ModelName:    modelName,
		APIEndpoint:  os.Getenv("AICORE_BASE_URL"),
		AuthURL:      os.Getenv("AICORE_AUTH_URL"),
		ClientID:     os.Getenv("AICORE_CLIENT_ID"),
		ClientSecret: os.Getenv("AICORE_CLIENT_SECRET"),
	}
}

// getToken handles acquisition and caching of the OAuth token.
func (c *SAPAIClient) getToken(ctx context.Context) (string, error) {
	c.mu.RLock()
	if time.Now().Before(c.tokenExpiry) {
		token := c.token
		c.mu.RUnlock()
		return token, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()

	// Double-check after acquiring write lock
	if time.Now().Before(c.tokenExpiry) {
		return c.token, nil
	}

	data := url.Values{}
	data.Set("grant_type", "client_credentials")
	data.Set("client_id", c.ClientID)
	data.Set("client_secret", c.ClientSecret)

	req, err := http.NewRequestWithContext(ctx, "POST", c.AuthURL, strings.NewReader(data.Encode()))
	if err != nil {
		return "", fmt.Errorf("failed to create auth request: %w", err)
	}
	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("auth request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("auth failed with status %d: %s", resp.StatusCode, string(body))
	}

	var authResp AuthResponse
	if err := json.NewDecoder(resp.Body).Decode(&authResp); err != nil {
		return "", fmt.Errorf("failed to decode auth response: %w", err)
	}

	c.token = authResp.AccessToken
	c.tokenExpiry = time.Now().Add(time.Duration(authResp.ExpiresIn-60) * time.Second) // Refresh 60s before expiry

	return c.token, nil
}

// InferenceRequest defines the payload for the AI Core inference endpoint.
type InferenceRequest struct {
	Messages []struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"messages"`
	MaxTokens int `json:"max_tokens"`
}

// InferenceResponse defines the structure of the response from the AI Core.
type InferenceResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// GenerateCompletion calls SAP AI Core for text generation.
func (c *SAPAIClient) GenerateCompletion(ctx context.Context, prompt string, maxTokens int) (string, error) {
	token, err := c.getToken(ctx)
	if err != nil {
		return "", err
	}

	inferenceURL := fmt.Sprintf("%s/v2/inference/deployments/%s/chat/completions?api-version=2023-05-15", c.APIEndpoint, c.DeploymentID)

	payload := InferenceRequest{
		Messages: []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{
			{Role: "user", Content: prompt},
		},
		MaxTokens: maxTokens,
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal inference request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", inferenceURL, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create inference request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("AI-Resource-Group", "default") // Or from config

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("inference request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("inference failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	var inferenceResp InferenceResponse
	if err := json.NewDecoder(resp.Body).Decode(&inferenceResp); err != nil {
		return "", fmt.Errorf("failed to decode inference response: %w", err)
	}

	if len(inferenceResp.Choices) == 0 {
		return "", fmt.Errorf("no choices returned from AI Core")
	}

	return inferenceResp.Choices[0].Message.Content, nil
}

// Helper functions for prompt parsing
func extractPassageFromPrompt(prompt string) string {
	// Extract passage between "Passage:" and "Generate"
	start := strings.Index(prompt, "Passage:")
	end := strings.Index(prompt, "Generate")

	if start == -1 || end == -1 || start >= end {
		return prompt
	}

	passage := prompt[start+8 : end]
	return strings.TrimSpace(passage)
}

func extractMainSubject(passage string) string {
	// Extract likely subject (first capitalized word or noun)
	words := strings.Fields(passage)
	for _, word := range words {
		if len(word) > 0 && word[0] >= 'A' && word[0] <= 'Z' {
			return strings.ToLower(word)
		}
	}
	return "the subject"
}

func extractMainAction(passage string) string {
	// Extract main verb/action
	actions := []string{"achieved", "reported", "increased", "decreased", "improved",
		"developed", "created", "implemented", "completed", "demonstrated"}

	passageLower := strings.ToLower(passage)
	for _, action := range actions {
		if strings.Contains(passageLower, action) {
			return action + " the goal"
		}
	}
	return "the action occurred"
}

func extractImplication(passage string) string {
	// Generate plausible implication
	if strings.Contains(strings.ToLower(passage), "profit") || strings.Contains(strings.ToLower(passage), "success") {
		return "performance was positive"
	}
	if strings.Contains(strings.ToLower(passage), "loss") || strings.Contains(strings.ToLower(passage), "decline") {
		return "there were challenges"
	}
	return "this represents a significant development"
}

func extractKeyEntity(passage string) string {
	// Extract first named entity
	words := strings.Fields(passage)
	for i, word := range words {
		if len(word) > 0 && word[0] >= 'A' && word[0] <= 'Z' {
			// Check if it's likely a proper noun (not sentence start)
			if i > 0 {
				return word
			}
		}
	}
	return "the entity"
}

func extractSummary(passage string) string {
	// Generate brief summary
	words := strings.Fields(passage)
	if len(words) > 10 {
		return strings.Join(words[:10], " ") + "..."
	}
	return passage
}

// SAPAIQuestionGenerator uses SAP AI Core for question generation
type SAPAIQuestionGenerator struct {
	Client    *SAPAIClient
	MaxTokens int
}

func NewSAPAIQuestionGenerator(deploymentID, modelName string) *SAPAIQuestionGenerator {
	return &SAPAIQuestionGenerator{
		Client:    NewSAPAIClient(deploymentID, modelName),
		MaxTokens: 500,
	}
}

func (g *SAPAIQuestionGenerator) GenerateQuestions(passage string, count int) ([]QuestionCandidate, error) {
	ctx := context.Background()

	// Build prompt for SAP AI Core
	prompt := fmt.Sprintf(`Given this passage, generate %d diverse yes/no questions that test different types of reasoning:

Passage: %s

Generate questions that require:
1. Paraphrasing (rewording passage content)
2. Factual reasoning (direct inference from facts)
3. Implicit reasoning (reading between the lines)
4. Missing mention (information not stated)
5. By example (reasoning from examples)

Generate %d different questions, one per line:`, count, passage, count)

	// Call SAP AI Core
	response, err := g.Client.GenerateCompletion(ctx, prompt, g.MaxTokens)
	if err != nil {
		// Fallback to template-based generation
		return g.generateFallback(passage, count), nil
	}

	// Parse response into question candidates
	return g.parseQuestions(response, passage), nil
}

func (g *SAPAIQuestionGenerator) parseQuestions(response string, passage string) []QuestionCandidate {
	lines := strings.Split(response, "\n")
	candidates := make([]QuestionCandidate, 0, len(lines))

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) == 0 || !strings.Contains(line, "?") {
			continue
		}

		// Clean up question
		line = strings.TrimPrefix(line, "- ")
		line = strings.TrimPrefix(line, "* ")

		// Classify inference type
		inferenceType := classifyQuestionType(line, passage)

		// Estimate complexity
		complexity := estimateComplexity(line)

		candidates = append(candidates, QuestionCandidate{
			Text:          line,
			InferenceType: inferenceType,
			Complexity:    complexity,
			Confidence:    0.8, // High confidence from real model
		})
	}

	return candidates
}

func (g *SAPAIQuestionGenerator) generateFallback(passage string, count int) []QuestionCandidate {
	// Use template-based generation as fallback
	templates := getAdvancedTemplates(passage)

	candidates := make([]QuestionCandidate, 0, count)
	for i := 0; i < count && i < len(templates); i++ {
		candidates = append(candidates, templates[i])
	}

	return candidates
}

func classifyQuestionType(question string, passage string) InferenceType {
	qLower := strings.ToLower(question)
	pLower := strings.ToLower(passage)

	// Check for direct mentions (paraphrasing)
	qWords := strings.Fields(qLower)
	directMentions := 0
	for _, word := range qWords {
		if len(word) > 3 && strings.Contains(pLower, word) {
			directMentions++
		}
	}
	if float64(directMentions)/float64(len(qWords)) > 0.6 {
		return Paraphrasing
	}

	// Check for inference keywords
	if strings.Contains(qLower, "infer") || strings.Contains(qLower, "imply") || strings.Contains(qLower, "suggest") {
		return Implicit
	}

	// Check for negation/absence
	if strings.Contains(qLower, "not") || strings.Contains(qLower, "never") || strings.Contains(qLower, "any") {
		return MissingMention
	}

	// Check for examples
	if strings.Contains(qLower, "example") || strings.Contains(qLower, "instance") {
		return ByExample
	}

	// Default to factual reasoning
	return FactualReasoning
}

func estimateComplexity(question string) float64 {
	// Longer questions tend to be more complex
	words := strings.Fields(question)
	baseComplexity := float64(len(words)) / 20.0

	// Adjust for complex words
	complexWords := 0
	for _, word := range words {
		if len(word) > 8 {
			complexWords++
		}
	}

	complexity := baseComplexity + float64(complexWords)*0.1

	// Normalize to 0-1
	if complexity > 1.0 {
		complexity = 1.0
	}

	return complexity
}

func getAdvancedTemplates(passage string) []QuestionCandidate {
	entities := extractEntities(passage)
	actions := extractActions(passage)

	var templates []QuestionCandidate

	// Paraphrasing questions
	for _, entity := range entities {
		templates = append(templates, QuestionCandidate{
			Text:          fmt.Sprintf("Is %s mentioned in the passage?", entity),
			InferenceType: Paraphrasing,
			Complexity:    0.3,
			Confidence:    0.9,
		})
	}

	// Factual reasoning
	for _, action := range actions {
		templates = append(templates, QuestionCandidate{
			Text:          fmt.Sprintf("Does the passage indicate that %s occurred?", action),
			InferenceType: FactualReasoning,
			Complexity:    0.6,
			Confidence:    0.8,
		})
	}

	// Implicit reasoning
	templates = append(templates, QuestionCandidate{
		Text:          "Can we infer that the situation was successful?",
		InferenceType: Implicit,
		Complexity:    0.8,
		Confidence:    0.7,
	})

	return templates
}
