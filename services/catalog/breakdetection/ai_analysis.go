package breakdetection

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// AIAnalysisService provides AI-powered analysis using LocalAI
type AIAnalysisService struct {
	localaiURL string
	httpClient *http.Client
	logger     *log.Logger
}

// NewAIAnalysisService creates a new AI analysis service
func NewAIAnalysisService(localaiURL string, logger *log.Logger) *AIAnalysisService {
	if localaiURL == "" {
		localaiURL = "http://localhost:8081" // Default LocalAI URL
	}
	return &AIAnalysisService{
		localaiURL: localaiURL,
		httpClient: &http.Client{Timeout: 60 * time.Second},
		logger:     logger,
	}
}

// GenerateBreakDescription generates a natural language description of a break using LocalAI
func (aas *AIAnalysisService) GenerateBreakDescription(ctx context.Context, breakRecord *Break) (string, error) {
	if aas.localaiURL == "" {
		return "", fmt.Errorf("LocalAI URL not configured")
	}

	// Build prompt for break description
	prompt := aas.buildDescriptionPrompt(breakRecord)

	// Call LocalAI
	response, err := aas.callLocalAI(ctx, prompt, "Generate a clear, concise natural language description of this break.")
	if err != nil {
		return "", fmt.Errorf("failed to generate break description: %w", err)
	}

	if aas.logger != nil {
		aas.logger.Printf("Generated AI description for break: %s", breakRecord.BreakID)
	}

	return response, nil
}

// CategorizeBreak categorizes a break using AI
func (aas *AIAnalysisService) CategorizeBreak(ctx context.Context, breakRecord *Break) (string, error) {
	if aas.localaiURL == "" {
		return "", fmt.Errorf("LocalAI URL not configured")
	}

	// Build categorization prompt
	prompt := aas.buildCategorizationPrompt(breakRecord)

	// Call LocalAI
	response, err := aas.callLocalAI(ctx, prompt, "Categorize this break into one of these categories: data_quality, system_error, configuration_error, migration_issue, calculation_error, or other.")
	if err != nil {
		return "", fmt.Errorf("failed to categorize break: %w", err)
	}

	// Clean and normalize category
	category := strings.ToLower(strings.TrimSpace(response))
	category = strings.Split(category, "\n")[0] // Take first line
	category = strings.Split(category, ".")[0]  // Remove period if present

	if aas.logger != nil {
		aas.logger.Printf("Categorized break %s as: %s", breakRecord.BreakID, category)
	}

	return category, nil
}

// CalculatePriorityScore calculates an AI-based priority score for a break
func (aas *AIAnalysisService) CalculatePriorityScore(ctx context.Context, breakRecord *Break) (float64, error) {
	if aas.localaiURL == "" {
		return 0.0, fmt.Errorf("LocalAI URL not configured")
	}

	// Build priority calculation prompt
	prompt := aas.buildPriorityPrompt(breakRecord)

	// Call LocalAI
	response, err := aas.callLocalAI(ctx, prompt, "Calculate a priority score from 0.0 to 1.0 based on severity, impact, and business criticality. Return only the numeric score.")
	if err != nil {
		return 0.0, fmt.Errorf("failed to calculate priority score: %w", err)
	}

	// Parse score from response
	var score float64
	if _, err := fmt.Sscanf(strings.TrimSpace(response), "%f", &score); err != nil {
		// Try to extract number from text
		score = aas.extractScoreFromText(response)
	}

	// Ensure score is in valid range
	if score < 0.0 {
		score = 0.0
	}
	if score > 1.0 {
		score = 1.0
	}

	if aas.logger != nil {
		aas.logger.Printf("Calculated priority score for break %s: %.4f", breakRecord.BreakID, score)
	}

	return score, nil
}

// AnalyzeBreakSemantically performs semantic analysis of a break using LocalAI
func (aas *AIAnalysisService) AnalyzeBreakSemantically(ctx context.Context, breakRecord *Break) (map[string]interface{}, error) {
	if aas.localaiURL == "" {
		return nil, fmt.Errorf("LocalAI URL not configured")
	}

	// Build semantic analysis prompt
	prompt := aas.buildSemanticAnalysisPrompt(breakRecord)

	// Call LocalAI
	response, err := aas.callLocalAI(ctx, prompt, "Provide a comprehensive semantic analysis of this break including impact, dependencies, and business context.")
	if err != nil {
		return nil, fmt.Errorf("failed to perform semantic analysis: %w", err)
	}

	// Parse response into structured data
	analysis := aas.parseSemanticAnalysis(response)

	if aas.logger != nil {
		aas.logger.Printf("Performed semantic analysis for break: %s", breakRecord.BreakID)
	}

	return analysis, nil
}

// buildDescriptionPrompt builds a prompt for break description generation
func (aas *AIAnalysisService) buildDescriptionPrompt(breakRecord *Break) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("Generate a clear, concise natural language description of this break:\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	promptBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))
	
	if len(breakRecord.AffectedEntities) > 0 {
		promptBuilder.WriteString(fmt.Sprintf("Affected Entities: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}
	
	if breakRecord.Difference != nil {
		promptBuilder.WriteString(fmt.Sprintf("Difference: %v\n", breakRecord.Difference))
	}

	return promptBuilder.String()
}

// buildCategorizationPrompt builds a prompt for break categorization
func (aas *AIAnalysisService) buildCategorizationPrompt(breakRecord *Break) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("Categorize this break into one category:\n\n")
	promptBuilder.WriteString("Categories: data_quality, system_error, configuration_error, migration_issue, calculation_error, other\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	promptBuilder.WriteString(fmt.Sprintf("Detection Type: %s\n", breakRecord.DetectionType))
	
	if breakRecord.RootCauseAnalysis != "" {
		promptBuilder.WriteString(fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis))
	}

	return promptBuilder.String()
}

// buildPriorityPrompt builds a prompt for priority calculation
func (aas *AIAnalysisService) buildPriorityPrompt(breakRecord *Break) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("Calculate a priority score (0.0 to 1.0) for this break based on:\n")
	promptBuilder.WriteString("- Severity (critical=1.0, high=0.75, medium=0.5, low=0.25)\n")
	promptBuilder.WriteString("- Business impact\n")
	promptBuilder.WriteString("- Number of affected entities\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))
	promptBuilder.WriteString(fmt.Sprintf("Affected Entities Count: %d\n", len(breakRecord.AffectedEntities)))
	
	if breakRecord.SystemName == SystemSAPFioneer {
		promptBuilder.WriteString("System Impact: Finance system - high business criticality\n")
	}

	return promptBuilder.String()
}

// buildSemanticAnalysisPrompt builds a prompt for semantic analysis
func (aas *AIAnalysisService) buildSemanticAnalysisPrompt(breakRecord *Break) string {
	var promptBuilder strings.Builder

	promptBuilder.WriteString("Provide a comprehensive semantic analysis of this break:\n\n")
	promptBuilder.WriteString(fmt.Sprintf("Break Type: %s\n", breakRecord.BreakType))
	promptBuilder.WriteString(fmt.Sprintf("System: %s\n", breakRecord.SystemName))
	promptBuilder.WriteString(fmt.Sprintf("Severity: %s\n", breakRecord.Severity))
	
	if breakRecord.RootCauseAnalysis != "" {
		promptBuilder.WriteString(fmt.Sprintf("Root Cause: %s\n", breakRecord.RootCauseAnalysis))
	}
	
	if len(breakRecord.AffectedEntities) > 0 {
		promptBuilder.WriteString(fmt.Sprintf("Affected: %s\n", strings.Join(breakRecord.AffectedEntities, ", ")))
	}

	return promptBuilder.String()
}

// callLocalAI calls the LocalAI service
func (aas *AIAnalysisService) callLocalAI(ctx context.Context, prompt, systemPrompt string) (string, error) {
	// Prepare chat completion request
	messages := []map[string]interface{}{
		{
			"role":    "system",
			"content": systemPrompt,
		},
		{
			"role":    "user",
			"content": prompt,
		},
	}

	payload := map[string]interface{}{
		"model":       "general", // Default model
		"messages":    messages,
		"temperature": 0.3, // Lower temperature for more deterministic responses
		"max_tokens":  500,
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1/chat/completions", aas.localaiURL)
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonData))
	if err != nil {
		return "", fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := aas.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("LocalAI returned status %d", resp.StatusCode)
	}

	var result map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", fmt.Errorf("failed to decode response: %w", err)
	}

	// Extract content from response
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

// extractScoreFromText extracts a numeric score from text
func (aas *AIAnalysisService) extractScoreFromText(text string) float64 {
	// Look for number patterns
	var score float64
	text = strings.ToLower(text)
	
	// Try to find decimal number
	if _, err := fmt.Sscanf(text, "%f", &score); err == nil {
		return score
	}
	
	// Try percentage format
	if strings.Contains(text, "%") {
		if _, err := fmt.Sscanf(text, "%f%%", &score); err == nil {
			return score / 100.0
		}
	}
	
	// Default based on keywords
	if strings.Contains(text, "high") || strings.Contains(text, "critical") {
		return 0.8
	}
	if strings.Contains(text, "medium") {
		return 0.5
	}
	if strings.Contains(text, "low") {
		return 0.2
	}
	
	return 0.5 // Default
}

// parseSemanticAnalysis parses semantic analysis response into structured data
func (aas *AIAnalysisService) parseSemanticAnalysis(response string) map[string]interface{} {
	analysis := make(map[string]interface{})
	analysis["raw_response"] = response
	
	// Try to extract structured information
	lines := strings.Split(response, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		
		// Look for key-value pairs
		if strings.Contains(line, ":") {
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				key := strings.ToLower(strings.TrimSpace(parts[0]))
				value := strings.TrimSpace(parts[1])
				
				// Map common keys
				switch {
				case strings.Contains(key, "impact"):
					analysis["impact"] = value
				case strings.Contains(key, "dependenc"):
					analysis["dependencies"] = value
				case strings.Contains(key, "business"):
					analysis["business_context"] = value
				case strings.Contains(key, "risk"):
					analysis["risk"] = value
				}
			}
		}
	}
	
	return analysis
}

