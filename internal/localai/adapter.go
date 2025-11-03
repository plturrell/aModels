package localai

import (
	"fmt"
	"strings"
)

// BenchmarkAdapter adapts benchmark tasks to LocalAI inference
type BenchmarkAdapter struct {
	Client *Client
}

// NewBenchmarkAdapter creates a new adapter
func NewBenchmarkAdapter(client *Client) *BenchmarkAdapter {
	return &BenchmarkAdapter{
		Client: client,
	}
}

// MCQPrompt formats a multiple choice question for the model
func (ba *BenchmarkAdapter) MCQPrompt(question string, choices []string) string {
	var sb strings.Builder
	sb.WriteString("Question: ")
	sb.WriteString(question)
	sb.WriteString("\n\nChoices:\n")

	for i, choice := range choices {
		sb.WriteString(fmt.Sprintf("%c) %s\n", 'A'+i, choice))
	}

	sb.WriteString("\nAnswer (letter only): ")
	return sb.String()
}

// BoolQPrompt formats a boolean question with passage
func (ba *BenchmarkAdapter) BoolQPrompt(question, passage string) string {
	var sb strings.Builder
	if passage != "" {
		sb.WriteString("Passage: ")
		sb.WriteString(passage)
		sb.WriteString("\n\n")
	}
	sb.WriteString("Question: ")
	sb.WriteString(question)
	sb.WriteString("\n\nAnswer (yes/no): ")
	return sb.String()
}

// HellaSwagPrompt formats a context completion task
func (ba *BenchmarkAdapter) HellaSwagPrompt(context string, endings []string) string {
	var sb strings.Builder
	sb.WriteString("Context: ")
	sb.WriteString(context)
	sb.WriteString("\n\nPossible continuations:\n")

	for i, ending := range endings {
		sb.WriteString(fmt.Sprintf("%d) %s\n", i+1, ending))
	}

	sb.WriteString("\nMost likely continuation (number only): ")
	return sb.String()
}

// TriviaQAPrompt formats a trivia question
func (ba *BenchmarkAdapter) TriviaQAPrompt(question string) string {
	return fmt.Sprintf("Question: %s\n\nAnswer: ", question)
}

// ARCPrompt formats an ARC reasoning task
func (ba *BenchmarkAdapter) ARCPrompt(description string, grid [][]int) string {
	var sb strings.Builder
	sb.WriteString("Task: ")
	sb.WriteString(description)
	sb.WriteString("\n\nInput Grid:\n")

	for _, row := range grid {
		for _, cell := range row {
			sb.WriteString(fmt.Sprintf("%d ", cell))
		}
		sb.WriteString("\n")
	}

	sb.WriteString("\nPredict the output grid pattern.")
	return sb.String()
}

// ParseMCQAnswer extracts the letter choice from model output
func (ba *BenchmarkAdapter) ParseMCQAnswer(output string) string {
	output = strings.TrimSpace(output)
	output = strings.ToUpper(output)

	// Look for single letter A-Z
	if len(output) == 1 && output[0] >= 'A' && output[0] <= 'Z' {
		return output
	}

	// Look for pattern like "A)" or "A."
	if len(output) >= 2 && output[0] >= 'A' && output[0] <= 'Z' {
		if output[1] == ')' || output[1] == '.' || output[1] == ':' {
			return string(output[0])
		}
	}

	// Look for "Answer: A" pattern
	if strings.Contains(output, "Answer:") {
		parts := strings.Split(output, "Answer:")
		if len(parts) > 1 {
			answer := strings.TrimSpace(parts[1])
			if len(answer) > 0 && answer[0] >= 'A' && answer[0] <= 'Z' {
				return string(answer[0])
			}
		}
	}

	return ""
}

// ParseBoolAnswer extracts yes/no from model output
func (ba *BenchmarkAdapter) ParseBoolAnswer(output string) bool {
	output = strings.ToLower(strings.TrimSpace(output))

	// Direct matches
	if output == "yes" || output == "true" || output == "1" {
		return true
	}
	if output == "no" || output == "false" || output == "0" {
		return false
	}

	// Pattern matches
	if strings.Contains(output, "yes") || strings.Contains(output, "true") {
		return true
	}

	return false
}

// ParseNumberAnswer extracts a number from model output
func (ba *BenchmarkAdapter) ParseNumberAnswer(output string) int {
	output = strings.TrimSpace(output)

	// Look for single digit
	if len(output) == 1 && output[0] >= '0' && output[0] <= '9' {
		return int(output[0] - '0')
	}

	// Look for pattern like "1)" or "1."
	if len(output) >= 2 && output[0] >= '0' && output[0] <= '9' {
		if output[1] == ')' || output[1] == '.' || output[1] == ':' {
			return int(output[0] - '0')
		}
	}

	return -1
}

// InferMCQ runs inference for multiple choice question
func (ba *BenchmarkAdapter) InferMCQ(modelName, question string, choices []string, temp float64) (string, error) {
	prompt := ba.MCQPrompt(question, choices)

	req := CompletionRequest{
		Model:       modelName,
		Prompt:      prompt,
		Temperature: temp,
		MaxTokens:   10,
		Stop:        []string{"\n"},
	}

	resp, err := ba.Client.Complete(req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from model")
	}

	answer := ba.ParseMCQAnswer(resp.Choices[0].Text)
	return answer, nil
}

// InferBool runs inference for boolean question
func (ba *BenchmarkAdapter) InferBool(modelName, question, passage string, temp float64) (bool, error) {
	prompt := ba.BoolQPrompt(question, passage)

	req := CompletionRequest{
		Model:       modelName,
		Prompt:      prompt,
		Temperature: temp,
		MaxTokens:   5,
		Stop:        []string{"\n"},
	}

	resp, err := ba.Client.Complete(req)
	if err != nil {
		return false, err
	}

	if len(resp.Choices) == 0 {
		return false, fmt.Errorf("no response from model")
	}

	answer := ba.ParseBoolAnswer(resp.Choices[0].Text)
	return answer, nil
}

// InferCompletion runs inference for completion task
func (ba *BenchmarkAdapter) InferCompletion(modelName, context string, endings []string, temp float64) (int, error) {
	prompt := ba.HellaSwagPrompt(context, endings)

	req := CompletionRequest{
		Model:       modelName,
		Prompt:      prompt,
		Temperature: temp,
		MaxTokens:   5,
		Stop:        []string{"\n"},
	}

	resp, err := ba.Client.Complete(req)
	if err != nil {
		return -1, err
	}

	if len(resp.Choices) == 0 {
		return -1, fmt.Errorf("no response from model")
	}

	answer := ba.ParseNumberAnswer(resp.Choices[0].Text)
	return answer, nil
}

// InferOpenEnded runs inference for open-ended questions
func (ba *BenchmarkAdapter) InferOpenEnded(modelName, prompt string, temp float64, maxTokens int) (string, error) {
	req := CompletionRequest{
		Model:       modelName,
		Prompt:      prompt,
		Temperature: temp,
		MaxTokens:   maxTokens,
	}

	resp, err := ba.Client.Complete(req)
	if err != nil {
		return "", err
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response from model")
	}

	return strings.TrimSpace(resp.Choices[0].Text), nil
}
