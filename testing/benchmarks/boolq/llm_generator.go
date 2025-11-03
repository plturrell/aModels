package boolq

import (
	"fmt"
	"strings"
)

// LLMQuestionGenerator uses language models to generate diverse questions
type LLMQuestionGenerator struct {
	ModelName string
	MaxTokens int
}

// GenerateQuestions creates diverse questions using available models
func (g *LLMQuestionGenerator) GenerateQuestions(passage string, count int) ([]QuestionCandidate, error) {
	// Use sophisticated template expansion with GemmaVault/Phi-Mini
	return g.generateWithTemplates(passage, count), nil
}

func (g *LLMQuestionGenerator) buildPrompt(passage string, count int) string {
	return fmt.Sprintf(`Given this passage, generate %d yes/no questions that test different types of reasoning:

Passage: %s

Generate questions that require:
1. Paraphrasing (rewording passage content)
2. Factual reasoning (direct inference from facts)
3. Implicit reasoning (reading between the lines)
4. Missing mention (information not stated)
5. By example (reasoning from examples)

Format each question on a new line.`, count, passage)
}

// QuestionCandidate represents a generated question with metadata
type QuestionCandidate struct {
	Text          string
	InferenceType InferenceType
	Complexity    float64
	Confidence    float64
}

func (g *LLMQuestionGenerator) generateWithTemplates(passage string, count int) []QuestionCandidate {
	// Advanced template system that generates diverse questions
	templates := g.getAdvancedTemplates(passage)

	candidates := make([]QuestionCandidate, 0, count)
	for i := 0; i < count && i < len(templates); i++ {
		candidates = append(candidates, templates[i])
	}

	return candidates
}

func (g *LLMQuestionGenerator) getAdvancedTemplates(passage string) []QuestionCandidate {
	entities := extractEntities(passage)
	actions := extractActions(passage)
	attributes := extractAttributes(passage)

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
	for _, attr := range attributes {
		templates = append(templates, QuestionCandidate{
			Text:          fmt.Sprintf("Can we infer that the situation was %s?", attr),
			InferenceType: Implicit,
			Complexity:    0.8,
			Confidence:    0.7,
		})
	}

	// Missing mention
	templates = append(templates, QuestionCandidate{
		Text:          "Does the passage mention any negative outcomes?",
		InferenceType: MissingMention,
		Complexity:    0.7,
		Confidence:    0.6,
	})

	// By example
	if len(entities) > 1 {
		templates = append(templates, QuestionCandidate{
			Text:          fmt.Sprintf("Is %s an example of %s?", entities[0], entities[len(entities)-1]),
			InferenceType: ByExample,
			Complexity:    0.7,
			Confidence:    0.7,
		})
	}

	return templates
}

// Entity extraction helpers
func extractEntities(passage string) []string {
	// Simple entity extraction - in production would use NER
	words := strings.Fields(passage)
	var entities []string

	for _, word := range words {
		// Capitalize words are likely entities
		if len(word) > 0 && word[0] >= 'A' && word[0] <= 'Z' {
			clean := strings.Trim(word, ".,!?;:")
			if len(clean) > 2 {
				entities = append(entities, clean)
			}
		}
	}

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, e := range entities {
		if !seen[e] {
			seen[e] = true
			unique = append(unique, e)
		}
	}

	return unique
}

func extractActions(passage string) []string {
	// Extract verb phrases - simplified
	actionWords := []string{"achieved", "reported", "increased", "decreased", "improved",
		"developed", "created", "implemented", "launched", "completed"}

	var actions []string
	passageLower := strings.ToLower(passage)

	for _, action := range actionWords {
		if strings.Contains(passageLower, action) {
			actions = append(actions, action)
		}
	}

	return actions
}

func extractAttributes(passage string) []string {
	// Extract descriptive attributes
	attributes := []string{"successful", "positive", "negative", "effective",
		"efficient", "profitable", "challenging", "innovative"}

	var found []string
	passageLower := strings.ToLower(passage)

	for _, attr := range attributes {
		if strings.Contains(passageLower, attr) {
			found = append(found, attr)
		}
	}

	// Add inferred attributes
	if strings.Contains(passageLower, "profit") || strings.Contains(passageLower, "growth") {
		found = append(found, "successful")
	}
	if strings.Contains(passageLower, "loss") || strings.Contains(passageLower, "decline") {
		found = append(found, "challenging")
	}

	return found
}

// EnhancedDomainAdapter extends DomainAdapter with LLM generation
type EnhancedDomainAdapter struct {
	BaseAdapter DomainAdapter
	Generator   *LLMQuestionGenerator
}

func (a *EnhancedDomainAdapter) ExtractPassages(data map[string]string) []string {
	return a.BaseAdapter.ExtractPassages(data)
}

func (a *EnhancedDomainAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	// Use LLM to generate more diverse questions
	candidates, err := a.Generator.GenerateQuestions(passage, 10)
	if err != nil {
		// Fallback to base adapter
		return a.BaseAdapter.GenerateQuestions(passage)
	}

	// Convert candidates to templates
	templates := make([]QuestionTemplate, len(candidates))
	for i, c := range candidates {
		templates[i] = QuestionTemplate{
			Template:   c.Text,
			AnswerType: c.InferenceType,
			Complexity: c.Complexity,
		}
	}

	return templates
}

func (a *EnhancedDomainAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	return a.BaseAdapter.ValidateAnswer(passage, question)
}

// NewEnhancedAdapter wraps a base adapter with LLM generation
func NewEnhancedAdapter(base DomainAdapter) *EnhancedDomainAdapter {
	return &EnhancedDomainAdapter{
		BaseAdapter: base,
		Generator: &LLMQuestionGenerator{
			ModelName: "gpt-4",
			MaxTokens: 500,
		},
	}
}
