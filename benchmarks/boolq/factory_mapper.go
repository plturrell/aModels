package boolq

import (
	"ai_benchmarks/scripts/factory"
	"fmt"
	"strings"
)

// InferenceType represents the reasoning categories from BoolQ paper
type InferenceType string

const (
	Paraphrasing     InferenceType = "paraphrasing"
	ByExample        InferenceType = "by_example"
	FactualReasoning InferenceType = "factual_reasoning"
	Implicit         InferenceType = "implicit"
	MissingMention   InferenceType = "missing_mention"
	OtherInference   InferenceType = "other_inference"
)

// BoolQTask defines the structure for generated BoolQ training data
type BoolQTask struct {
	Question      string        `json:"question"`
	Passage       string        `json:"passage"`
	Answer        bool          `json:"answer"`
	InferenceType InferenceType `json:"inference_type,omitempty"`
	Domain        string        `json:"domain,omitempty"`
	Confidence    float64       `json:"confidence,omitempty"` // For calibration weighting
}

// DomainAdapter defines the interface for domain-specific data conversion
type DomainAdapter interface {
	ExtractPassages(data map[string]string) []string
	GenerateQuestions(passage string) []QuestionTemplate
	ValidateAnswer(passage, question string) (bool, float64, InferenceType)
}

// QuestionTemplate defines a question generation pattern
type QuestionTemplate struct {
	Template   string
	AnswerType InferenceType
	Complexity float64 // 0.0 to 1.0 for calibration weighting
}

// GenericBoolQFactory creates BoolQ training data from any domain
type GenericBoolQFactory struct {
	Adapters map[string]DomainAdapter
}

// NewGenericBoolQFactory creates a new factory with domain adapters
func NewGenericBoolQFactory() *GenericBoolQFactory {
	return &GenericBoolQFactory{
		Adapters: map[string]DomainAdapter{
			"financial": &FinancialAdapter{},
			"technical": &TechnicalAdapter{},
			"medical":   &MedicalAdapter{},
			"legal":     &LegalAdapter{},
			"general":   &GeneralAdapter{},
		},
	}
}

// Map transforms domain data into BoolQ training examples
func (f *GenericBoolQFactory) Map(data factory.SourceData) ([]factory.BenchmarkTask, error) {
	row, ok := data.Content.(map[string]string)
	if !ok {
		return nil, fmt.Errorf("invalid data type for GenericBoolQFactory")
	}

	// Detect domain or use default
	domain := f.detectDomain(row)
	adapter, exists := f.Adapters[domain]
	if !exists {
		adapter = f.Adapters["general"]
	}

	// Extract passages from domain data
	passages := adapter.ExtractPassages(row)
	var tasks []factory.BenchmarkTask

	// Generate BoolQ examples from each passage
	for _, passage := range passages {
		questionTemplates := adapter.GenerateQuestions(passage)

		for _, template := range questionTemplates {
			question := f.instantiateTemplate(template.Template, passage, row)
			answer, confidence, inferenceType := adapter.ValidateAnswer(passage, question)

			task := BoolQTask{
				Question:      question,
				Passage:       passage,
				Answer:        answer,
				InferenceType: inferenceType,
				Domain:        domain,
				Confidence:    confidence * template.Complexity,
			}

			tasks = append(tasks, task)
		}
	}

	return tasks, nil
}

// detectDomain automatically identifies the data domain
func (f *GenericBoolQFactory) detectDomain(row map[string]string) string {
	content := strings.ToLower(fmt.Sprintf("%v", row))

	switch {
	case containsFinancialTerms(content):
		return "financial"
	case containsTechnicalTerms(content):
		return "technical"
	case containsMedicalTerms(content):
		return "medical"
	case containsLegalTerms(content):
		return "legal"
	default:
		return "general"
	}
}

func (f *GenericBoolQFactory) instantiateTemplate(template, passage string, data map[string]string) string {
	question := template

	// Handle {fact} placeholder
	if strings.Contains(question, "{fact}") {
		fact := extractRandomFact(passage)
		question = strings.Replace(question, "{fact}", fact, 1)
	}

	// Handle {inference} placeholder
	if strings.Contains(question, "{inference}") {
		inference := generatePlausibleInference(passage)
		question = strings.Replace(question, "{inference}", inference, 1)
	}

	return question
}

// Helper functions
func containsFinancialTerms(content string) bool {
	financialTerms := []string{"revenue", "income", "profit", "margin", "financial", "earnings", "ebitda"}
	return containsAny(content, financialTerms)
}

func containsTechnicalTerms(content string) bool {
	technicalTerms := []string{"system", "technical", "specification", "protocol", "api", "integration"}
	return containsAny(content, technicalTerms)
}

func containsMedicalTerms(content string) bool {
	medicalTerms := []string{"patient", "medical", "treatment", "diagnosis", "clinical", "health"}
	return containsAny(content, medicalTerms)
}

func containsLegalTerms(content string) bool {
	legalTerms := []string{"legal", "contract", "agreement", "compliance", "regulation", "law"}
	return containsAny(content, legalTerms)
}

func containsAny(content string, terms []string) bool {
	for _, term := range terms {
		if strings.Contains(content, term) {
			return true
		}
	}
	return false
}

func extractKeyTerms(text string) []string {
	stopWords := map[string]bool{"the": true, "is": true, "are": true, "this": true, "that": true, "and": true, "or": true}
	words := strings.Fields(text)
	var terms []string

	for _, word := range words {
		if len(word) > 3 && !stopWords[word] {
			terms = append(terms, word)
		}
	}
	return terms
}

func extractRandomFact(passage string) string {
	sentences := strings.Split(passage, ".")
	if len(sentences) > 1 {
		return strings.TrimSpace(sentences[0]) + "."
	}
	return passage
}

func generatePlausibleInference(passage string) string {
	return "this represents positive performance"
}
