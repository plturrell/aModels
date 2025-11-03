package boolq

import (
	"fmt"
	"strings"
)

// FinancialAdapter handles financial data
type FinancialAdapter struct{}

func (a *FinancialAdapter) ExtractPassages(data map[string]string) []string {
	var passages []string

	if company, exists := data["CompanyName"]; exists {
		passage := fmt.Sprintf("%s reported financial results for the period. ", company)

		if revenue, exists := data["Revenue"]; exists {
			passage += fmt.Sprintf("The company achieved revenue of $%s. ", revenue)
		}
		if netIncome, exists := data["NetIncome"]; exists {
			passage += fmt.Sprintf("Net income was $%s. ", netIncome)
		}
		if profitMargin, exists := data["ProfitMargin"]; exists {
			passage += fmt.Sprintf("The profit margin was %s%%. ", profitMargin)
		}

		passages = append(passages, passage)
	}

	return passages
}

func (a *FinancialAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	return []QuestionTemplate{
		{
			Template:   "Did the company achieve profitability during this period?",
			AnswerType: FactualReasoning,
			Complexity: 0.7,
		},
		{
			Template:   "Was the revenue growth positive?",
			AnswerType: Paraphrasing,
			Complexity: 0.5,
		},
		{
			Template:   "Does the passage indicate strong financial performance?",
			AnswerType: Implicit,
			Complexity: 0.9,
		},
	}
}

func (a *FinancialAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	passage = strings.ToLower(passage)
	question = strings.ToLower(question)

	switch {
	case strings.Contains(question, "profitability"):
		hasProfit := (strings.Contains(passage, "net income") && !strings.Contains(passage, "net income -")) ||
			(strings.Contains(passage, "profit") && !strings.Contains(passage, "loss"))
		return hasProfit, 0.8, FactualReasoning

	case strings.Contains(question, "revenue growth"):
		return true, 0.6, Paraphrasing

	case strings.Contains(question, "strong financial performance"):
		hasPositiveMetrics := strings.Contains(passage, "net income") &&
			(strings.Contains(passage, "profit") || strings.Contains(passage, "growth"))
		return hasPositiveMetrics, 0.7, Implicit
	}

	return false, 0.5, OtherInference
}

// TechnicalAdapter handles technical/scientific data
type TechnicalAdapter struct{}

func (a *TechnicalAdapter) ExtractPassages(data map[string]string) []string {
	var passages []string

	if system, exists := data["SystemName"]; exists {
		passage := fmt.Sprintf("%s is a technical system. ", system)

		if spec, exists := data["Specification"]; exists {
			passage += fmt.Sprintf("It supports %s. ", spec)
		}
		if capability, exists := data["Capability"]; exists {
			passage += fmt.Sprintf("The system provides %s functionality. ", capability)
		}

		passages = append(passages, passage)
	}

	return passages
}

func (a *TechnicalAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	return []QuestionTemplate{
		{
			Template:   "Does the system support this functionality?",
			AnswerType: ByExample,
			Complexity: 0.6,
		},
		{
			Template:   "Is this a production-ready system?",
			AnswerType: Implicit,
			Complexity: 0.8,
		},
	}
}

func (a *TechnicalAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	passage = strings.ToLower(passage)
	question = strings.ToLower(question)

	if strings.Contains(question, "support") && strings.Contains(passage, "support") {
		return true, 0.8, ByExample
	}
	if strings.Contains(question, "production-ready") {
		return strings.Contains(passage, "production") || strings.Contains(passage, "stable"), 0.7, Implicit
	}

	return true, 0.6, OtherInference
}

// MedicalAdapter handles medical data
type MedicalAdapter struct{}

func (a *MedicalAdapter) ExtractPassages(data map[string]string) []string {
	var passages []string

	if condition, exists := data["Condition"]; exists {
		passage := fmt.Sprintf("The patient presents with %s. ", condition)

		if treatment, exists := data["Treatment"]; exists {
			passage += fmt.Sprintf("Treatment includes %s. ", treatment)
		}
		if outcome, exists := data["Outcome"]; exists {
			passage += fmt.Sprintf("The outcome was %s. ", outcome)
		}

		passages = append(passages, passage)
	}

	return passages
}

func (a *MedicalAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	return []QuestionTemplate{
		{
			Template:   "Was the treatment effective?",
			AnswerType: FactualReasoning,
			Complexity: 0.7,
		},
		{
			Template:   "Does the patient require follow-up care?",
			AnswerType: Implicit,
			Complexity: 0.8,
		},
	}
}

func (a *MedicalAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	passage = strings.ToLower(passage)
	question = strings.ToLower(question)

	if strings.Contains(question, "effective") {
		isEffective := strings.Contains(passage, "improved") || strings.Contains(passage, "successful") ||
			strings.Contains(passage, "positive")
		return isEffective, 0.8, FactualReasoning
	}
	if strings.Contains(question, "follow-up") {
		needsFollowup := strings.Contains(passage, "ongoing") || strings.Contains(passage, "monitor")
		return needsFollowup, 0.7, Implicit
	}

	return false, 0.5, OtherInference
}

// LegalAdapter handles legal data
type LegalAdapter struct{}

func (a *LegalAdapter) ExtractPassages(data map[string]string) []string {
	var passages []string

	if agreement, exists := data["AgreementType"]; exists {
		passage := fmt.Sprintf("This is a %s agreement. ", agreement)

		if terms, exists := data["Terms"]; exists {
			passage += fmt.Sprintf("The terms specify %s. ", terms)
		}
		if compliance, exists := data["Compliance"]; exists {
			passage += fmt.Sprintf("Compliance with %s is required. ", compliance)
		}

		passages = append(passages, passage)
	}

	return passages
}

func (a *LegalAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	return []QuestionTemplate{
		{
			Template:   "Is this agreement legally binding?",
			AnswerType: FactualReasoning,
			Complexity: 0.7,
		},
		{
			Template:   "Does this require regulatory approval?",
			AnswerType: Implicit,
			Complexity: 0.8,
		},
	}
}

func (a *LegalAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	passage = strings.ToLower(passage)
	question = strings.ToLower(question)

	if strings.Contains(question, "legally binding") {
		isBinding := strings.Contains(passage, "agreement") || strings.Contains(passage, "contract")
		return isBinding, 0.8, FactualReasoning
	}
	if strings.Contains(question, "regulatory") {
		needsApproval := strings.Contains(passage, "compliance") || strings.Contains(passage, "regulation")
		return needsApproval, 0.7, Implicit
	}

	return false, 0.5, OtherInference
}

// GeneralAdapter fallback for unknown domains
type GeneralAdapter struct{}

func (a *GeneralAdapter) ExtractPassages(data map[string]string) []string {
	var passages []string
	var facts []string

	for key, value := range data {
		if value != "" {
			facts = append(facts, fmt.Sprintf("%s is %s", key, value))
		}
	}

	if len(facts) > 0 {
		passage := strings.Join(facts, ". ") + "."
		passages = append(passages, passage)
	}

	return passages
}

func (a *GeneralAdapter) GenerateQuestions(passage string) []QuestionTemplate {
	return []QuestionTemplate{
		{
			Template:   "Is this statement supported by the passage: '{fact}'?",
			AnswerType: Paraphrasing,
			Complexity: 0.5,
		},
		{
			Template:   "Can we infer that '{inference}' from this information?",
			AnswerType: Implicit,
			Complexity: 0.8,
		},
	}
}

func (a *GeneralAdapter) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	passageLower := strings.ToLower(passage)
	questionLower := strings.ToLower(question)

	questionTerms := extractKeyTerms(questionLower)
	matches := 0
	for _, term := range questionTerms {
		if strings.Contains(passageLower, term) {
			matches++
		}
	}

	if len(questionTerms) == 0 {
		return false, 0.5, Paraphrasing
	}

	confidence := float64(matches) / float64(len(questionTerms))
	return matches > len(questionTerms)/2, confidence, Paraphrasing
}
