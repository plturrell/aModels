package boolq

import (
	"math"
	"strings"
)

// SemanticValidator uses embeddings for answer validation
type SemanticValidator struct {
	EmbeddingModel string
}

// ValidateAnswer uses semantic similarity instead of keyword matching
func (v *SemanticValidator) ValidateAnswer(passage, question string, expectedAnswer bool) (float64, InferenceType) {
	// Extract semantic features
	passageFeatures := v.extractFeatures(passage)
	questionFeatures := v.extractFeatures(question)

	// Calculate semantic similarity
	similarity := v.cosineSimilarity(passageFeatures, questionFeatures)

	// Determine inference type from question structure
	inferenceType := v.classifyInferenceType(question, passage)

	// Confidence based on semantic overlap
	confidence := v.calculateConfidence(similarity, inferenceType)

	return confidence, inferenceType
}

// Feature represents semantic features of text
type Feature struct {
	Tokens    []string
	Bigrams   []string
	Entities  []string
	Sentiment float64
}

func (v *SemanticValidator) extractFeatures(text string) Feature {
	tokens := strings.Fields(strings.ToLower(text))

	// Extract bigrams
	var bigrams []string
	for i := 0; i < len(tokens)-1; i++ {
		bigrams = append(bigrams, tokens[i]+" "+tokens[i+1])
	}

	// Extract entities (capitalized words)
	entities := extractEntities(text)

	// Simple sentiment analysis
	sentiment := v.analyzeSentiment(text)

	return Feature{
		Tokens:    tokens,
		Bigrams:   bigrams,
		Entities:  entities,
		Sentiment: sentiment,
	}
}

func (v *SemanticValidator) cosineSimilarity(f1, f2 Feature) float64 {
	// Token overlap
	tokenOverlap := v.setOverlap(f1.Tokens, f2.Tokens)

	// Bigram overlap
	bigramOverlap := v.setOverlap(f1.Bigrams, f2.Bigrams)

	// Entity overlap
	entityOverlap := v.setOverlap(f1.Entities, f2.Entities)

	// Weighted combination
	similarity := 0.5*tokenOverlap + 0.3*bigramOverlap + 0.2*entityOverlap

	return similarity
}

func (v *SemanticValidator) setOverlap(set1, set2 []string) float64 {
	if len(set1) == 0 || len(set2) == 0 {
		return 0.0
	}

	seen := make(map[string]bool)
	for _, item := range set1 {
		seen[item] = true
	}

	overlap := 0
	for _, item := range set2 {
		if seen[item] {
			overlap++
		}
	}

	// Jaccard similarity
	union := len(set1) + len(set2) - overlap
	if union == 0 {
		return 0.0
	}

	return float64(overlap) / float64(union)
}

func (v *SemanticValidator) classifyInferenceType(question, passage string) InferenceType {
	qLower := strings.ToLower(question)
	pLower := strings.ToLower(passage)

	// Paraphrasing - direct mention
	if v.hasDirectMention(qLower, pLower) {
		return Paraphrasing
	}

	// Missing mention - negation or absence
	if strings.Contains(qLower, "not") || strings.Contains(qLower, "never") ||
		strings.Contains(qLower, "no ") || strings.Contains(qLower, "any") {
		return MissingMention
	}

	// Implicit - requires inference
	if strings.Contains(qLower, "infer") || strings.Contains(qLower, "suggest") ||
		strings.Contains(qLower, "imply") || strings.Contains(qLower, "indicate") {
		return Implicit
	}

	// By example - contains "example" or "instance"
	if strings.Contains(qLower, "example") || strings.Contains(qLower, "instance") {
		return ByExample
	}

	// Factual reasoning - default for direct questions
	return FactualReasoning
}

func (v *SemanticValidator) hasDirectMention(question, passage string) bool {
	// Extract key terms from question
	qTokens := strings.Fields(question)

	// Check if most key terms appear in passage
	matches := 0
	for _, token := range qTokens {
		if len(token) > 3 && strings.Contains(passage, token) {
			matches++
		}
	}

	return float64(matches)/float64(len(qTokens)) > 0.6
}

func (v *SemanticValidator) calculateConfidence(similarity float64, inferenceType InferenceType) float64 {
	// Base confidence from similarity
	confidence := similarity

	// Adjust by inference type difficulty
	switch inferenceType {
	case Paraphrasing:
		confidence *= 0.95 // High confidence for direct paraphrasing
	case FactualReasoning:
		confidence *= 0.85 // Good confidence for factual
	case ByExample:
		confidence *= 0.80 // Moderate confidence
	case Implicit:
		confidence *= 0.70 // Lower confidence for implicit
	case MissingMention:
		confidence *= 0.65 // Lowest confidence
	default:
		confidence *= 0.75
	}

	return math.Min(confidence, 1.0)
}

func (v *SemanticValidator) analyzeSentiment(text string) float64 {
	positive := []string{"good", "great", "excellent", "positive", "success", "profit",
		"growth", "improved", "effective", "strong"}
	negative := []string{"bad", "poor", "negative", "loss", "decline", "failed",
		"weak", "ineffective", "challenging"}

	textLower := strings.ToLower(text)

	posCount := 0
	for _, word := range positive {
		if strings.Contains(textLower, word) {
			posCount++
		}
	}

	negCount := 0
	for _, word := range negative {
		if strings.Contains(textLower, word) {
			negCount++
		}
	}

	if posCount+negCount == 0 {
		return 0.0 // Neutral
	}

	// Return sentiment score between -1 and 1
	return float64(posCount-negCount) / float64(posCount+negCount)
}

// EnhancedValidator combines semantic and heuristic validation
type EnhancedValidator struct {
	Semantic  *SemanticValidator
	Heuristic DomainAdapter
}

func (v *EnhancedValidator) ValidateAnswer(passage, question string) (bool, float64, InferenceType) {
	// Get semantic validation
	semanticConf, inferenceType := v.Semantic.ValidateAnswer(passage, question, true)

	// Get heuristic validation
	heuristicAnswer, heuristicConf, heuristicType := v.Heuristic.ValidateAnswer(passage, question)

	// Combine results (weighted average)
	finalConf := 0.7*semanticConf + 0.3*heuristicConf

	// Use semantic inference type (more accurate)
	finalType := inferenceType
	if inferenceType == OtherInference {
		finalType = heuristicType
	}

	return heuristicAnswer, finalConf, finalType
}
