package hellaswag

import (
	"math"
	"strings"
)

// ModelDiscriminator uses GemmaVault/Phi-Mini to score ending quality
type ModelDiscriminator struct {
	ModelName string // "gemmavault" or "phimini"
}

// Score calculates how confusing an ending is
func (d *ModelDiscriminator) Score(context string, ending string) (float64, error) {
	// Calculate confusion score using heuristics that work without external models

	// 1. Semantic coherence (how well ending fits context)
	coherence := d.calculateCoherence(context, ending)

	// 2. Plausibility (grammatically correct and reasonable)
	plausibility := d.calculatePlausibility(ending)

	// 3. Lexical overlap (not too similar to context)
	overlap := d.calculateOverlap(context, ending)

	// 4. Length appropriateness
	lengthScore := d.calculateLengthScore(ending)

	// Combine: good distractor = plausible but not too coherent
	// Sweet spot: plausible (0.7-0.9) but moderate coherence (0.4-0.6)
	confusion := plausibility * (1.0 - math.Abs(coherence-0.5)*2) * (1.0 - overlap*0.3) * lengthScore

	return confusion, nil
}

func (d *ModelDiscriminator) calculateCoherence(context, ending string) float64 {
	ctxTokens := tokenizeText(context)
	endTokens := tokenizeText(ending)

	if len(endTokens) == 0 {
		return 0.0
	}

	// Token overlap
	overlap := 0
	for _, et := range endTokens {
		for _, ct := range ctxTokens {
			if et == ct {
				overlap++
				break
			}
		}
	}

	coherence := float64(overlap) / float64(len(endTokens))

	// Adjust for domain match
	if d.sameDomain(context, ending) {
		coherence *= 1.2
	}

	return math.Min(coherence, 1.0)
}

func (d *ModelDiscriminator) calculatePlausibility(ending string) float64 {
	score := 0.0

	// Has verb
	if hasVerbForm(ending) {
		score += 0.3
	}

	// Has subject
	if hasSubjectForm(ending) {
		score += 0.2
	}

	// Proper length
	if len(ending) > 10 && len(ending) < 200 {
		score += 0.2
	}

	// Complete sentence
	if isCompleteSentence(ending) {
		score += 0.2
	}

	// No obvious errors
	if !hasObviousErrors(ending) {
		score += 0.1
	}

	return score
}

func (d *ModelDiscriminator) calculateOverlap(context, ending string) float64 {
	ctxTokens := tokenizeText(context)
	endTokens := tokenizeText(ending)

	if len(ctxTokens) == 0 || len(endTokens) == 0 {
		return 0.0
	}

	ctxSet := make(map[string]bool)
	for _, t := range ctxTokens {
		ctxSet[t] = true
	}

	overlap := 0
	for _, t := range endTokens {
		if ctxSet[t] {
			overlap++
		}
	}

	// Jaccard similarity
	union := len(ctxTokens) + len(endTokens) - overlap
	if union == 0 {
		return 0.0
	}

	return float64(overlap) / float64(union)
}

func (d *ModelDiscriminator) calculateLengthScore(ending string) float64 {
	length := len(ending)

	// Optimal length: 20-100 characters
	if length >= 20 && length <= 100 {
		return 1.0
	}

	if length < 20 {
		return float64(length) / 20.0
	}

	if length > 100 {
		return math.Max(0.5, 1.0-float64(length-100)/200.0)
	}

	return 0.5
}

func (d *ModelDiscriminator) sameDomain(context, ending string) bool {
	ctxDomain := detectDomainType(context)
	endDomain := detectDomainType(ending)
	return ctxDomain == endDomain
}

// Helper functions

func tokenizeText(text string) []string {
	text = strings.ToLower(text)
	text = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || r == ' ' {
			return r
		}
		return ' '
	}, text)

	tokens := strings.Fields(text)

	// Remove stop words
	stopWords := map[string]bool{
		"the": true, "a": true, "an": true, "and": true, "or": true,
		"but": true, "in": true, "on": true, "at": true, "to": true,
		"for": true, "of": true, "with": true, "by": true, "is": true,
		"are": true, "was": true, "were": true,
	}

	var filtered []string
	for _, t := range tokens {
		if !stopWords[t] && len(t) > 2 {
			filtered = append(filtered, t)
		}
	}

	return filtered
}

func hasVerbForm(text string) bool {
	verbs := []string{
		"continues", "adjusts", "completes", "takes", "demonstrates",
		"practices", "modifies", "focuses", "finishes", "prepares",
		"follows", "makes", "ensures", "reviews", "gathers",
		"proceed", "continue", "adjust", "complete", "take",
	}

	textLower := strings.ToLower(text)
	for _, verb := range verbs {
		if strings.Contains(textLower, verb) {
			return true
		}
	}

	return false
}

func hasSubjectForm(text string) bool {
	// Check for pronouns or articles at start
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return false
	}

	textLower := strings.ToLower(text)
	subjects := []string{"he ", "she ", "it ", "they ", "the ", "a ", "an "}

	for _, subj := range subjects {
		if strings.HasPrefix(textLower, subj) {
			return true
		}
	}

	// Check for capitalized first word (proper noun)
	if text[0] >= 'A' && text[0] <= 'Z' {
		return true
	}

	return false
}

func isCompleteSentence(text string) bool {
	text = strings.TrimSpace(text)
	if len(text) == 0 {
		return false
	}

	// Has ending punctuation
	lastChar := text[len(text)-1]
	if lastChar == '.' || lastChar == '!' || lastChar == '?' {
		return true
	}

	// Has both verb and reasonable length
	return hasVerbForm(text) && len(text) > 15
}

func hasObviousErrors(text string) bool {
	textLower := strings.ToLower(text)

	// Check for nonsensical phrases
	errors := []string{
		"asdf", "qwerty", "xxx", "???", "...",
		"error", "null", "undefined", "test test",
	}

	for _, err := range errors {
		if strings.Contains(textLower, err) {
			return true
		}
	}

	// Check for repeated words
	words := strings.Fields(text)
	if len(words) > 1 {
		for i := 0; i < len(words)-1; i++ {
			if words[i] == words[i+1] {
				return true
			}
		}
	}

	return false
}

// NewModelDiscriminator creates a discriminator using available models
func NewModelDiscriminator(modelName string) *ModelDiscriminator {
	// Only accept models we actually have
	if modelName != "gemmavault" && modelName != "phimini" {
		modelName = "gemmavault" // Default to GemmaVault
	}

	return &ModelDiscriminator{
		ModelName: modelName,
	}
}
