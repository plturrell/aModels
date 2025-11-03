package models

import (
	"strings"

	"ai_benchmarks/internal/mathvec"
)

// MathsMCQ selects the option with the highest cosine similarity to the prompt.
// prompt is typically question or question+context; options are candidate answers.
func MathsMCQ(prompt string, options []string, dim int) int {
	if len(options) == 0 {
		return -1
	}
	v := mathvec.NewVectorizer(dim)
	qp := v.Vec(prompt)
	best, bestScore := 0, -1.0
	for i, opt := range options {
		score := mathvec.Cosine(qp, v.Vec(opt))
		if score > bestScore {
			best, bestScore = i, score
		}
	}
	return best
}

// MathsYesNo reduces yes/no to MCQ with prompt vs {"yes","no"}.
func MathsYesNo(prompt string, dim int) bool {
	v := mathvec.NewVectorizer(dim)
	qp := v.Vec(prompt)
	y := v.Vec("yes")
	n := v.Vec("no")
	sy := mathvec.Cosine(qp, y)
	sn := mathvec.Cosine(qp, n)
	predYes := sy >= sn
	// Negation-aware flip when ambiguous
	lower := strings.ToLower(prompt)
	hasNeg := strings.Contains(lower, " not ") || strings.Contains(lower, "n't") || strings.Contains(lower, " never ")
	if hasNeg && mathAbs(sy-sn) < 0.15 {
		predYes = !predYes
	}
	return predYes
}

func mathAbs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

// JoinPrompt concatenates strings with space for vectorization.
func JoinPrompt(parts ...string) string { return strings.Join(parts, " ") }
