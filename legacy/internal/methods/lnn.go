package methods

import (
	"strings"
)

// ScoreOptionsLNN computes a simple logic-style compatibility between prompt and each option.
// Rules (pure maths):
// - Reward token overlap (content words)
// - Penalize contradictions with basic negation cues
// - Reward causal/explanatory markers for social/commonsense tasks
func ScoreOptionsLNN(prompt string, options []string) []float64 {
	p := norm(prompt)
	pt := tokenSet(p)
	scores := make([]float64, len(options))
	for i, opt := range options {
		o := norm(opt)
		ot := tokenSet(o)
		// Overlap score
		ov := 0.0
		for t := range ot {
			if _, ok := pt[t]; ok {
				ov += 1
			}
		}
		// Negation heuristic
		negP := hasNeg(p)
		negO := hasNeg(o)
		neg := 0.0
		if negP != negO {
			neg -= 0.5
		} else {
			neg += 0.1
		}
		// Causal markers
		caus := 0.0
		if hasAny(o, []string{"because", "so", "so that", "in order", "therefore", "thus"}) {
			caus += 0.3
		}
		scores[i] = ov + neg + caus
	}
	return scores
}

func norm(s string) string {
	s = strings.ToLower(s)
	repl := strings.NewReplacer(
		",", " ", ".", " ", ";", " ", ":", " ", "?", " ", "!", " ", "\n", " ", "\t", " ",
		"(", " ", ")", " ", "[", " ", "]", " ", "{", " ", "}", " ",
	)
	return repl.Replace(s)
}

func tokenSet(s string) map[string]struct{} {
	m := map[string]struct{}{}
	for _, w := range strings.Fields(s) {
		m[w] = struct{}{}
	}
	return m
}

func hasNeg(s string) bool { return hasAny(s, []string{" not ", "n't", " never ", "no "}) }
func hasAny(s string, keys []string) bool {
	for _, k := range keys {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}
