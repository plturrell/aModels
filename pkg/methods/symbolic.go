package methods

import "strings"

// ScoreOptionsSymbolic: lightweight symbolic heuristics using hand-crafted patterns.
// - Prefer options that complete or justify the prompt (contains connectors)
// - Penalize options with contradictions or hedges
func ScoreOptionsSymbolic(prompt string, options []string) []float64 {
	p := norm(prompt)
	scores := make([]float64, len(options))
	for i, opt := range options {
		o := norm(opt)
		s := 0.0
		// Completion markers
		if anyWord(o, []string{"to", "so", "because", "by", "when"}) {
			s += 0.5
		}
		// Temporal/causal phrases
		if strings.Contains(o, " in order ") || strings.Contains(o, " so that ") {
			s += 0.5
		}
		// Hedging penalty
		if anyWord(o, []string{"maybe", "perhaps", "sometimes"}) {
			s -= 0.3
		}
		// Redundancy with prompt
		if overlapCount(p, o) > 0 {
			s += 0.2
		}
		// Contradictions
		if (hasNeg(p) && !hasNeg(o)) || (!hasNeg(p) && hasNeg(o)) {
			s -= 0.4
		}
		scores[i] = s
	}
	return scores
}

func anyWord(s string, words []string) bool {
	for _, w := range words {
		if strings.Contains(s, " "+w+" ") {
			return true
		}
	}
	return false
}

func overlapCount(a, b string) int {
	as := tokenSet(a)
	bs := tokenSet(b)
	c := 0
	for t := range as {
		if _, ok := bs[t]; ok {
			c++
		}
	}
	return c
}
