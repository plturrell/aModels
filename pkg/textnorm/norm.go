package textnorm

import (
	"regexp"
	"strings"
	"unicode"
)

var articleRE = regexp.MustCompile(`\b(a|an|the)\b`)

// Normalize applies lowercasing, removes punctuation and articles, and collapses whitespace.
func Normalize(s string) string {
	s = strings.ToLower(s)
	// Remove punctuation
	s = strings.Map(func(r rune) rune {
		if unicode.IsPunct(r) {
			return -1
		}
		return r
	}, s)
	// Remove articles
	s = articleRE.ReplaceAllString(s, " ")
	// Collapse whitespace
	s = strings.Join(strings.Fields(s), " ")
	return s
}

func Tokens(s string) []string {
	n := Normalize(s)
	if n == "" {
		return nil
	}
	return strings.Fields(n)
}

// F1 computes token-level F1 between predicted and ground-truth answer strings.
func F1(pred, truth string) float64 {
	pt := Tokens(pred)
	tt := Tokens(truth)
	if len(pt) == 0 && len(tt) == 0 {
		return 1
	}
	if len(pt) == 0 || len(tt) == 0 {
		return 0
	}
	// Count overlaps
	m := map[string]int{}
	for _, t := range tt {
		m[t]++
	}
	common := 0
	for _, p := range pt {
		if m[p] > 0 {
			common++
			m[p]--
		}
	}
	if common == 0 {
		return 0
	}
	prec := float64(common) / float64(len(pt))
	rec := float64(common) / float64(len(tt))
	return 2 * prec * rec / (prec + rec)
}

// ExactMatch compares normalized strings.
func ExactMatch(pred, truth string) bool {
	return Normalize(pred) == Normalize(truth)
}
