package utils

import (
	"math"
)

// CalculateEntropy calculates the Shannon entropy of a set of values
func CalculateEntropy(values []string) float64 {
	if len(values) == 0 {
		return 0.0
	}

	counts := make(map[string]int)
	for _, value := range values {
		counts[value]++
	}

	entropy := 0.0
	total := float64(len(values))
	for _, count := range counts {
		p := float64(count) / total
		entropy -= p * math.Log2(p)
	}

	return entropy
}

// CalculateKLDivergence calculates the Kullback-Leibler divergence between two distributions
func CalculateKLDivergence(p, q map[string]float64) float64 {
	klDivergence := 0.0
	for value, pValue := range p {
		qValue, ok := q[value]
		if !ok {
			qValue = 1e-10 // Add-one smoothing
		}
		klDivergence += pValue * math.Log2(pValue/qValue)
	}
	return klDivergence
}
