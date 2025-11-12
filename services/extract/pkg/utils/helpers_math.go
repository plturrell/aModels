package utils

import "math"

func min(a, b float64) float64 {
	return math.Min(a, b)
}

func Sqrt(x float64) float64 {
	if x <= 0 {
		if x == 0 {
			return 0
		}
		return 0
	}
	return math.Sqrt(x)
}

func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct float64
	var normA, normB float64

	for i := range a {
		dotProduct += float64(a[i] * b[i])
		normA += float64(a[i] * a[i])
		normB += float64(b[i] * b[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}
