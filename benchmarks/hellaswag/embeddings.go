package hellaswag

import (
	"hash/fnv"
	"math"
	"strings"
)

// EmbeddingGenerator creates vector representations of text
type EmbeddingGenerator struct {
	ModelName  string
	Dimensions int
}

// NewEmbeddingGenerator creates an embedding generator
func NewEmbeddingGenerator(modelName string) *EmbeddingGenerator {
	dims := 384 // Standard for sentence embeddings
	if modelName == "gemmavault" {
		dims = 512 // Higher capacity
	}

	return &EmbeddingGenerator{
		ModelName:  modelName,
		Dimensions: dims,
	}
}

// Embed converts text to vector representation
func (e *EmbeddingGenerator) Embed(text string) []float64 {
	// Create embedding using TF-IDF-like approach with semantic features
	embedding := make([]float64, e.Dimensions)

	if e.Dimensions == 0 {
		return embedding
	}

	tokens := tokenizeText(text)
	tokenCount := len(tokens)

	tokenDim := e.Dimensions / 3
	if tokenDim == 0 {
		tokenDim = e.Dimensions
	}

	shortCount, mediumCount, longCount := 0, 0, 0
	uniqueTokens := 0

	if tokenCount > 0 {
		tokenCounts := make(map[string]int, tokenCount)
		for _, token := range tokens {
			tokenCounts[token]++
			tokenLen := len(token)
			switch {
			case tokenLen <= 3:
				shortCount++
			case tokenLen <= 6:
				mediumCount++
			default:
				longCount++
			}
		}
		uniqueTokens = len(tokenCounts)

		for token, count := range tokenCounts {
			idx := hashToRange(token, tokenDim)
			if idx >= tokenDim {
				continue
			}
			embedding[idx] += e.tokenWeight(token, count, tokenCount) * 1.2
		}

		bigrams := extractBigrams(tokens)
		if len(bigrams) > 0 {
			bigramCounts := make(map[string]int, len(bigrams))
			for _, bigram := range bigrams {
				bigramCounts[bigram]++
			}

			bigramStart := tokenDim
			bigramDim := tokenDim
			for bigram, count := range bigramCounts {
				idx := hashToRange(bigram, bigramDim)
				target := bigramStart + idx
				if idx >= bigramDim || target >= len(embedding) {
					continue
				}
				embedding[target] += e.bigramWeight(bigram, count) * 0.6
			}
		}
	}

	semanticStart := tokenDim * 2
	if semanticStart < len(embedding) {
		embedding[semanticStart] = e.sentimentScore(text)
	}
	if semanticStart+1 < len(embedding) {
		embedding[semanticStart+1] = e.complexityScore(text)
	}
	if semanticStart+2 < len(embedding) {
		embedding[semanticStart+2] = e.formalityScore(text)
	}
	if semanticStart+3 < len(embedding) {
		embedding[semanticStart+3] = float64(tokenCount) / 100.0
	}
	if semanticStart+4 < len(embedding) && tokenCount > 0 {
		embedding[semanticStart+4] = float64(shortCount) / float64(tokenCount)
	}
	if semanticStart+5 < len(embedding) && tokenCount > 0 {
		embedding[semanticStart+5] = float64(mediumCount) / float64(tokenCount)
	}
	if semanticStart+6 < len(embedding) && tokenCount > 0 {
		embedding[semanticStart+6] = float64(longCount) / float64(tokenCount)
	}
	if semanticStart+7 < len(embedding) && tokenCount > 0 {
		embedding[semanticStart+7] = float64(uniqueTokens) / float64(tokenCount)
	}

	// Normalize
	return normalizeVector(embedding)
}

func (e *EmbeddingGenerator) tokenWeight(token string, count, total int) float64 {
	if total == 0 {
		return 0.0
	}

	tf := float64(count) / float64(total)

	idf := 1.0
	if len(token) > 6 {
		idf += 0.4
	} else if len(token) <= 3 {
		idf += 0.1
	}
	if isCommonWord(token) {
		idf *= 0.8
	} else {
		idf *= 1.3
	}

	boost := 1.0 + 0.15*float64(count-1)

	return tf * idf * boost
}

func (e *EmbeddingGenerator) bigramWeight(bigram string, count int) float64 {
	parts := strings.Split(bigram, " ")
	if len(parts) != 2 {
		return 0.0
	}

	weight := 1.0
	if isCommonWord(parts[0]) && isCommonWord(parts[1]) {
		weight = 0.2
	} else if isCommonWord(parts[0]) || isCommonWord(parts[1]) {
		weight = 0.6
	}

	if count > 1 {
		weight *= 1.0 + math.Log1p(float64(count-1))*0.5
	}

	return weight
}

func (e *EmbeddingGenerator) sentimentScore(text string) float64 {
	positive := []string{"good", "great", "excellent", "success", "improve", "better"}
	negative := []string{"bad", "poor", "fail", "worse", "problem", "difficult"}

	textLower := strings.ToLower(text)
	posCount, negCount := 0, 0

	for _, word := range positive {
		if strings.Contains(textLower, word) {
			posCount++
		}
	}
	for _, word := range negative {
		if strings.Contains(textLower, word) {
			negCount++
		}
	}

	if posCount+negCount == 0 {
		return 0.5 // Neutral
	}

	return float64(posCount) / float64(posCount+negCount)
}

func (e *EmbeddingGenerator) complexityScore(text string) float64 {
	words := strings.Fields(text)
	if len(words) == 0 {
		return 0.0
	}

	// Average word length as complexity proxy
	totalLen := 0
	for _, word := range words {
		totalLen += len(word)
	}

	avgLen := float64(totalLen) / float64(len(words))
	return math.Min(avgLen/10.0, 1.0) // Normalize to 0-1
}

func (e *EmbeddingGenerator) formalityScore(text string) float64 {
	formal := []string{"therefore", "however", "consequently", "furthermore", "moreover"}
	informal := []string{"gonna", "wanna", "yeah", "okay", "stuff"}

	textLower := strings.ToLower(text)
	formalCount, informalCount := 0, 0

	for _, word := range formal {
		if strings.Contains(textLower, word) {
			formalCount++
		}
	}
	for _, word := range informal {
		if strings.Contains(textLower, word) {
			informalCount++
		}
	}

	if formalCount+informalCount == 0 {
		return 0.5
	}

	return float64(formalCount) / float64(formalCount+informalCount)
}

// CosineSimilarity calculates similarity between two embeddings
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// EuclideanDistance calculates distance between embeddings
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.MaxFloat64
	}

	sum := 0.0
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}

	return math.Sqrt(sum)
}

func extractBigrams(tokens []string) []string {
	if len(tokens) < 2 {
		return nil
	}

	bigrams := make([]string, 0, len(tokens)-1)
	for i := 0; i < len(tokens)-1; i++ {
		bigrams = append(bigrams, tokens[i]+" "+tokens[i+1])
	}

	return bigrams
}

func isCommonWord(word string) bool {
	common := map[string]bool{
		"the": true, "be": true, "to": true, "of": true, "and": true,
		"in": true, "that": true, "have": true, "it": true, "for": true,
		"not": true, "on": true, "with": true, "he": true, "as": true,
		"you": true, "do": true, "at": true, "this": true, "but": true,
	}

	return common[strings.ToLower(word)]
}

func hashToRange(text string, size int) int {
	if size <= 0 {
		return 0
	}

	h := fnv.New32a()
	_, _ = h.Write([]byte(text))
	return int(h.Sum32() % uint32(size))
}

func normalizeVector(vec []float64) []float64 {
	norm := 0.0
	for _, v := range vec {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if norm == 0 {
		return vec
	}

	normalized := make([]float64, len(vec))
	for i, v := range vec {
		normalized[i] = v / norm
	}

	return normalized
}

// EmbeddingBasedDiscriminator uses embeddings for similarity
type EmbeddingBasedDiscriminator struct {
	Generator *EmbeddingGenerator
	Threshold float64
}

func NewEmbeddingBasedDiscriminator(modelName string) *EmbeddingBasedDiscriminator {
	return &EmbeddingBasedDiscriminator{
		Generator: NewEmbeddingGenerator(modelName),
		Threshold: 0.5, // Sweet spot for confusion
	}
}

func (d *EmbeddingBasedDiscriminator) Score(context, ending string) (float64, error) {
	// Get embeddings
	ctxEmbed := d.Generator.Embed(context)
	endEmbed := d.Generator.Embed(ending)

	// Calculate similarity
	similarity := CosineSimilarity(ctxEmbed, endEmbed)

	// Confusion score: plausible but not too similar
	// Sweet spot around 0.5-0.7 similarity
	confusion := 1.0 - math.Abs(similarity-0.6)

	// Adjust for plausibility
	plausibility := calculatePlausibility(ending)
	confusion *= plausibility

	return confusion, nil
}

func calculatePlausibility(ending string) float64 {
	score := 0.0

	if hasVerbForm(ending) {
		score += 0.3
	}
	if hasSubjectForm(ending) {
		score += 0.3
	}
	if len(ending) > 10 && len(ending) < 200 {
		score += 0.2
	}
	if isCompleteSentence(ending) {
		score += 0.2
	}

	return score
}
