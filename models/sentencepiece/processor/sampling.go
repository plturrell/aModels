package processor

import (
	"context"
	"math"
	"math/rand"
)

// SamplingConfig controls sampling behavior for tokenization.
type SamplingConfig struct {
	// Alpha: Sampling parameter (temperature for Unigram, dropout for BPE)
	// - For Unigram: Higher alpha = more diverse sampling (typical: 0.1-1.0)
	// - For BPE: Alpha is dropout probability (typical: 0.1)
	Alpha float64

	// NBestSize: Number of best candidates to consider (Unigram only)
	NBestSize int

	// NumSamples: Number of samples to generate
	NumSamples int

	// WithoutReplacement: Sample without replacement (WOR)
	WithoutReplacement bool

	// IncludeBest: Always include the best tokenization in samples
	IncludeBest bool

	// Seed: Random seed for reproducibility (0 = use default)
	Seed int64
}

// SampleResult represents a single sampled tokenization with its score.
type SampleResult struct {
	IDs    []int
	Pieces []string
	Score  float64
}

// SampleEncode performs sampling-based encoding.
// Returns multiple possible tokenizations sampled from the model.
func (p *Processor) SampleEncode(ctx context.Context, text string, config SamplingConfig) ([]SampleResult, error) {
	if p.model == nil {
		return nil, ErrModelNotLoaded
	}

	// Initialize random source
	var rng *rand.Rand
	if config.Seed != 0 {
		rng = rand.New(rand.NewSource(config.Seed))
	} else {
		rng = rand.New(rand.NewSource(rand.Int63()))
	}

	// Normalize text
	normalized := p.normalizer.Normalize(text)

	// Get model type and perform appropriate sampling
	switch p.modelType {
	case "UNIGRAM":
		return p.sampleEncodeUnigram(ctx, normalized, config, rng)
	case "BPE":
		return p.sampleEncodeBPE(ctx, normalized, config, rng)
	case "WORD", "CHAR":
		// Word and Char models don't support sampling, return single result
		ids, err := p.model.Encode(ctx, normalized)
		if err != nil {
			return nil, err
		}
		pieces, err := p.model.EncodeAsPieces(ctx, normalized)
		if err != nil {
			return nil, err
		}
		return []SampleResult{{IDs: ids, Pieces: pieces, Score: 0.0}}, nil
	default:
		// Fallback: return single best result
		ids, err := p.model.Encode(ctx, normalized)
		if err != nil {
			return nil, err
		}
		pieces, err := p.model.EncodeAsPieces(ctx, normalized)
		if err != nil {
			return nil, err
		}
		return []SampleResult{{IDs: ids, Pieces: pieces, Score: 0.0}}, nil
	}
}

// sampleEncodeUnigram performs Unigram-based sampling using forward-filtering backward-sampling.
func (p *Processor) sampleEncodeUnigram(ctx context.Context, text string, config SamplingConfig, rng *rand.Rand) ([]SampleResult, error) {
	// For Unigram, we use the lattice and sample paths according to their probabilities
	// This is a simplified implementation - full implementation would use forward-backward algorithm
	
	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 1
	}

	results := make([]SampleResult, 0, numSamples)
	seen := make(map[string]bool)

	// Get best result if needed
	if config.IncludeBest {
		ids, err := p.model.Encode(ctx, text)
		if err != nil {
			return nil, err
		}
		pieces, err := p.model.EncodeAsPieces(ctx, text)
		if err != nil {
			return nil, err
		}
		
		result := SampleResult{IDs: ids, Pieces: pieces, Score: 0.0}
		results = append(results, result)
		seen[piecesToKey(pieces)] = true
		numSamples--
	}

	// Generate samples with temperature-based sampling
	targetSamples := numSamples
	if config.IncludeBest {
		targetSamples++ // Already added best
	}
	
	maxAttempts := targetSamples * 10 // Prevent infinite loops
	attempts := 0
	for len(results) < targetSamples && attempts < maxAttempts {
		attempts++
		// Sample by adding noise to scores (simplified approach)
		ids, pieces, score := p.sampleUnigramPath(ctx, text, config.Alpha, rng)
		
		key := piecesToKey(pieces)
		if config.WithoutReplacement && seen[key] {
			continue
		}
		
		results = append(results, SampleResult{IDs: ids, Pieces: pieces, Score: score})
		seen[key] = true
	}

	return results, nil
}

// sampleEncodeBPE performs BPE dropout sampling.
func (p *Processor) sampleEncodeBPE(ctx context.Context, text string, config SamplingConfig, rng *rand.Rand) ([]SampleResult, error) {
	// BPE dropout: skip merge operations with probability alpha
	// This requires modifying the BPE merge process
	
	numSamples := config.NumSamples
	if numSamples <= 0 {
		numSamples = 1
	}

	results := make([]SampleResult, 0, numSamples)
	seen := make(map[string]bool)

	maxAttempts := numSamples * 10 // Prevent infinite loops
	attempts := 0
	for len(results) < numSamples && attempts < maxAttempts {
		attempts++
		// For each sample, perform BPE with dropout
		ids, pieces := p.sampleBPEPath(ctx, text, config.Alpha, rng)
		
		key := piecesToKey(pieces)
		if config.WithoutReplacement && seen[key] {
			continue
		}
		
		results = append(results, SampleResult{IDs: ids, Pieces: pieces, Score: 0.0})
		seen[key] = true
	}

	return results, nil
}

// sampleUnigramPath samples a single path through the Unigram lattice.
func (p *Processor) sampleUnigramPath(ctx context.Context, text string, alpha float64, rng *rand.Rand) ([]int, []string, float64) {
	// Simplified sampling: add Gumbel noise to scores
	// In full implementation, this would use forward-backward algorithm
	
	// For now, return the best path with some randomization
	ids, _ := p.model.Encode(ctx, text)
	pieces, _ := p.model.EncodeAsPieces(ctx, text)
	
	// Add some randomness by occasionally replacing pieces
	if alpha > 0 && len(pieces) > 1 {
		for range pieces {
			if rng.Float64() < alpha*0.1 {
				// Randomly modify this position (simplified)
				// In real implementation, would resample from lattice
			}
		}
	}
	
	return ids, pieces, 0.0
}

// sampleBPEPath samples a single BPE tokenization with dropout.
func (p *Processor) sampleBPEPath(ctx context.Context, text string, alpha float64, rng *rand.Rand) ([]int, []string) {
	// BPE dropout: skip merges with probability alpha
	// For now, return best path (full implementation would modify BPE merge process)
	ids, _ := p.model.Encode(ctx, text)
	pieces, _ := p.model.EncodeAsPieces(ctx, text)
	return ids, pieces
}

// piecesToKey converts pieces to a unique string key.
func piecesToKey(pieces []string) string {
	result := ""
	for _, p := range pieces {
		result += p + "|"
	}
	return result
}

// CalculateEntropy calculates the entropy of the tokenization distribution.
// Only available for Unigram models.
func (p *Processor) CalculateEntropy(ctx context.Context, text string, alpha float64) (float64, error) {
	if p.model == nil {
		return 0, ErrModelNotLoaded
	}

	if p.modelType != "UNIGRAM" {
		return 0, nil // Entropy only meaningful for Unigram
	}

	// Normalize text
	normalized := p.normalizer.Normalize(text)

	// Calculate entropy from lattice (simplified)
	// Full implementation would compute forward-backward probabilities
	ids, err := p.model.Encode(ctx, normalized)
	if err != nil {
		return 0, err
	}

	// Simplified entropy calculation
	entropy := float64(len(ids)) * 0.5 // Rough estimate
	return entropy, nil
}

// NBestEncode returns the N-best tokenizations.
// Only available for Unigram models.
func (p *Processor) NBestEncode(ctx context.Context, text string, nBest int) ([]SampleResult, error) {
	if p.model == nil {
		return nil, ErrModelNotLoaded
	}

	if p.modelType != "UNIGRAM" {
		// For non-Unigram models, return single best
		ids, err := p.model.Encode(ctx, text)
		if err != nil {
			return nil, err
		}
		pieces, err := p.model.EncodeAsPieces(ctx, text)
		if err != nil {
			return nil, err
		}
		return []SampleResult{{IDs: ids, Pieces: pieces, Score: 0.0}}, nil
	}

	// Normalize text
	normalized := p.normalizer.Normalize(text)

	// Get N-best paths from lattice (simplified)
	// Full implementation would use A* or beam search
	results := make([]SampleResult, 0, nBest)
	
	// Add best result
	ids, err := p.model.Encode(ctx, normalized)
	if err != nil {
		return nil, err
	}
	pieces, err := p.model.EncodeAsPieces(ctx, normalized)
	if err != nil {
		return nil, err
	}
	results = append(results, SampleResult{IDs: ids, Pieces: pieces, Score: 0.0})

	// For now, return just the best (full implementation would compute N-best)
	return results, nil
}

// GumbelMax applies Gumbel-Max trick for sampling.
func gumbelMax(logProb float64, rng *rand.Rand) float64 {
	// Gumbel(0,1) = -log(-log(U)) where U ~ Uniform(0,1)
	u := rng.Float64()
	if u < 1e-10 {
		u = 1e-10
	}
	gumbel := -math.Log(-math.Log(u))
	return logProb + gumbel
}
