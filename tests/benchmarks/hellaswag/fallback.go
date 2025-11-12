package hellaswag

import (
	"fmt"
)

// FallbackStrategy defines how to handle missing dependencies
type FallbackStrategy int

const (
	FallbackToHeuristics FallbackStrategy = iota
	FallbackToSimpler
	FallbackToCache
	FailFast
)

// ResilientDiscriminator handles missing model dependencies gracefully
type ResilientDiscriminator struct {
	Primary  BERTDiscriminator
	Fallback BERTDiscriminator
	Strategy FallbackStrategy
	Cache    *ScoreCache
}

func NewResilientDiscriminator(modelName string, strategy FallbackStrategy) *ResilientDiscriminator {
	return &ResilientDiscriminator{
		Primary:  NewEmbeddingBasedDiscriminator(modelName),
		Fallback: NewModelDiscriminator(modelName),
		Strategy: strategy,
		Cache:    NewScoreCache(),
	}
}

func (d *ResilientDiscriminator) Score(context, ending string) (float64, error) {
	// Try cache first
	if d.Strategy == FallbackToCache {
		cacheKey := context + "|" + ending
		if score, exists := d.Cache.Get(cacheKey); exists {
			return score, nil
		}
	}

	// Try primary discriminator
	score, err := d.Primary.Score(context, ending)
	if err == nil {
		// Cache successful result
		if d.Cache != nil {
			d.Cache.Set(context+"|"+ending, score)
		}
		return score, nil
	}

	// Handle failure based on strategy
	switch d.Strategy {
	case FallbackToHeuristics:
		return d.Fallback.Score(context, ending)

	case FallbackToSimpler:
		// Use simple string matching
		return simpleMatchScore(context, ending), nil

	case FallbackToCache:
		// Already tried cache, use heuristics
		return d.Fallback.Score(context, ending)

	case FailFast:
		return 0.0, fmt.Errorf("primary discriminator failed: %w", err)
	}

	return 0.0, err
}

func simpleMatchScore(context, ending string) float64 {
	// Very simple fallback: token overlap
	ctxTokens := tokenizeText(context)
	endTokens := tokenizeText(ending)

	if len(endTokens) == 0 {
		return 0.0
	}

	overlap := 0
	for _, et := range endTokens {
		for _, ct := range ctxTokens {
			if et == ct {
				overlap++
				break
			}
		}
	}

	// Moderate overlap = confusing
	overlapRatio := float64(overlap) / float64(len(endTokens))
	if overlapRatio > 0.3 && overlapRatio < 0.7 {
		return 0.6
	}

	return 0.4
}

// ResilientGenerator handles missing model dependencies
type ResilientGenerator struct {
	Primary  EndingGenerator
	Fallback EndingGenerator
	Strategy FallbackStrategy
}

func NewResilientGenerator(modelName string, strategy FallbackStrategy) *ResilientGenerator {
	return &ResilientGenerator{
		Primary:  NewModelBasedGenerator(modelName),
		Fallback: NewModelBasedGenerator("gemmavault"), // Always fallback to GemmaVault
		Strategy: strategy,
	}
}

func (g *ResilientGenerator) Generate(context string, numCandidates int) ([]string, error) {
	// Try primary generator
	candidates, err := g.Primary.Generate(context, numCandidates)
	if err == nil && len(candidates) > 0 {
		return candidates, nil
	}

	// Handle failure based on strategy
	switch g.Strategy {
	case FallbackToHeuristics, FallbackToSimpler:
		return g.Fallback.Generate(context, numCandidates)

	case FailFast:
		return nil, fmt.Errorf("primary generator failed: %w", err)

	default:
		return g.Fallback.Generate(context, numCandidates)
	}
}

// HealthCheck verifies model availability
type HealthCheck struct {
	ModelName string
}

func (h *HealthCheck) CheckDiscriminator() error {
	disc := NewModelDiscriminator(h.ModelName)
	_, err := disc.Score("test context", "test ending")
	return err
}

func (h *HealthCheck) CheckGenerator() error {
	gen := NewModelBasedGenerator(h.ModelName)
	candidates, err := gen.Generate("test context", 1)
	if err != nil {
		return err
	}
	if len(candidates) == 0 {
		return fmt.Errorf("generator returned no candidates")
	}
	return nil
}

func (h *HealthCheck) CheckAll() map[string]error {
	results := make(map[string]error)

	results["discriminator"] = h.CheckDiscriminator()
	results["generator"] = h.CheckGenerator()

	return results
}

// AdaptiveFilter automatically adjusts to available resources
type AdaptiveFilter struct {
	*AdversarialFilter
	Health   *HealthCheck
	Strategy FallbackStrategy
}

func NewAdaptiveFilter(modelName string) *AdaptiveFilter {
	health := &HealthCheck{ModelName: modelName}

	// Check what's available
	healthResults := health.CheckAll()

	// Choose strategy based on health
	strategy := FallbackToHeuristics
	if healthResults["discriminator"] != nil || healthResults["generator"] != nil {
		strategy = FallbackToSimpler
	}

	return &AdaptiveFilter{
		AdversarialFilter: &AdversarialFilter{
			Config:        DefaultAFConfig(),
			Generator:     NewResilientGenerator(modelName, strategy),
			Discriminator: NewResilientDiscriminator(modelName, strategy),
		},
		Health:   health,
		Strategy: strategy,
	}
}

func (f *AdaptiveFilter) GenerateDistractorsWithFallback(context, goldEnding string, numDistractors int) ([]FilteredEnding, error) {
	// Try normal generation
	distractors, err := f.GenerateDistractors(context, goldEnding, numDistractors)
	if err == nil && len(distractors) >= numDistractors {
		return distractors, nil
	}

	// Fallback: generate more candidates and be less strict
	f.Config.MinBERTConfusion *= 0.8 // Lower threshold
	f.Config.NumCandidates *= 2      // More candidates

	distractors, err = f.GenerateDistractors(context, goldEnding, numDistractors)
	if err != nil {
		return nil, fmt.Errorf("fallback generation failed: %w", err)
	}

	return distractors, nil
}
