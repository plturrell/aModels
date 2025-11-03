package hellaswag

import (
	"fmt"
	"strings"
)

// AdversarialFilter implements the core HellaSwag innovation
// Generates and filters distractors that fool models but not humans

// AFConfig configures adversarial filtering
type AFConfig struct {
	MinBERTConfusion float64 // Minimum BERT confusion score (0-1)
	MaxHumanError    float64 // Maximum human error rate (0-1)
	NumCandidates    int     // Number of candidates to generate per ending
	FilterThreshold  float64 // Threshold for keeping distractors
}

// DefaultAFConfig returns recommended AF settings from the paper
func DefaultAFConfig() AFConfig {
	return AFConfig{
		MinBERTConfusion: 0.4,  // BERT should be confused
		MaxHumanError:    0.05, // Humans should rarely get it wrong
		NumCandidates:    10,   // Generate 10 candidates per slot
		FilterThreshold:  0.6,  // Keep top 60% confusing
	}
}

// AdversarialFilter generates challenging distractors
type AdversarialFilter struct {
	Config        AFConfig
	Generator     EndingGenerator
	Discriminator BERTDiscriminator
}

// EndingGenerator generates candidate endings
type EndingGenerator interface {
	Generate(context string, numCandidates int) ([]string, error)
}

// BERTDiscriminator scores how confusing an ending is
type BERTDiscriminator interface {
	Score(context string, ending string) (float64, error)
}

// FilteredEnding represents an ending that passed AF
type FilteredEnding struct {
	Text           string
	BERTScore      float64 // How much it fools BERT
	HumanValidated bool
	ConfusionRank  int
}

// GenerateDistractors creates adversarially filtered distractors
func (af *AdversarialFilter) GenerateDistractors(context, goldEnding string, numDistractors int) ([]FilteredEnding, error) {
	// Step 1: Generate candidates using language model
	candidates, err := af.Generator.Generate(context, af.Config.NumCandidates*numDistractors)
	if err != nil {
		return nil, fmt.Errorf("generation failed: %w", err)
	}

	// Step 2: Score each candidate with BERT discriminator
	scored := make([]FilteredEnding, 0, len(candidates))
	for _, candidate := range candidates {
		// Skip if too similar to gold
		if isTooSimilar(candidate, goldEnding) {
			continue
		}

		// Score with BERT
		score, err := af.Discriminator.Score(context, candidate)
		if err != nil {
			continue
		}

		// Keep if it fools BERT enough
		if score >= af.Config.MinBERTConfusion {
			scored = append(scored, FilteredEnding{
				Text:      candidate,
				BERTScore: score,
			})
		}
	}

	// Step 3: Rank by confusion and select top N
	sortByConfusion(scored)

	if len(scored) < numDistractors {
		return nil, fmt.Errorf("insufficient distractors after filtering: got %d, need %d", len(scored), numDistractors)
	}

	// Step 4: Return top N most confusing
	result := scored[:numDistractors]
	for i := range result {
		result[i].ConfusionRank = i + 1
	}

	return result, nil
}

// ValidateWithHumans checks if distractors are obviously wrong to humans
func (af *AdversarialFilter) ValidateWithHumans(context string, endings []FilteredEnding) ([]FilteredEnding, error) {
	// In production, this would use human annotation
	// For now, use heuristics that approximate human judgment
	validated := make([]FilteredEnding, 0, len(endings))

	for _, ending := range endings {
		if isObviouslyWrong(context, ending.Text) {
			ending.HumanValidated = true
			validated = append(validated, ending)
		}
	}

	return validated, nil
}

// Helper functions

func isTooSimilar(candidate, gold string) bool {
	// Simple similarity check - in production would use embeddings
	candWords := strings.Fields(strings.ToLower(candidate))
	goldWords := strings.Fields(strings.ToLower(gold))

	if len(candWords) == 0 || len(goldWords) == 0 {
		return false
	}

	// Check word overlap
	overlap := 0
	for _, cw := range candWords {
		for _, gw := range goldWords {
			if cw == gw {
				overlap++
				break
			}
		}
	}

	similarity := float64(overlap) / float64(len(goldWords))
	return similarity > 0.7 // More than 70% overlap
}

func isObviouslyWrong(context, ending string) bool {
	// Heuristics for obviously wrong endings
	// In production, this would be human annotation

	ending = strings.ToLower(ending)

	// Check for nonsensical actions
	nonsensical := []string{
		"eat the", "drink the", "fly to", "teleport",
		"magic", "impossible", "never", "cannot",
	}
	for _, phrase := range nonsensical {
		if strings.Contains(ending, phrase) {
			return true
		}
	}

	// Check for contradictions with context
	if strings.Contains(context, "painting") && strings.Contains(ending, "cooking") {
		return true
	}

	// More sophisticated checks would go here
	return false
}

func sortByConfusion(endings []FilteredEnding) {
	// Simple bubble sort by BERTScore (descending)
	n := len(endings)
	for i := 0; i < n-1; i++ {
		for j := 0; j < n-i-1; j++ {
			if endings[j].BERTScore < endings[j+1].BERTScore {
				endings[j], endings[j+1] = endings[j+1], endings[j]
			}
		}
	}
}

// DefaultGenerator uses ModelBasedGenerator with GemmaVault
type DefaultGenerator struct {
	*ModelBasedGenerator
}

func NewDefaultGenerator() *DefaultGenerator {
	return &DefaultGenerator{
		ModelBasedGenerator: NewModelBasedGenerator("gemmavault"),
	}
}

// DefaultDiscriminator uses ModelDiscriminator with GemmaVault
type DefaultDiscriminator struct {
	*ModelDiscriminator
}

func NewDefaultDiscriminator() *DefaultDiscriminator {
	return &DefaultDiscriminator{
		ModelDiscriminator: NewModelDiscriminator("gemmavault"),
	}
}
