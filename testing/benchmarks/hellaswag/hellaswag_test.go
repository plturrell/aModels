package hellaswag

import (
	"testing"
)

// TestAdversarialExamples tests known adversarial cases
func TestAdversarialExamples(t *testing.T) {
	tests := []struct {
		name          string
		context       string
		gold          string
		distractor    string
		shouldConfuse bool
	}{
		{
			name:          "Activity - Painting",
			context:       "A person is painting a wall with a brush.",
			gold:          "continues painting with smooth, even strokes.",
			distractor:    "stops painting and starts cooking dinner.",
			shouldConfuse: false, // Too different
		},
		{
			name:          "WikiHow - Cooking",
			context:       "To make pasta, first boil water in a large pot.",
			gold:          "Add salt to the boiling water and put in the pasta.",
			distractor:    "Add the pasta to cold water and wait.",
			shouldConfuse: true, // Plausible but wrong
		},
		{
			name:          "Subtle Error",
			context:       "The athlete is preparing for the high jump.",
			gold:          "takes a running start and leaps over the bar.",
			distractor:    "takes a running start and leaps under the bar.",
			shouldConfuse: true, // One word difference
		},
	}

	discriminator := NewEmbeddingBasedDiscriminator("gemmavault")

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			goldScore, _ := discriminator.Score(tt.context, tt.gold)
			distractorScore, _ := discriminator.Score(tt.context, tt.distractor)

			if tt.shouldConfuse {
				// Distractor should have moderate score (confusing)
				if distractorScore < 0.3 || distractorScore > 0.9 {
					t.Errorf("Distractor score %f not in confusing range [0.3, 0.9]", distractorScore)
				}
			} else {
				// Distractor should have low score (obviously wrong)
				if distractorScore > 0.5 {
					t.Errorf("Distractor score %f too high for obviously wrong answer", distractorScore)
				}
			}

			// Gold should generally score higher
			if goldScore < distractorScore-0.1 {
				t.Logf("Warning: Gold score %f lower than distractor %f", goldScore, distractorScore)
			}
		})
	}
}

// TestEmbeddingSimilarity tests embedding-based similarity
func TestEmbeddingSimilarity(t *testing.T) {
	gen := NewEmbeddingGenerator("gemmavault")

	tests := []struct {
		name   string
		text1  string
		text2  string
		minSim float64
		maxSim float64
	}{
		{
			name:   "Identical",
			text1:  "The cat sat on the mat.",
			text2:  "The cat sat on the mat.",
			minSim: 0.95,
			maxSim: 1.0,
		},
		{
			name:   "Similar",
			text1:  "The cat sat on the mat.",
			text2:  "The dog sat on the rug.",
			minSim: 0.6,
			maxSim: 0.9,
		},
		{
			name:   "Different",
			text1:  "The cat sat on the mat.",
			text2:  "Quantum physics is fascinating.",
			minSim: 0.0,
			maxSim: 0.4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			emb1 := gen.Embed(tt.text1)
			emb2 := gen.Embed(tt.text2)

			sim := CosineSimilarity(emb1, emb2)

			if sim < tt.minSim || sim > tt.maxSim {
				t.Errorf("Similarity %f not in range [%f, %f]", sim, tt.minSim, tt.maxSim)
			}
		})
	}
}

// TestModelGenerator tests ending generation
func TestModelGenerator(t *testing.T) {
	gen := NewModelBasedGenerator("gemmavault")

	tests := []struct {
		name          string
		context       string
		numCandidates int
	}{
		{
			name:          "Activity",
			context:       "A person is doing pushups in the gym.",
			numCandidates: 5,
		},
		{
			name:          "WikiHow",
			context:       "To bake a cake, first preheat the oven to 350 degrees.",
			numCandidates: 5,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			candidates, err := gen.Generate(tt.context, tt.numCandidates)
			if err != nil {
				t.Fatalf("Generate failed: %v", err)
			}

			if len(candidates) == 0 {
				t.Error("No candidates generated")
			}

			// Check candidates are plausible
			for i, cand := range candidates {
				if len(cand) < 5 {
					t.Errorf("Candidate %d too short: %s", i, cand)
				}
				if !hasVerbForm(cand) && !hasSubjectForm(cand) {
					t.Logf("Warning: Candidate %d may not be grammatical: %s", i, cand)
				}
			}
		})
	}
}

// TestDomainDetection tests domain classification
func TestDomainDetection(t *testing.T) {
	tests := []struct {
		text   string
		domain string
	}{
		{"A person is playing basketball.", "activity"},
		{"How to make a sandwich: First, get two slices of bread.", "wikihow"},
		{"The weather is nice today.", "generic"},
	}

	for _, tt := range tests {
		t.Run(tt.text, func(t *testing.T) {
			domain := detectDomainType(tt.text)
			if domain != tt.domain {
				t.Errorf("Expected domain %s, got %s", tt.domain, domain)
			}
		})
	}
}

// BenchmarkEmbedding benchmarks embedding generation
func BenchmarkEmbedding(b *testing.B) {
	gen := NewEmbeddingGenerator("gemmavault")
	text := "The quick brown fox jumps over the lazy dog."

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		gen.Embed(text)
	}
}

// BenchmarkSimilarity benchmarks similarity calculation
func BenchmarkSimilarity(b *testing.B) {
	gen := NewEmbeddingGenerator("gemmavault")
	emb1 := gen.Embed("The cat sat on the mat.")
	emb2 := gen.Embed("The dog sat on the rug.")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		CosineSimilarity(emb1, emb2)
	}
}
