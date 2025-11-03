package piqa

import (
	"context"
	"testing"
)

// TestPhraseEnumeration tests phrase enumeration logic
func TestPhraseEnumeration(t *testing.T) {
	tests := []struct {
		name     string
		text     string
		minLen   int
		maxLen   int
		expected int
	}{
		{
			name:     "simple sentence",
			text:     "The cat sat on the mat",
			minLen:   1,
			maxLen:   3,
			expected: 18, // All 1, 2, and 3-word phrases
		},
		{
			name:     "single word phrases",
			text:     "Hello world",
			minLen:   1,
			maxLen:   1,
			expected: 2,
		},
		{
			name:     "empty text",
			text:     "",
			minLen:   1,
			maxLen:   3,
			expected: 0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			phrases := enumeratePhrases(tt.text, tt.minLen, tt.maxLen)
			if len(phrases) != tt.expected {
				t.Errorf("expected %d phrases, got %d", tt.expected, len(phrases))
			}
		})
	}
}

// TestCosineSimilarity tests cosine similarity calculation
func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		vec1     []float32
		vec2     []float32
		expected float64
		delta    float64
	}{
		{
			name:     "identical vectors",
			vec1:     []float32{1.0, 0.0, 0.0},
			vec2:     []float32{1.0, 0.0, 0.0},
			expected: 1.0,
			delta:    0.001,
		},
		{
			name:     "orthogonal vectors",
			vec1:     []float32{1.0, 0.0},
			vec2:     []float32{0.0, 1.0},
			expected: 0.0,
			delta:    0.001,
		},
		{
			name:     "opposite vectors",
			vec1:     []float32{1.0, 0.0},
			vec2:     []float32{-1.0, 0.0},
			expected: -1.0,
			delta:    0.001,
		},
		{
			name:     "zero vector",
			vec1:     []float32{0.0, 0.0},
			vec2:     []float32{1.0, 1.0},
			expected: 0.0,
			delta:    0.001,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := float64(cosineSimilarity(tt.vec1, tt.vec2))
			if abs64(result-tt.expected) > tt.delta {
				t.Errorf("expected %f, got %f", tt.expected, result)
			}
		})
	}
}

// TestVectorSerialization tests vector serialization/deserialization
func TestVectorSerialization(t *testing.T) {
	tests := []struct {
		name   string
		vector []float32
	}{
		{
			name:   "simple vector",
			vector: []float32{1.0, 2.0, 3.0},
		},
		{
			name:   "zero vector",
			vector: []float32{0.0, 0.0, 0.0},
		},
		{
			name:   "negative values",
			vector: []float32{-1.5, 2.3, -0.7},
		},
		{
			name:   "large vector",
			vector: make([]float32, 768),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Serialize
			bytes := serializeVector(tt.vector)

			// Deserialize
			result := deserializeVector(bytes, len(tt.vector))

			// Compare
			if len(result) != len(tt.vector) {
				t.Errorf("length mismatch: expected %d, got %d", len(tt.vector), len(result))
			}

			for i := range tt.vector {
				if abs32(result[i]-tt.vector[i]) > 0.0001 {
					t.Errorf("value mismatch at index %d: expected %f, got %f", i, tt.vector[i], result[i])
				}
			}
		})
	}
}

// TestConfigValidation tests configuration validation
func TestConfigValidation(t *testing.T) {
	tests := []struct {
		name      string
		config    PIQAConfig
		expectErr bool
	}{
		{
			name:      "valid default config",
			config:    DefaultConfig(),
			expectErr: false,
		},
		{
			name: "invalid input size",
			config: PIQAConfig{
				Model: ModelConfig{InputSize: -1},
			},
			expectErr: true,
		},
		{
			name: "invalid learning rate",
			config: PIQAConfig{
				Model: ModelConfig{
					InputSize:    768,
					HiddenSize:   512,
					LearningRate: 1.5,
				},
			},
			expectErr: true,
		},
		{
			name: "invalid batch size",
			config: PIQAConfig{
				Model: ModelConfig{
					InputSize:    768,
					HiddenSize:   512,
					LearningRate: 0.001,
				},
				Retrieval: RetrievalConfig{TopK: 10},
				Performance: PerformanceConfig{
					BatchSize: -1,
				},
			},
			expectErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if (err != nil) != tt.expectErr {
				t.Errorf("expected error: %v, got: %v", tt.expectErr, err)
			}
		})
	}
}

// TestMetricsCollection tests metrics collector
func TestMetricsCollection(t *testing.T) {
	metrics := NewMetricsCollector()

	// Record some operations
	metrics.RecordQuery(true, 1000000) // 1ms
	metrics.RecordQuery(true, 2000000) // 2ms
	metrics.RecordQuery(false, 500000) // 0.5ms
	metrics.RecordCacheHit()
	metrics.RecordCacheMiss()
	metrics.RecordEmbedding(true, 5000000) // 5ms
	metrics.UpdateMemory(100)

	// Get snapshot
	snapshot := metrics.Snapshot()

	// Verify metrics
	if snapshot.TotalQueries != 3 {
		t.Errorf("expected 3 total queries, got %d", snapshot.TotalQueries)
	}
	if snapshot.SuccessfulQueries != 2 {
		t.Errorf("expected 2 successful queries, got %d", snapshot.SuccessfulQueries)
	}
	if snapshot.FailedQueries != 1 {
		t.Errorf("expected 1 failed query, got %d", snapshot.FailedQueries)
	}
	if snapshot.CacheHits != 1 {
		t.Errorf("expected 1 cache hit, got %d", snapshot.CacheHits)
	}
	if snapshot.CacheMisses != 1 {
		t.Errorf("expected 1 cache miss, got %d", snapshot.CacheMisses)
	}
	if snapshot.EmbeddingsGenerated != 1 {
		t.Errorf("expected 1 embedding generated, got %d", snapshot.EmbeddingsGenerated)
	}
	if snapshot.CurrentMemoryMB != 100 {
		t.Errorf("expected 100 MB memory, got %d", snapshot.CurrentMemoryMB)
	}
}

// TestLSHIndex tests LSH indexing
func TestLSHIndex(t *testing.T) {
	dimension := 128
	index := NewLSHIndex(5, 3, dimension)

	// Create test vectors
	vec1 := make([]float32, dimension)
	vec2 := make([]float32, dimension)
	vec3 := make([]float32, dimension)

	for i := 0; i < dimension; i++ {
		vec1[i] = float32(i) / float32(dimension)
		vec2[i] = float32(i)/float32(dimension) + 0.1       // Similar to vec1
		vec3[i] = float32(dimension-i) / float32(dimension) // Different
	}

	// Create retriever
	retriever := &ANNRetriever{
		index:        index,
		contextCache: make(map[string]*ContextEmbeddings),
		cacheEnabled: true,
	}

	// Index vectors
	ctx := context.Background()
	retriever.IndexContext(ctx, &ContextEmbeddings{
		ParagraphID: "p1",
		Phrases:     []Phrase{{Text: "test1", Start: 0, End: 10}},
		Embeddings:  [][]float32{vec1},
	})
	retriever.IndexContext(ctx, &ContextEmbeddings{
		ParagraphID: "p2",
		Phrases:     []Phrase{{Text: "test2", Start: 0, End: 10}},
		Embeddings:  [][]float32{vec2},
	})
	retriever.IndexContext(ctx, &ContextEmbeddings{
		ParagraphID: "p3",
		Phrases:     []Phrase{{Text: "test3", Start: 0, End: 10}},
		Embeddings:  [][]float32{vec3},
	})

	// Retrieve similar vectors
	results, err := retriever.RetrieveTopK(ctx, vec1, 2)
	if err != nil {
		t.Fatalf("retrieval failed: %v", err)
	}

	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}

	// First result should be vec1 itself
	if results[0].ParagraphID != "p1" {
		t.Errorf("expected first result to be p1, got %s", results[0].ParagraphID)
	}
}

// TestLogger tests logging functionality
func TestLogger(t *testing.T) {
	cfg := LoggingConfig{
		Level:  "debug",
		Format: "json",
		Output: "stdout",
	}

	logger := NewLogger(cfg)
	defer logger.Close()

	// Test different log levels
	logger.Debug("debug message")
	logger.Info("info message")
	logger.Warn("warning message")
	logger.Error("error message")

	// Test with fields
	logger.Info("message with fields", map[string]interface{}{
		"key1": "value1",
		"key2": 123,
	})

	// Test WithFields
	contextLogger := logger.WithFields(map[string]interface{}{
		"component": "test",
	})
	contextLogger.Info("context message")
}

// Helper functions

func enumeratePhrases(text string, minLen, maxLen int) []string {
	if text == "" {
		return nil
	}

	words := []string{}
	word := ""
	for _, r := range text {
		if r == ' ' {
			if word != "" {
				words = append(words, word)
				word = ""
			}
		} else {
			word += string(r)
		}
	}
	if word != "" {
		words = append(words, word)
	}

	var phrases []string
	for length := minLen; length <= maxLen; length++ {
		if len(words) == 0 {
			break
		}
		for i := 0; i < len(words); i++ {
			end := i + length
			if end > len(words) {
				end = len(words)
			}
			if end <= i {
				continue
			}
			if end-i < minLen {
				continue
			}

			phrase := ""
			for j := i; j < end; j++ {
				if j > i {
					phrase += " "
				}
				phrase += words[j]
			}
			phrases = append(phrases, phrase)
		}
	}

	return phrases
}

func abs32(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}

func abs64(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
