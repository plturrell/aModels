package piqa

import (
	"context"
	"fmt"
	"math"
	"strings"
)

// Encoder interface for both document and question encoding
type Encoder interface {
	// Encode text into a vector representation
	Encode(ctx context.Context, text string) ([]float32, error)

	// Dimension returns the embedding dimension
	Dimension() int
}

// DocumentEncoder encodes context and enumerates phrase-vector pairs
type DocumentEncoder struct {
	encoder          Encoder
	phraseEnumerator *PhraseEnumerator
}

// NewDocumentEncoder creates a new document encoder
func NewDocumentEncoder(encoder Encoder, maxPhraseLen int) *DocumentEncoder {
	return &DocumentEncoder{
		encoder:          encoder,
		phraseEnumerator: NewPhraseEnumerator(maxPhraseLen, 1),
	}
}

// EncodeContext encodes a context paragraph and returns all phrase embeddings
func (de *DocumentEncoder) EncodeContext(ctx context.Context, paragraphID, contextText string) (*ContextEmbeddings, error) {
	// Enumerate all phrases
	phrases := de.phraseEnumerator.EnumeratePhrases(contextText)
	phrases = de.phraseEnumerator.FilterPhrases(phrases)

	if len(phrases) == 0 {
		return nil, fmt.Errorf("no valid phrases found in context")
	}

	// Encode each phrase
	embeddings := make([][]float32, len(phrases))
	for i, phrase := range phrases {
		emb, err := de.encoder.Encode(ctx, phrase.Text)
		if err != nil {
			return nil, fmt.Errorf("encode phrase %d: %w", i, err)
		}
		embeddings[i] = emb
	}

	return &ContextEmbeddings{
		ParagraphID: paragraphID,
		Phrases:     phrases,
		Embeddings:  embeddings,
	}, nil
}

// QuestionEncoder encodes questions into the same vector space
type QuestionEncoder struct {
	encoder Encoder
}

// NewQuestionEncoder creates a new question encoder
func NewQuestionEncoder(encoder Encoder) *QuestionEncoder {
	return &QuestionEncoder{
		encoder: encoder,
	}
}

// EncodeQuestion encodes a single question
func (qe *QuestionEncoder) EncodeQuestion(ctx context.Context, questionID, questionText string) (*QuestionEmbedding, error) {
	emb, err := qe.encoder.Encode(ctx, questionText)
	if err != nil {
		return nil, fmt.Errorf("encode question: %w", err)
	}

	return &QuestionEmbedding{
		QuestionID: questionID,
		Embedding:  emb,
	}, nil
}

// SimpleEncoder implements a basic bag-of-words + GloVe-style encoder
type SimpleEncoder struct {
	dimension int
	vocab     map[string][]float32
}

// NewSimpleEncoder creates a simple encoder with random embeddings
func NewSimpleEncoder(dimension int) *SimpleEncoder {
	return &SimpleEncoder{
		dimension: dimension,
		vocab:     make(map[string][]float32),
	}
}

// Encode implements the Encoder interface
func (se *SimpleEncoder) Encode(ctx context.Context, text string) ([]float32, error) {
	tokens := strings.Fields(strings.ToLower(text))
	if len(tokens) == 0 {
		return make([]float32, se.dimension), nil
	}

	// Average word embeddings
	result := make([]float32, se.dimension)
	count := 0

	for _, token := range tokens {
		// Get or create embedding for token
		emb, ok := se.vocab[token]
		if !ok {
			// Create random embedding (in practice, use GloVe/Word2Vec)
			emb = se.randomEmbedding(token)
			se.vocab[token] = emb
		}

		for i := range result {
			result[i] += emb[i]
		}
		count++
	}

	// Average
	if count > 0 {
		for i := range result {
			result[i] /= float32(count)
		}
	}

	// Normalize
	return normalize(result), nil
}

// Dimension returns the embedding dimension
func (se *SimpleEncoder) Dimension() int {
	return se.dimension
}

// randomEmbedding creates a deterministic "random" embedding based on token
func (se *SimpleEncoder) randomEmbedding(token string) []float32 {
	// Simple hash-based initialization for reproducibility
	hash := 0
	for _, r := range token {
		hash = hash*31 + int(r)
	}

	emb := make([]float32, se.dimension)
	for i := range emb {
		// Pseudo-random value based on hash
		hash = hash*1103515245 + 12345
		emb[i] = float32(hash%1000-500) / 500.0
	}

	return normalize(emb)
}

// normalize normalizes a vector to unit length
func normalize(vec []float32) []float32 {
	var norm float32
	for _, v := range vec {
		norm += v * v
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm > 0 {
		for i := range vec {
			vec[i] /= norm
		}
	}

	return vec
}

// cosineSimilarity computes cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}

	return dot // assumes normalized vectors
}
