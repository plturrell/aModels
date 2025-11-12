package piqa

import (
	"context"
	"fmt"
	"sort"
)

// PhraseRetriever performs nearest neighbor search to retrieve answer phrases
type PhraseRetriever struct {
	contextEmbeddings map[string]*ContextEmbeddings // paragraphID -> embeddings
	topK              int
}

// NewPhraseRetriever creates a new phrase retriever
func NewPhraseRetriever(topK int) *PhraseRetriever {
	if topK <= 0 {
		topK = 1
	}
	return &PhraseRetriever{
		contextEmbeddings: make(map[string]*ContextEmbeddings),
		topK:              topK,
	}
}

// IndexContext adds context embeddings to the retriever
func (pr *PhraseRetriever) IndexContext(ctx context.Context, embeddings *ContextEmbeddings) error {
	pr.contextEmbeddings[embeddings.ParagraphID] = embeddings
	return nil
}

// Retrieve finds the top-K most similar phrases for a question
func (pr *PhraseRetriever) Retrieve(ctx context.Context, questionEmb *QuestionEmbedding, paragraphID string) ([]RetrievalResult, error) {
	// Get context embeddings for the paragraph
	contextEmb, ok := pr.contextEmbeddings[paragraphID]
	if !ok {
		return nil, fmt.Errorf("paragraph %s not indexed", paragraphID)
	}

	// Compute similarity scores for all phrases
	type scoredPhrase struct {
		phrase Phrase
		score  float32
		index  int
	}

	scored := make([]scoredPhrase, len(contextEmb.Phrases))
	for i, phrase := range contextEmb.Phrases {
		score := cosineSimilarity(questionEmb.Embedding, contextEmb.Embeddings[i])
		scored[i] = scoredPhrase{
			phrase: phrase,
			score:  score,
			index:  i,
		}
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Return top-K results
	k := pr.topK
	if k > len(scored) {
		k = len(scored)
	}

	results := make([]RetrievalResult, k)
	for i := 0; i < k; i++ {
		results[i] = RetrievalResult{
			QuestionID:  questionEmb.QuestionID,
			PhraseText:  scored[i].phrase.Text,
			Score:       scored[i].score,
			ParagraphID: paragraphID,
			CharStart:   scored[i].phrase.Start,
			CharEnd:     scored[i].phrase.End,
		}
	}

	return results, nil
}

// RetrieveAll retrieves answers for all questions
func (pr *PhraseRetriever) RetrieveAll(ctx context.Context, questions []*QuestionEmbedding, questionToParagraph map[string]string) (map[string]string, error) {
	predictions := make(map[string]string)

	for _, qEmb := range questions {
		paragraphID, ok := questionToParagraph[qEmb.QuestionID]
		if !ok {
			return nil, fmt.Errorf("no paragraph mapping for question %s", qEmb.QuestionID)
		}

		results, err := pr.Retrieve(ctx, qEmb, paragraphID)
		if err != nil {
			return nil, fmt.Errorf("retrieve for question %s: %w", qEmb.QuestionID, err)
		}

		if len(results) > 0 {
			predictions[qEmb.QuestionID] = results[0].PhraseText
		} else {
			predictions[qEmb.QuestionID] = ""
		}
	}

	return predictions, nil
}

// BruteForceRetriever performs exhaustive nearest neighbor search
type BruteForceRetriever struct {
	*PhraseRetriever
}

// NewBruteForceRetriever creates a brute-force retriever
func NewBruteForceRetriever(topK int) *BruteForceRetriever {
	return &BruteForceRetriever{
		PhraseRetriever: NewPhraseRetriever(topK),
	}
}
