package piqa

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"time"

	"ai_benchmarks/internal/registry"
)

// Phrase-Indexed Question Answering (PIQA) - Seo et al. 2018, EMNLP
// Paper: https://arxiv.org/abs/1804.07726
// GitHub: https://github.com/seominjoon/piqa
//
// PIQA enforces complete independence between document encoder and question encoder
// for scalable document comprehension via phrase retrieval.

type runner struct{}

func (runner) ID() string {
	return "piqa"
}

func (runner) Description() string {
	return "PIQA: Phrase-Indexed Question Answering (SQuAD-based extractive QA); metric=EM/F1"
}

func (runner) DefaultMetric() string {
	return "exact_match"
}

func (runner) Run(ctx context.Context, opts registry.RunOptions) (*registry.Summary, error) {
	// Validate data path
	if fi, err := os.Stat(opts.DataPath); err != nil || fi.IsDir() {
		return nil, registry.Errf("--data=<path to SQuAD JSON>", "data must be a SQuAD JSON file: %v", err)
	}

	// Load SQuAD dataset
	squad, err := loadSQuADDataset(opts.DataPath)
	if err != nil {
		return nil, fmt.Errorf("load squad dataset: %w", err)
	}

	// Initialize encoder (simple bag-of-words for baseline)
	embeddingDim := 300
	if v, ok := opts.Params["embedding_dim"]; ok {
		embeddingDim = int(v)
	}
	encoder := NewSimpleEncoder(embeddingDim)

	// Initialize document and question encoders
	maxPhraseLen := 7
	if v, ok := opts.Params["max_phrase_len"]; ok {
		maxPhraseLen = int(v)
	}
	docEncoder := NewDocumentEncoder(encoder, maxPhraseLen)
	qEncoder := NewQuestionEncoder(encoder)

	// Initialize retriever
	retriever := NewPhraseRetriever(1) // top-1 retrieval

	started := time.Now().Unix()
	totalQuestions := 0
	correctEM := 0
	totalF1 := 0.0

	// Process each article
	for _, article := range squad.Data {
		for paraIdx, para := range article.Paragraphs {
			paragraphID := fmt.Sprintf("%s_%d", article.Title, paraIdx)

			// Encode context (document encoder)
			contextEmb, err := docEncoder.EncodeContext(ctx, paragraphID, para.Context)
			if err != nil {
				return nil, fmt.Errorf("encode context %s: %w", paragraphID, err)
			}

			// Index context embeddings
			if err := retriever.IndexContext(ctx, contextEmb); err != nil {
				return nil, fmt.Errorf("index context %s: %w", paragraphID, err)
			}

			// Process each question
			for _, qa := range para.QAs {
				if opts.Limit > 0 && totalQuestions >= opts.Limit {
					goto done
				}

				// Encode question (question encoder - independent of context)
				questionEmb, err := qEncoder.EncodeQuestion(ctx, qa.ID, qa.Question)
				if err != nil {
					return nil, fmt.Errorf("encode question %s: %w", qa.ID, err)
				}

				// Retrieve answer phrase
				results, err := retriever.Retrieve(ctx, questionEmb, paragraphID)
				if err != nil {
					return nil, fmt.Errorf("retrieve for question %s: %w", qa.ID, err)
				}

				if len(results) == 0 {
					totalQuestions++
					continue
				}

				prediction := results[0].PhraseText

				// Evaluate against gold answers
				if len(qa.Answers) > 0 {
					// Compute exact match
					em := false
					maxF1 := 0.0
					for _, ans := range qa.Answers {
						if normalizeAnswer(prediction) == normalizeAnswer(ans.Text) {
							em = true
						}
						f1 := computeF1(prediction, ans.Text)
						if f1 > maxF1 {
							maxF1 = f1
						}
					}

					if em {
						correctEM++
					}
					totalF1 += maxF1
				}

				totalQuestions++
			}
		}
	}

done:
	finished := time.Now().Unix()

	if totalQuestions == 0 {
		return nil, errors.New("no questions processed")
	}

	exactMatch := float64(correctEM) / float64(totalQuestions)
	f1Score := totalF1 / float64(totalQuestions)

	sum := &registry.Summary{
		Task:       "piqa",
		Model:      opts.Model,
		Count:      totalQuestions,
		Metrics:    map[string]float64{"exact_match": exactMatch, "f1": f1Score},
		StartedAt:  started,
		FinishedAt: finished,
		Details: map[string]any{
			"correct_em":      correctEM,
			"embedding_dim":   embeddingDim,
			"max_phrase_len":  maxPhraseLen,
			"total_questions": totalQuestions,
		},
	}

	return sum, nil
}

func init() {
	registry.Register(runner{})
}

// loadSQuADDataset loads a SQuAD format JSON file
func loadSQuADDataset(path string) (*SQuADDataset, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var squad SQuADDataset
	if err := json.Unmarshal(data, &squad); err != nil {
		return nil, err
	}

	return &squad, nil
}

// normalizeAnswer normalizes answer text for comparison
func normalizeAnswer(s string) string {
	s = strings.ToLower(s)
	s = strings.TrimSpace(s)

	// Remove articles
	s = strings.TrimPrefix(s, "a ")
	s = strings.TrimPrefix(s, "an ")
	s = strings.TrimPrefix(s, "the ")

	// Remove punctuation
	s = strings.Map(func(r rune) rune {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') || r == ' ' {
			return r
		}
		return -1
	}, s)

	// Normalize whitespace
	s = strings.Join(strings.Fields(s), " ")

	return s
}

// computeF1 computes F1 score between prediction and gold answer
func computeF1(prediction, gold string) float64 {
	predTokens := strings.Fields(normalizeAnswer(prediction))
	goldTokens := strings.Fields(normalizeAnswer(gold))

	if len(predTokens) == 0 || len(goldTokens) == 0 {
		if len(predTokens) == 0 && len(goldTokens) == 0 {
			return 1.0
		}
		return 0.0
	}

	// Count common tokens
	common := make(map[string]int)
	for _, token := range goldTokens {
		common[token]++
	}

	numCommon := 0
	for _, token := range predTokens {
		if common[token] > 0 {
			numCommon++
			common[token]--
		}
	}

	if numCommon == 0 {
		return 0.0
	}

	precision := float64(numCommon) / float64(len(predTokens))
	recall := float64(numCommon) / float64(len(goldTokens))

	f1 := 2 * (precision * recall) / (precision + recall)
	return f1
}
