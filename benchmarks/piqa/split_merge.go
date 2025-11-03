package piqa

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
)

// SplitDataset splits a SQuAD dataset into context-only and question-only files
// This enforces independence between document and question encoders
func SplitDataset(squadPath, contextPath, questionPath string) error {
	// Load SQuAD dataset
	data, err := os.ReadFile(squadPath)
	if err != nil {
		return fmt.Errorf("read squad file: %w", err)
	}

	var squad SQuADDataset
	if err := json.Unmarshal(data, &squad); err != nil {
		return fmt.Errorf("unmarshal squad: %w", err)
	}

	// Create context-only dataset
	contextOnly := ContextOnly{
		Version: squad.Version,
		Data:    make([]ContextOnlyArticle, 0),
	}

	// Create question-only dataset
	questionOnly := QuestionOnly{
		Version:   squad.Version,
		Questions: make([]QuestionOnlyItem, 0),
	}

	// Process each article
	for _, article := range squad.Data {
		contextArticle := ContextOnlyArticle{
			Title:      article.Title,
			Paragraphs: make([]ContextOnlyParagraph, 0),
		}

		// Process each paragraph
		for paraIdx, para := range article.Paragraphs {
			paragraphID := fmt.Sprintf("%s_%d", article.Title, paraIdx)

			// Add to context-only
			contextArticle.Paragraphs = append(contextArticle.Paragraphs, ContextOnlyParagraph{
				Context:     para.Context,
				ParagraphID: paragraphID,
			})

			// Add questions to question-only
			for _, qa := range para.QAs {
				questionOnly.Questions = append(questionOnly.Questions, QuestionOnlyItem{
					ID:          qa.ID,
					Question:    qa.Question,
					Answers:     qa.Answers,
					ParagraphID: paragraphID,
				})
			}
		}

		contextOnly.Data = append(contextOnly.Data, contextArticle)
	}

	// Write context-only file
	contextData, err := json.MarshalIndent(contextOnly, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}
	if err := os.WriteFile(contextPath, contextData, 0644); err != nil {
		return fmt.Errorf("write context file: %w", err)
	}

	// Write question-only file
	questionData, err := json.MarshalIndent(questionOnly, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal questions: %w", err)
	}
	if err := os.WriteFile(questionPath, questionData, 0644); err != nil {
		return fmt.Errorf("write question file: %w", err)
	}

	return nil
}

// MergePredictions merges context and question embeddings to produce predictions
func MergePredictions(contextEmbeddings map[string]*ContextEmbeddings,
	questionEmbeddings []*QuestionEmbedding,
	questionToParagraph map[string]string) (map[string]string, error) {

	// Create retriever
	retriever := NewPhraseRetriever(1) // top-1 retrieval

	// Index all context embeddings
	ctx := context.TODO()
	for _, ctxEmb := range contextEmbeddings {
		if err := retriever.IndexContext(ctx, ctxEmb); err != nil {
			return nil, fmt.Errorf("index context: %w", err)
		}
	}

	// Retrieve answers for all questions
	return retriever.RetrieveAll(ctx, questionEmbeddings, questionToParagraph)
}

// LoadContextEmbeddings loads context embeddings from directory
func LoadContextEmbeddings(contextEmbDir string) (map[string]*ContextEmbeddings, error) {
	// In a full implementation, this would load .npz and .json files
	// For now, return empty map as placeholder
	return make(map[string]*ContextEmbeddings), nil
}

// LoadQuestionEmbeddings loads question embeddings from directory
func LoadQuestionEmbeddings(questionEmbDir string) ([]*QuestionEmbedding, error) {
	// In a full implementation, this would load .npz files
	// For now, return empty slice as placeholder
	return make([]*QuestionEmbedding, 0), nil
}

// SaveContextEmbeddings saves context embeddings to directory
func SaveContextEmbeddings(contextEmb *ContextEmbeddings, outputDir string) error {
	// Save phrases as JSON
	phrasesPath := fmt.Sprintf("%s/%s.json", outputDir, contextEmb.ParagraphID)
	phrasesData, err := json.MarshalIndent(contextEmb.Phrases, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal phrases: %w", err)
	}
	if err := os.WriteFile(phrasesPath, phrasesData, 0644); err != nil {
		return fmt.Errorf("write phrases: %w", err)
	}

	// In a full implementation, save embeddings as .npz (numpy format)
	// For now, save as JSON for simplicity
	embPath := fmt.Sprintf("%s/%s_embeddings.json", outputDir, contextEmb.ParagraphID)
	embData, err := json.MarshalIndent(contextEmb.Embeddings, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal embeddings: %w", err)
	}
	if err := os.WriteFile(embPath, embData, 0644); err != nil {
		return fmt.Errorf("write embeddings: %w", err)
	}

	return nil
}

// SaveQuestionEmbedding saves question embedding to directory
func SaveQuestionEmbedding(questionEmb *QuestionEmbedding, outputDir string) error {
	// In a full implementation, save as .npz (numpy format)
	// For now, save as JSON for simplicity
	embPath := fmt.Sprintf("%s/%s.json", outputDir, questionEmb.QuestionID)
	embData, err := json.MarshalIndent(questionEmb, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal question embedding: %w", err)
	}
	if err := os.WriteFile(embPath, embData, 0644); err != nil {
		return fmt.Errorf("write question embedding: %w", err)
	}

	return nil
}
