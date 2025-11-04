package hana

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/storage"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_Orchestration/schema"
)

// HANAVectorStore implements vector store interface using HANA
type HANAVectorStore struct {
	pool        *hanapool.Pool
	vectorStore *storage.VectorStore
}

// NewHANAVectorStore creates a new HANA vector store
func NewHANAVectorStore(pool *hanapool.Pool) (*HANAVectorStore, error) {
	vectorStore := storage.NewVectorStore(pool)

	// Ensure table exists
	ctx := context.Background()
	if err := vectorStore.CreateTable(ctx); err != nil {
		return nil, fmt.Errorf("failed to create vector table: %w", err)
	}

	return &HANAVectorStore{
		pool:        pool,
		vectorStore: vectorStore,
	}, nil
}

// AddDocuments adds documents to the vector store
func (h *HANAVectorStore) AddDocuments(ctx context.Context, documents []schema.Document, embeddings [][]float64) error {
	if len(documents) != len(embeddings) {
		return fmt.Errorf("documents and embeddings count mismatch")
	}

	for i, doc := range documents {
		// Convert document metadata
		metadata := make(map[string]string)
		if doc.Metadata != nil {
			for k, v := range doc.Metadata {
				metadata[k] = fmt.Sprintf("%v", v)
			}
		}

		// Add embedding
		_, err := h.vectorStore.InsertEmbedding(ctx, embeddings[i], doc.PageContent, metadata)
		if err != nil {
			return fmt.Errorf("failed to add document %d: %w", i, err)
		}
	}

	return nil
}

// AddTexts adds texts to the vector store
func (h *HANAVectorStore) AddTexts(ctx context.Context, texts []string, embeddings [][]float64, metadatas []map[string]interface{}) error {
	if len(texts) != len(embeddings) {
		return fmt.Errorf("texts and embeddings count mismatch")
	}

	for i, text := range texts {
		// Convert metadata
		metadata := make(map[string]string)
		if i < len(metadatas) && metadatas[i] != nil {
			for k, v := range metadatas[i] {
				metadata[k] = fmt.Sprintf("%v", v)
			}
		}

		// Add embedding
		_, err := h.vectorStore.InsertEmbedding(ctx, embeddings[i], text, metadata)
		if err != nil {
			return fmt.Errorf("failed to add text %d: %w", i, err)
		}
	}

	return nil
}

// SimilaritySearch performs similarity search
func (h *HANAVectorStore) SimilaritySearch(ctx context.Context, queryEmbedding []float64, k int) ([]schema.Document, error) {
	// Perform similarity search
	results, err := h.vectorStore.SimilaritySearch(ctx, queryEmbedding, k, 0.0)
	if err != nil {
		return nil, fmt.Errorf("failed to perform similarity search: %w", err)
	}

	// Convert results to documents
	documents := make([]schema.Document, len(results))
	for i, result := range results {
		// Convert metadata back to interface{}
		metadata := make(map[string]interface{})
		for k, v := range result.Metadata {
			metadata[k] = v
		}

		documents[i] = schema.Document{
			PageContent: result.Content,
			Metadata:    metadata,
		}
	}

	return documents, nil
}

// SimilaritySearchWithScore performs similarity search with scores
func (h *HANAVectorStore) SimilaritySearchWithScore(ctx context.Context, queryEmbedding []float64, k int) ([]schema.Document, []float64, error) {
	// Perform similarity search
	results, err := h.vectorStore.SimilaritySearch(ctx, queryEmbedding, k, 0.0)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to perform similarity search: %w", err)
	}

	// Convert results to documents and scores
	documents := make([]schema.Document, len(results))
	scores := make([]float64, len(results))

	for i, result := range results {
		// Convert metadata back to interface{}
		metadata := make(map[string]interface{})
		for k, v := range result.Metadata {
			metadata[k] = v
		}

		documents[i] = schema.Document{
			PageContent: result.Content,
			Metadata:    metadata,
		}
		scores[i] = result.Score
	}

	return documents, scores, nil
}

// Delete deletes documents by IDs
func (h *HANAVectorStore) Delete(ctx context.Context, ids []string) error {
	for _, id := range ids {
		// Convert string ID to int64
		var docID int64
		if _, err := fmt.Sscanf(id, "%d", &docID); err != nil {
			return fmt.Errorf("invalid document ID: %s", id)
		}

		err := h.vectorStore.DeleteEmbedding(ctx, docID)
		if err != nil {
			return fmt.Errorf("failed to delete document %s: %w", id, err)
		}
	}

	return nil
}

// GetDocument retrieves a document by ID
func (h *HANAVectorStore) GetDocument(ctx context.Context, id string) (schema.Document, error) {
	// Convert string ID to int64
	var docID int64
	if _, err := fmt.Sscanf(id, "%d", &docID); err != nil {
		return schema.Document{}, fmt.Errorf("invalid document ID: %s", id)
	}

	embedding, err := h.vectorStore.GetEmbedding(ctx, docID)
	if err != nil {
		return schema.Document{}, fmt.Errorf("failed to get document: %w", err)
	}

	// Convert metadata back to interface{}
	metadata := make(map[string]interface{})
	for k, v := range embedding.Metadata {
		metadata[k] = v
	}

	return schema.Document{
		PageContent: embedding.Content,
		Metadata:    metadata,
	}, nil
}

// UpdateDocument updates a document
func (h *HANAVectorStore) UpdateDocument(ctx context.Context, id string, document schema.Document, embedding []float64) error {
	// Convert string ID to int64
	var docID int64
	if _, err := fmt.Sscanf(id, "%d", &docID); err != nil {
		return fmt.Errorf("invalid document ID: %s", id)
	}

	// Convert document metadata
	metadata := make(map[string]string)
	if document.Metadata != nil {
		for k, v := range document.Metadata {
			metadata[k] = fmt.Sprintf("%v", v)
		}
	}

	err := h.vectorStore.UpdateEmbedding(ctx, docID, embedding, document.PageContent, metadata)
	if err != nil {
		return fmt.Errorf("failed to update document: %w", err)
	}

	return nil
}

// BatchInsertEmbeddings inserts multiple embeddings in a single transaction
func (h *HANAVectorStore) BatchInsertEmbeddings(ctx context.Context, embeddings []storage.Embedding) error {
	return h.vectorStore.BatchInsertEmbeddings(ctx, embeddings)
}

// GetStats returns vector store statistics
func (h *HANAVectorStore) GetStats(ctx context.Context) (map[string]interface{}, error) {
	// Get total count
	query := `SELECT COUNT(*) FROM embeddings`
	row := h.pool.QueryRow(ctx, query)

	var totalCount int64
	err := row.Scan(&totalCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get total count: %w", err)
	}

	// Get recent activity
	recentQuery := `SELECT COUNT(*) FROM embeddings WHERE created_at >= CURRENT_TIMESTAMP - INTERVAL 1 DAY`
	recentRow := h.pool.QueryRow(ctx, recentQuery)

	var recentCount int64
	err = recentRow.Scan(&recentCount)
	if err != nil {
		return nil, fmt.Errorf("failed to get recent count: %w", err)
	}

	return map[string]interface{}{
		"total_embeddings":  totalCount,
		"recent_embeddings": recentCount,
		"store_type":        "hana",
	}, nil
}
