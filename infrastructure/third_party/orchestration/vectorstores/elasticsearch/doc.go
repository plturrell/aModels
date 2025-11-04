// Package elasticsearch provides Elasticsearch-based vector storage for the Orchestration framework.
//
// The elasticsearch package implements vector similarity search using Elasticsearch's
// dense_vector field type and kNN search capabilities. It provides:
//
// - Vector similarity search with cosine similarity
// - Document storage and retrieval
// - Bulk operations for efficient data loading
// - Metadata filtering and search
// - Integration with Orchestration schema.Document
//
// Example usage:
//
//	// Create vector store
//	store := elasticsearch.NewElasticsearchVectorStore(
//		"http://localhost:9200",
//		"my_vectors",
//		"username",
//		"password",
//		768, // embedding dimension
//	)
//
//	// Initialize index
//	if err := store.Initialize(ctx); err != nil {
//		log.Fatal(err)
//	}
//
//	// Add documents
//	documents := []schema.Document{
//		{
//			ID: "1",
//			PageContent: "This is a sample document",
//			Metadata: map[string]interface{}{"source": "web"},
//			Embedding: []float64{0.1, 0.2, 0.3, ...},
//		},
//	}
//	store.AddDocuments(ctx, documents)
//
//	// Search for similar documents
//	results, err := store.SimilaritySearch(ctx, "sample query", 10)
//
// The vector store integrates with:
// - LocalAI for embedding generation
// - Orchestration chains for retrieval
// - HANA for metadata storage
// - Differential privacy for data protection
//
// Elasticsearch Configuration:
// - Requires Elasticsearch 8.0+ with kNN support
// - Uses dense_vector field type for embeddings
// - Configurable similarity metrics (cosine, dot_product, l2_norm)
// - Supports filtering and metadata search
package elasticsearch
