package workflows

import (
	"os"

	"github.com/langchain-ai/langgraph-go/pkg/stategraph"
)

// NewGNNProcessorWorkflow creates a standalone workflow for GNN queries.
//
// This workflow supports:
// - Direct GNN queries (embeddings, classify, predict-links, structural-insights)
// - Hybrid queries (KG + GNN)
//
// State Input:
//   - gnn_query_request: GNNQueryRequest for direct GNN queries
//   - hybrid_query_request: HybridQueryRequest for hybrid queries
//
// State Output:
//   - gnn_result: GNN query results
//   - hybrid_result: Combined KG + GNN results
func NewGNNProcessorWorkflow(opts GNNProcessorOptions) (*stategraph.CompiledStateGraph, error) {
	extractServiceURL := opts.ExtractServiceURL
	if extractServiceURL == "" {
		extractServiceURL = os.Getenv("EXTRACT_SERVICE_URL")
		if extractServiceURL == "" {
			extractServiceURL = "http://extract-service:19080"
		}
	}

	trainingServiceURL := opts.TrainingServiceURL
	if trainingServiceURL == "" {
		trainingServiceURL = os.Getenv("TRAINING_SERVICE_URL")
		if trainingServiceURL == "" {
			trainingServiceURL = "http://training-service:8080"
		}
	}

	gnnOpts := GNNProcessorOptions{
		TrainingServiceURL: trainingServiceURL,
		ExtractServiceURL:  extractServiceURL,
	}

	// Define workflow nodes
	nodes := map[string]stategraph.NodeFunc{
		"query_gnn":    QueryGNNNode(gnnOpts),
		"hybrid_query": HybridQueryNode(gnnOpts),
	}

	// Define edges
	edges := []EdgeSpec{
		{From: "query_gnn", To: "hybrid_query", Label: "to_hybrid"},
	}

	// Build workflow
	return BuildGraphWithOptions("query_gnn", "hybrid_query", nodes, edges, nil, nil)
}

