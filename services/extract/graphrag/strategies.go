package graphrag

import (
	"fmt"
)

// BreadthFirstStrategy implements breadth-first traversal.
type BreadthFirstStrategy struct{}

// NewBreadthFirstStrategy creates a new breadth-first traversal strategy.
func NewBreadthFirstStrategy() *BreadthFirstStrategy {
	return &BreadthFirstStrategy{}
}

// Name returns the strategy name.
func (bfs *BreadthFirstStrategy) Name() string {
	return "breadth_first"
}

// BuildQuery builds a Cypher query for breadth-first traversal.
func (bfs *BreadthFirstStrategy) BuildQuery(query string, maxDepth int, maxResults int) (string, map[string]interface{}) {
	// Match nodes that match the query text in labels or properties
	cypher := `
		MATCH path = (start:Node)
		WHERE toLower(start.label) CONTAINS toLower($query)
		   OR any(prop in keys(start) WHERE toLower(toString(start[prop])) CONTAINS toLower($query))
		WITH start, path
		LIMIT $maxResults
		MATCH path = (start)-[*1..%d]-(connected:Node)
		RETURN DISTINCT path
		LIMIT $maxResults
	`

	cypher = fmt.Sprintf(cypher, maxDepth)

	params := map[string]interface{}{
		"query":      query,
		"maxResults": maxResults,
	}

	return cypher, params
}

// DepthFirstStrategy implements depth-first traversal.
type DepthFirstStrategy struct{}

// NewDepthFirstStrategy creates a new depth-first traversal strategy.
func NewDepthFirstStrategy() *DepthFirstStrategy {
	return &DepthFirstStrategy{}
}

// Name returns the strategy name.
func (dfs *DepthFirstStrategy) Name() string {
	return "depth_first"
}

// BuildQuery builds a Cypher query for depth-first traversal.
func (dfs *DepthFirstStrategy) BuildQuery(query string, maxDepth int, maxResults int) (string, map[string]interface{}) {
	// Similar to BFS but optimized for depth-first exploration
	cypher := `
		MATCH path = (start:Node)
		WHERE toLower(start.label) CONTAINS toLower($query)
		   OR any(prop in keys(start) WHERE toLower(toString(start[prop])) CONTAINS toLower($query))
		WITH start
		LIMIT 1
		MATCH path = (start)-[*1..%d]-(connected:Node)
		RETURN DISTINCT path
		ORDER BY length(path) DESC
		LIMIT $maxResults
	`

	cypher = fmt.Sprintf(cypher, maxDepth)

	params := map[string]interface{}{
		"query":      query,
		"maxResults": maxResults,
	}

	return cypher, params
}

// WeightedTraversalStrategy implements weighted traversal based on edge properties.
type WeightedTraversalStrategy struct{}

// NewWeightedTraversalStrategy creates a new weighted traversal strategy.
func NewWeightedTraversalStrategy() *WeightedTraversalStrategy {
	return &WeightedTraversalStrategy{}
}

// Name returns the strategy name.
func (wts *WeightedTraversalStrategy) Name() string {
	return "weighted"
}

// BuildQuery builds a Cypher query for weighted traversal.
func (wts *WeightedTraversalStrategy) BuildQuery(query string, maxDepth int, maxResults int) (string, map[string]interface{}) {
	// Use edge weights to prioritize paths
	cypher := `
		MATCH path = (start:Node)
		WHERE toLower(start.label) CONTAINS toLower($query)
		   OR any(prop in keys(start) WHERE toLower(toString(start[prop])) CONTAINS toLower($query))
		WITH start
		LIMIT 10
		MATCH path = (start)-[rels:*1..%d]-(connected:Node)
		WITH path, 
		     reduce(weight = 0.0, rel in rels | 
		       weight + coalesce(rel.weight, 1.0)
		     ) as totalWeight
		RETURN path
		ORDER BY totalWeight DESC
		LIMIT $maxResults
	`

	cypher = fmt.Sprintf(cypher, maxDepth)

	params := map[string]interface{}{
		"query":      query,
		"maxResults": maxResults,
	}

	return cypher, params
}

// SemanticTraversalStrategy implements semantic traversal using embeddings.
type SemanticTraversalStrategy struct{}

// NewSemanticTraversalStrategy creates a new semantic traversal strategy.
func NewSemanticTraversalStrategy() *SemanticTraversalStrategy {
	return &SemanticTraversalStrategy{}
}

// Name returns the strategy name.
func (sts *SemanticTraversalStrategy) Name() string {
	return "semantic"
}

// BuildQuery builds a Cypher query for semantic traversal.
func (sts *SemanticTraversalStrategy) BuildQuery(query string, maxDepth int, maxResults int) (string, map[string]interface{}) {
	// Use vector similarity search for semantic matching
	// Note: This assumes embeddings are stored in node properties
	cypher := `
		MATCH (start:Node)
		WHERE start.embedding IS NOT NULL
		WITH start, 
		     gds.similarity.cosine(
		       start.embedding,
		       $queryEmbedding
		     ) as similarity
		WHERE similarity > 0.7
		WITH start, similarity
		ORDER BY similarity DESC
		LIMIT 10
		MATCH path = (start)-[*1..%d]-(connected:Node)
		RETURN DISTINCT path
		LIMIT $maxResults
	`

	cypher = fmt.Sprintf(cypher, maxDepth)

	params := map[string]interface{}{
		"query":         query,
		"queryEmbedding": nil, // Would be generated from query embedding
		"maxResults":    maxResults,
	}

	return cypher, params
}

