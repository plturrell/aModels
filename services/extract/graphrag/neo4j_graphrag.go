package graphrag

import (
	"context"
	"fmt"
	"log"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Neo4jGraphRetriever implements GraphRAG retrieval for Neo4j knowledge graphs.
type Neo4jGraphRetriever struct {
	driver     neo4j.DriverWithContext
	logger     *log.Logger
	strategies map[string]TraversalStrategy
}

// NewNeo4jGraphRetriever creates a new Neo4j GraphRAG retriever.
// Accepts DriverWithContext which is what Neo4jPersistence provides.
func NewNeo4jGraphRetriever(driver neo4j.DriverWithContext, logger *log.Logger) *Neo4jGraphRetriever {
	retriever := &Neo4jGraphRetriever{
		driver:     driver,
		logger:     logger,
		strategies: make(map[string]TraversalStrategy),
	}

	// Register default strategies
	retriever.strategies["breadth_first"] = NewBreadthFirstStrategy()
	retriever.strategies["depth_first"] = NewDepthFirstStrategy()
	retriever.strategies["weighted"] = NewWeightedTraversalStrategy()
	retriever.strategies["semantic"] = NewSemanticTraversalStrategy()

	return retriever
}

// Retrieve retrieves relevant graph nodes and relationships for a query.
func (ngr *Neo4jGraphRetriever) Retrieve(ctx context.Context, query string, strategy string, maxDepth int, maxResults int) ([]GraphNode, []GraphEdge, error) {
	if ngr.logger != nil {
		ngr.logger.Printf("GraphRAG retrieval: query=%s, strategy=%s, maxDepth=%d", query, strategy, maxDepth)
	}

	// Get traversal strategy
	traversalStrategy, exists := ngr.strategies[strategy]
	if !exists {
		return nil, nil, fmt.Errorf("unknown traversal strategy: %s", strategy)
	}

	// Build Cypher query based on strategy
	cypherQuery, params := traversalStrategy.BuildQuery(query, maxDepth, maxResults)

	// Execute query
	session := ngr.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx, cypherQuery, params)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to execute GraphRAG query: %w", err)
	}

	// Collect nodes and edges
	var nodes []GraphNode
	var edges []GraphEdge
	nodeMap := make(map[string]*GraphNode)
	edgeMap := make(map[string]bool)

	for result.Next(ctx) {
		record := result.Record()
		
		// Extract nodes and edges from record
		for _, value := range record.Values {
			if path, ok := value.(neo4j.Path); ok {
				// Process path
				for _, node := range path.Nodes {
					nodeID := node.ElementId
					if _, exists := nodeMap[nodeID]; !exists {
						gn := &GraphNode{
							ID:         nodeID,
							Labels:     node.Labels,
							Properties: nodePropsToMap(node.Props),
						}
						nodeMap[nodeID] = gn
						nodes = append(nodes, *gn)
					}
				}

				for _, rel := range path.Relationships {
					edgeID := rel.ElementId
					if !edgeMap[edgeID] {
						edge := GraphEdge{
							ID:         edgeID,
							Type:       rel.Type,
							StartNode:  rel.StartElementId,
							EndNode:    rel.EndElementId,
							Properties: relPropsToMap(rel.Props),
						}
						edges = append(edges, edge)
						edgeMap[edgeID] = true
					}
				}
			}
		}
	}

	if err := result.Err(); err != nil {
		return nil, nil, fmt.Errorf("error processing result: %w", err)
	}

	if ngr.logger != nil {
		ngr.logger.Printf("GraphRAG retrieval completed: %d nodes, %d edges", len(nodes), len(edges))
	}

	return nodes, edges, nil
}

// GraphNode represents a node in the knowledge graph.
type GraphNode struct {
	ID         string                 `json:"id"`
	Labels     []string               `json:"labels"`
	Properties map[string]interface{} `json:"properties"`
}

// GraphEdge represents an edge in the knowledge graph.
type GraphEdge struct {
	ID         string                 `json:"id"`
	Type       string                 `json:"type"`
	StartNode  string                 `json:"start_node"`
	EndNode    string                 `json:"end_node"`
	Properties map[string]interface{} `json:"properties"`
}

// TraversalStrategy defines a graph traversal strategy.
type TraversalStrategy interface {
	BuildQuery(query string, maxDepth int, maxResults int) (string, map[string]interface{})
	Name() string
}

// nodePropsToMap converts Neo4j node properties to a map.
func nodePropsToMap(props map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range props {
		result[k] = v
	}
	return result
}

// relPropsToMap converts Neo4j relationship properties to a map.
func relPropsToMap(props map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range props {
		result[k] = v
	}
	return result
}

