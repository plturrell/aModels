package graphrag

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// AnswerEnricher enriches answers with graph context and path tracing.
type AnswerEnricher struct {
	driver neo4j.DriverWithContext
	logger *log.Logger
}

// NewAnswerEnricher creates a new answer enricher.
func NewAnswerEnricher(driver neo4j.DriverWithContext, logger *log.Logger) *AnswerEnricher {
	return &AnswerEnricher{
		driver: driver,
		logger: logger,
	}
}

// EnrichAnswer enriches an answer with graph context.
func (ae *AnswerEnricher) EnrichAnswer(ctx context.Context, answer string, nodes []GraphNode, edges []GraphEdge, query string) (*EnrichedAnswer, error) {
	if ae.logger != nil {
		ae.logger.Printf("Enriching answer with %d nodes and %d edges", len(nodes), len(edges))
	}

	enriched := &EnrichedAnswer{
		Answer:       answer,
		OriginalQuery: query,
		GraphPaths:   []GraphPath{},
		Sources:      []SourceAttribution{},
		Confidence:   0.0,
		Explanation:  "",
	}

	// Extract graph paths
	paths := ae.extractPaths(nodes, edges)
	enriched.GraphPaths = paths

	// Generate source attributions
	sources := ae.generateSourceAttributions(nodes, edges)
	enriched.Sources = sources

	// Calculate confidence based on graph structure
	confidence := ae.calculateConfidence(nodes, edges)
	enriched.Confidence = confidence

	// Generate explanation
	explanation := ae.generateExplanation(answer, nodes, edges, paths)
	enriched.Explanation = explanation

	return enriched, nil
}

// extractPaths extracts graph paths from nodes and edges.
func (ae *AnswerEnricher) extractPaths(nodes []GraphNode, edges []GraphEdge) []GraphPath {
	var paths []GraphPath
	
	// Build node map
	nodeMap := make(map[string]*GraphNode)
	for i := range nodes {
		nodeMap[nodes[i].ID] = &nodes[i]
	}

	// Build adjacency list
	adjList := make(map[string][]string)
	for _, edge := range edges {
		adjList[edge.StartNode] = append(adjList[edge.StartNode], edge.EndNode)
	}

	// Find paths from source nodes to target nodes
	// Simplified: find all paths up to depth 3
	visited := make(map[string]bool)
	var dfs func(nodeID string, path []string, depth int)
	
	dfs = func(nodeID string, path []string, depth int) {
		if depth > 3 || visited[nodeID] {
			return
		}
		
		visited[nodeID] = true
		newPath := append(path, nodeID)
		
		if len(newPath) > 1 {
			paths = append(paths, GraphPath{
				Nodes:  newPath,
				Length: len(newPath),
			})
		}
		
		for _, neighbor := range adjList[nodeID] {
			dfs(neighbor, newPath, depth+1)
		}
		
		visited[nodeID] = false
	}

	// Start DFS from all nodes
	for _, node := range nodes {
		dfs(node.ID, []string{}, 0)
	}

	return paths
}

// generateSourceAttributions generates source attributions from graph nodes.
func (ae *AnswerEnricher) generateSourceAttributions(nodes []GraphNode, edges []GraphEdge) []SourceAttribution {
	var sources []SourceAttribution

	for _, node := range nodes {
		source := SourceAttribution{
			NodeID:     node.ID,
			Labels:     node.Labels,
			Properties: node.Properties,
			Relevance:  1.0, // Would calculate based on query similarity
		}

		// Extract source information from properties
		if sourceName, ok := node.Properties["source"].(string); ok {
			source.SourceName = sourceName
		}
		if sourceType, ok := node.Properties["source_type"].(string); ok {
			source.SourceType = sourceType
		}

		sources = append(sources, source)
	}

	return sources
}

// calculateConfidence calculates confidence score based on graph structure.
func (ae *AnswerEnricher) calculateConfidence(nodes []GraphNode, edges []GraphEdge) float64 {
	if len(nodes) == 0 {
		return 0.0
	}

	// Base confidence on:
	// 1. Number of nodes (more nodes = more context)
	// 2. Number of edges (more relationships = more connected)
	// 3. Node properties (more properties = more information)

	nodeScore := float64(len(nodes)) / 10.0 // Normalize to 0-1
	if nodeScore > 1.0 {
		nodeScore = 1.0
	}

	edgeScore := float64(len(edges)) / 20.0 // Normalize to 0-1
	if edgeScore > 1.0 {
		edgeScore = 1.0
	}

	// Weighted average
	confidence := (nodeScore*0.4 + edgeScore*0.6)
	if confidence > 1.0 {
		confidence = 1.0
	}

	return confidence
}

// generateExplanation generates an explanation of how the answer was derived.
func (ae *AnswerEnricher) generateExplanation(answer string, nodes []GraphNode, edges []GraphEdge, paths []GraphPath) string {
	var parts []string

	parts = append(parts, fmt.Sprintf("This answer was derived from %d nodes and %d relationships in the knowledge graph.", len(nodes), len(edges)))

	if len(paths) > 0 {
		parts = append(parts, fmt.Sprintf("Found %d relevant paths through the graph.", len(paths)))
	}

	if len(nodes) > 0 {
		// List key nodes
		nodeLabels := make(map[string]int)
		for _, node := range nodes {
			for _, label := range node.Labels {
				nodeLabels[label]++
			}
		}

		if len(nodeLabels) > 0 {
			parts = append(parts, "Key entities involved:")
			for label, count := range nodeLabels {
				parts = append(parts, fmt.Sprintf("  - %s (%d nodes)", label, count))
			}
		}
	}

	if len(edges) > 0 {
		// List key relationships
		relTypes := make(map[string]int)
		for _, edge := range edges {
			relTypes[edge.Type]++
		}

		if len(relTypes) > 0 {
			parts = append(parts, "Key relationships:")
			for relType, count := range relTypes {
				parts = append(parts, fmt.Sprintf("  - %s (%d relationships)", relType, count))
			}
		}
	}

	return strings.Join(parts, "\n")
}

// TraceLineage traces data lineage from a node.
func (ae *AnswerEnricher) TraceLineage(ctx context.Context, nodeID string, direction string, maxDepth int) (*LineageTrace, error) {
	if ae.logger != nil {
		ae.logger.Printf("Tracing lineage for node %s (direction: %s, depth: %d)", nodeID, direction, maxDepth)
	}

	// Build Cypher query based on direction
	var cypherQuery string
	switch direction {
	case "upstream":
		cypherQuery = fmt.Sprintf(`
			MATCH path = (start)
			WHERE start.id = $nodeID
			MATCH path = (start)<-[*1..%d]-(upstream)
			RETURN path
			LIMIT 50
		`, maxDepth)
	case "downstream":
		cypherQuery = fmt.Sprintf(`
			MATCH path = (start)
			WHERE start.id = $nodeID
			MATCH path = (start)-[*1..%d]->(downstream)
			RETURN path
			LIMIT 50
		`, maxDepth)
	case "both":
		cypherQuery = fmt.Sprintf(`
			MATCH path = (start)
			WHERE start.id = $nodeID
			MATCH path = (start)-[*1..%d]-(connected)
			RETURN path
			LIMIT 50
		`, maxDepth)
	default:
		return nil, fmt.Errorf("invalid direction: %s", direction)
	}

	session := ae.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx, cypherQuery, map[string]interface{}{"nodeID": nodeID})
	if err != nil {
		return nil, fmt.Errorf("failed to execute lineage query: %w", err)
	}

	var paths []GraphPath
	for result.Next(ctx) {
		record := result.Record()
		for _, value := range record.Values {
			if path, ok := value.(neo4j.Path); ok {
				var nodeIDs []string
				for _, node := range path.Nodes {
					nodeIDs = append(nodeIDs, node.ElementId)
				}
				paths = append(paths, GraphPath{
					Nodes:  nodeIDs,
					Length: len(nodeIDs),
				})
			}
		}
	}

	if err := result.Err(); err != nil {
		return nil, fmt.Errorf("error processing lineage result: %w", err)
	}

	return &LineageTrace{
		NodeID:    nodeID,
		Direction: direction,
		Paths:     paths,
		Depth:     maxDepth,
	}, nil
}

type EnrichedAnswer struct {
	Answer       string             `json:"answer"`
	OriginalQuery string            `json:"original_query"`
	GraphPaths   []GraphPath        `json:"graph_paths"`
	Sources      []SourceAttribution `json:"sources"`
	Confidence   float64            `json:"confidence"`
	Explanation  string             `json:"explanation"`
}

// GraphPath represents a path through the graph.
type GraphPath struct {
	Nodes  []string `json:"nodes"`
	Length int      `json:"length"`
}

// SourceAttribution represents attribution to a graph source.
type SourceAttribution struct {
	NodeID     string                 `json:"node_id"`
	Labels     []string               `json:"labels"`
	Properties map[string]interface{} `json:"properties"`
	SourceName string                 `json:"source_name,omitempty"`
	SourceType string                 `json:"source_type,omitempty"`
	Relevance  float64                 `json:"relevance"`
}

// LineageTrace represents a data lineage trace.
type LineageTrace struct {
	NodeID    string      `json:"node_id"`
	Direction string      `json:"direction"` // "upstream", "downstream", "both"
	Paths     []GraphPath `json:"paths"`
	Depth     int         `json:"depth"`
}

