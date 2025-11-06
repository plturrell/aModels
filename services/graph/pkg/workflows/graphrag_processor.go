package workflows

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/extract/graphrag"
)

// GraphRAGRequest represents a GraphRAG query request.
type GraphRAGRequest struct {
	Query      string            `json:"query"`                // Natural language or Cypher query
	Strategy   string            `json:"strategy,omitempty"`   // "breadth_first", "depth_first", "weighted", "semantic"
	MaxDepth   int               `json:"max_depth,omitempty"`  // Maximum traversal depth
	MaxResults int               `json:"max_results,omitempty"` // Maximum number of results
	Params     map[string]interface{} `json:"params,omitempty"` // Query parameters
	Enrich     bool              `json:"enrich,omitempty"`     // Whether to enrich answer with context
}

// GraphRAGResponse represents a GraphRAG query response.
type GraphRAGResponse struct {
	Nodes          []graphrag.GraphNode         `json:"nodes"`
	Edges          []graphrag.GraphEdge         `json:"edges"`
	Answer         string                       `json:"answer,omitempty"`
	EnrichedAnswer *graphrag.EnrichedAnswer     `json:"enriched_answer,omitempty"`
	Query          string                       `json:"query"`
	Strategy       string                       `json:"strategy"`
	Confidence     float64                      `json:"confidence,omitempty"`
}

// GraphRAGProcessorOptions configures the GraphRAG processor.
type GraphRAGProcessorOptions struct {
	Neo4jURI      string
	Neo4jUsername string
	Neo4jPassword string
	ExtractServiceURL string // For answer enrichment if needed
}

// ProcessGraphRAGNode returns a node that processes GraphRAG queries.
func ProcessGraphRAGNode(opts GraphRAGProcessorOptions) stategraph.NodeFunc {
	return wrapStateFunc(func(ctx context.Context, state map[string]any) (map[string]any, error) {
		// Extract GraphRAG request from state
		var graphragReq GraphRAGRequest

		if req, ok := state["graphrag_request"].(map[string]any); ok {
			graphragReq.Query = getString(req["query"])
			graphragReq.Strategy = getString(req["strategy"])
			if graphragReq.Strategy == "" {
				graphragReq.Strategy = "breadth_first" // Default strategy
			}
			graphragReq.MaxDepth = getInt(req["max_depth"])
			if graphragReq.MaxDepth == 0 {
				graphragReq.MaxDepth = 3 // Default depth
			}
			graphragReq.MaxResults = getInt(req["max_results"])
			if graphragReq.MaxResults == 0 {
				graphragReq.MaxResults = 50 // Default results
			}
			graphragReq.Params = parseMap(req["params"])
			graphragReq.Enrich = getBool(req["enrich"])
		} else {
			// Try to extract from unified request
			if unifiedReq, ok := state["unified_request"].(map[string]any); ok {
				if grReq, ok := unifiedReq["graphrag_request"].(map[string]any); ok {
					graphragReq.Query = getString(grReq["query"])
					graphragReq.Strategy = getString(grReq["strategy"])
					if graphragReq.Strategy == "" {
						graphragReq.Strategy = "breadth_first"
					}
					graphragReq.MaxDepth = getInt(grReq["max_depth"])
					if graphragReq.MaxDepth == 0 {
						graphragReq.MaxDepth = 3
					}
					graphragReq.MaxResults = getInt(grReq["max_results"])
					if graphragReq.MaxResults == 0 {
						graphragReq.MaxResults = 50
					}
					graphragReq.Params = parseMap(grReq["params"])
					graphragReq.Enrich = getBool(grReq["enrich"])
				}
			}
		}

		if graphragReq.Query == "" {
			log.Println("No GraphRAG query provided; skipping")
			return state, nil
		}

		// Get Neo4j connection details
		neo4jURI := opts.Neo4jURI
		if neo4jURI == "" {
			neo4jURI = os.Getenv("NEO4J_URI")
			if neo4jURI == "" {
				neo4jURI = "bolt://localhost:7687"
			}
		}

		neo4jUsername := opts.Neo4jUsername
		if neo4jUsername == "" {
			neo4jUsername = os.Getenv("NEO4J_USERNAME")
			if neo4jUsername == "" {
				neo4jUsername = "neo4j"
			}
		}

		neo4jPassword := opts.Neo4jPassword
		if neo4jPassword == "" {
			neo4jPassword = os.Getenv("NEO4J_PASSWORD")
			if neo4jPassword == "" {
				neo4jPassword = "password"
			}
		}

		log.Printf("Processing GraphRAG query: %s (strategy=%s, depth=%d)", 
			graphragReq.Query, graphragReq.Strategy, graphragReq.MaxDepth)

		// Connect to Neo4j
		driver, err := neo4j.NewDriverWithContext(neo4jURI, neo4j.AuthToken{
			Username: neo4jUsername,
			Password: neo4jPassword,
		})
		if err != nil {
			return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
		}
		defer driver.Close(ctx)

		// Create GraphRAG retriever
		logger := log.New(os.Stdout, "[graphrag] ", log.LstdFlags|log.Lmsgprefix)
		retriever := graphrag.NewNeo4jGraphRetriever(driver, logger)

		// Determine if query is Cypher or natural language
		isCypher := strings.Contains(strings.ToUpper(graphragReq.Query), "MATCH") ||
			strings.Contains(strings.ToUpper(graphragReq.Query), "RETURN") ||
			strings.Contains(strings.ToUpper(graphragReq.Query), "WHERE")

		var nodes []graphrag.GraphNode
		var edges []graphrag.GraphEdge

		if isCypher {
			// Execute Cypher query directly
			session := driver.NewSession(ctx, neo4j.SessionConfig{})
			defer session.Close(ctx)

			result, err := session.Run(ctx, graphragReq.Query, graphragReq.Params)
			if err != nil {
				return nil, fmt.Errorf("failed to execute Cypher query: %w", err)
			}

			// Convert results to GraphRAG format
			nodeMap := make(map[string]*graphrag.GraphNode)
			edgeMap := make(map[string]bool)

			for result.Next(ctx) {
				record := result.Record()
				for _, value := range record.Values {
					if path, ok := value.(neo4j.Path); ok {
						// Extract nodes
						for _, node := range path.Nodes {
							nodeID := node.ElementId
							if _, exists := nodeMap[nodeID]; !exists {
								gn := &graphrag.GraphNode{
									ID:         nodeID,
									Labels:     node.Labels,
									Properties: nodePropsToMap(node.Props),
								}
								nodeMap[nodeID] = gn
								nodes = append(nodes, *gn)
							}
						}

						// Extract edges
						for _, rel := range path.Relationships {
							edgeID := rel.ElementId
							if !edgeMap[edgeID] {
								edge := graphrag.GraphEdge{
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
				return nil, fmt.Errorf("error processing Cypher result: %w", err)
			}
		} else {
			// Use GraphRAG retriever with traversal strategy
			nodes, edges, err = retriever.Retrieve(ctx, graphragReq.Query, graphragReq.Strategy, 
				graphragReq.MaxDepth, graphragReq.MaxResults)
			if err != nil {
				return nil, fmt.Errorf("failed to retrieve graph data: %w", err)
			}
		}

		// Create response
		response := GraphRAGResponse{
			Nodes:    nodes,
			Edges:    edges,
			Query:    graphragReq.Query,
			Strategy: graphragReq.Strategy,
		}

		// Enrich answer if requested
		if graphragReq.Enrich && len(nodes) > 0 {
			enricher := graphrag.NewAnswerEnricher(driver, logger)
			enriched, err := enricher.EnrichAnswer(ctx, "", nodes, edges, graphragReq.Query)
			if err == nil {
				response.EnrichedAnswer = enriched
				response.Confidence = enriched.Confidence
				response.Answer = enriched.Answer
			} else {
				log.Printf("Warning: Failed to enrich answer: %v", err)
			}
		}

		// Store results in state
		newState := make(map[string]any, len(state)+3)
		for k, v := range state {
			newState[k] = v
		}
		newState["graphrag_response"] = response
		newState["graphrag_nodes"] = nodes
		newState["graphrag_edges"] = edges

		log.Printf("GraphRAG query completed: %d nodes, %d edges", len(nodes), len(edges))

		return newState, nil
	})
}

// Helper functions for Neo4j property conversion
func nodePropsToMap(props map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range props {
		result[k] = v
	}
	return result
}

func relPropsToMap(props map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range props {
		result[k] = v
	}
	return result
}

