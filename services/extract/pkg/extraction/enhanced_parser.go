package extraction

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/extract/pkg/ai"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// EnhancedParser wraps existing parsers with AI semantic analysis
type EnhancedParser struct {
	baseParser interface{} // Can be DDL parser, SQL parser, etc.
	cwmClient  *ai.CWMClient
	logger     *log.Logger
}

// NewEnhancedParser creates a new enhanced parser
func NewEnhancedParser(baseParser interface{}, cwmClient *ai.CWMClient, logger *log.Logger) *EnhancedParser {
	return &EnhancedParser{
		baseParser: baseParser,
		cwmClient:  cwmClient,
		logger:     logger,
	}
}

// ParseResult represents the result of parsing with AI enhancement
type ParseResult struct {
	Nodes          []graph.Node
	Edges          []graph.Edge
	Semantics      *ai.SemanticAnalysis
	ImplicitRels   []ai.Relationship
	Documentation  string
}

// EnhanceNodes enhances parsed nodes with AI semantic analysis
func (p *EnhancedParser) EnhanceNodes(ctx context.Context, nodes []graph.Node, codeType string) ([]graph.Node, error) {
	if p.cwmClient == nil {
		return nodes, nil // No AI client, return as-is
	}

	enhanced := make([]graph.Node, len(nodes))
	copy(enhanced, nodes)

	for i := range enhanced {
		// Extract code content from node
		codeContent := extractCodeFromNode(enhanced[i])
		if codeContent == "" {
			continue
		}

		// Analyze with CWM
		analysis, err := p.cwmClient.AnalyzeCodeSemantics(ctx, codeContent, codeType)
		if err != nil {
			p.logger.Printf("Warning: AI analysis failed for node %s: %v", enhanced[i].ID, err)
			continue
		}

		// Add semantic metadata to node properties
		if enhanced[i].Props == nil {
			enhanced[i].Props = make(map[string]interface{})
		}
		enhanced[i].Props["ai_intent"] = analysis.Intent
		enhanced[i].Props["ai_transformations"] = analysis.Transformations
		enhanced[i].Props["ai_dependencies"] = analysis.Dependencies
		if len(analysis.DataFlow) > 0 {
			enhanced[i].Props["ai_data_flow"] = analysis.DataFlow
		}
		if len(analysis.PerformanceNotes) > 0 {
			enhanced[i].Props["ai_performance_notes"] = analysis.PerformanceNotes
		}
		if len(analysis.SecurityNotes) > 0 {
			enhanced[i].Props["ai_security_notes"] = analysis.SecurityNotes
		}
	}

	return enhanced, nil
}

// DiscoverImplicitRelationships uses AI to find relationships between nodes
func (p *EnhancedParser) DiscoverImplicitRelationships(ctx context.Context, nodes []graph.Node) ([]ai.Relationship, error) {
	if p.cwmClient == nil {
		return nil, nil
	}

	// Convert nodes to JSON for AI analysis
	entitiesJSON := convertNodesToJSON(nodes)
	
	relationships, err := p.cwmClient.DiscoverImplicitRelationships(ctx, entitiesJSON)
	if err != nil {
		return nil, fmt.Errorf("discover relationships: %w", err)
	}

	return relationships, nil
}

func extractCodeFromNode(node graph.Node) string {
	if node.Props == nil {
		return ""
	}
	
	if content, ok := node.Props["content"].(string); ok {
		return content
	}
	if sql, ok := node.Props["sql_query"].(string); ok {
		return sql
	}
	if ddl, ok := node.Props["ddl"].(string); ok {
		return ddl
	}
	
	return ""
}

func convertNodesToJSON(nodes []graph.Node) string {
	type Entity struct {
		ID   string                 `json:"id"`
		Type string                 `json:"type"`
		Name string                 `json:"name"`
		Props map[string]interface{} `json:"properties"`
	}
	
	entities := make([]Entity, len(nodes))
	for i, node := range nodes {
		entities[i] = Entity{
			ID:    node.ID,
			Type:  node.Type,
			Name:  node.Label,
			Props: node.Props,
		}
	}
	
	data, _ := json.Marshal(entities)
	return string(data)
}

