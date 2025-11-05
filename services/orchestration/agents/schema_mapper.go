package agents

import (
	"context"
	"fmt"
	"log"
	"strings"
)

// DefaultSchemaMapper implements SchemaMapper with default mapping logic.
type DefaultSchemaMapper struct {
	logger *log.Logger
}

// NewDefaultSchemaMapper creates a new default schema mapper.
func NewDefaultSchemaMapper(logger *log.Logger) *DefaultSchemaMapper {
	return &DefaultSchemaMapper{
		logger: logger,
	}
}

// MapSchema maps a source schema to a graph schema.
func (sm *DefaultSchemaMapper) MapSchema(ctx context.Context, sourceSchema *SourceSchema) (*GraphSchema, error) {
	if sm.logger != nil {
		sm.logger.Printf("Mapping schema from %s to knowledge graph", sourceSchema.SourceType)
	}

	graphSchema := &GraphSchema{
		NodeLabels: []string{},
		EdgeTypes:  []string{},
		Properties: make(map[string]PropertyDefinition),
	}

	// Map tables to node labels
	for _, table := range sourceSchema.Tables {
		label := sm.inferLabel(table.Name)
		graphSchema.NodeLabels = append(graphSchema.NodeLabels, label)
	}

	// Map relations to edge types
	for _, relation := range sourceSchema.Relations {
		edgeType := sm.inferEdgeType(relation.Type)
		graphSchema.EdgeTypes = append(graphSchema.EdgeTypes, edgeType)
	}

	return graphSchema, nil
}

// MapData maps source data to graph nodes and edges using mapping rules.
func (sm *DefaultSchemaMapper) MapData(ctx context.Context, sourceData []map[string]interface{}, mapping *MappingRules) ([]GraphNode, []GraphEdge, error) {
	var nodes []GraphNode
	var edges []GraphEdge

	// Find the appropriate node mapping (simplified - would match by table name)
	if len(mapping.NodeMappings) == 0 {
		return nil, nil, fmt.Errorf("no node mappings defined")
	}

	nodeMapping := mapping.NodeMappings[0] // Use first mapping for now

	// Map each row to a node
	for i, row := range sourceData {
		nodeID := fmt.Sprintf("%s-%d", nodeMapping.SourceTable, i)
		properties := make(map[string]interface{})

		// Map columns to properties
		for _, colMapping := range nodeMapping.ColumnMappings {
			if value, ok := row[colMapping.SourceColumn]; ok {
				properties[colMapping.TargetProperty] = value
			} else if colMapping.Default != nil {
				properties[colMapping.TargetProperty] = colMapping.Default
			}
		}

		// Add source metadata
		properties["source_system"] = nodeMapping.SourceTable
		properties["source_id"] = row[nodeMapping.ColumnMappings[0].SourceColumn]

		nodes = append(nodes, GraphNode{
			ID:         nodeID,
			Labels:     []string{nodeMapping.TargetLabel},
			Properties: properties,
		})
	}

	return nodes, edges, nil
}

// inferLabel infers a graph label from a table name.
func (sm *DefaultSchemaMapper) inferLabel(tableName string) string {
	// Convert table name to PascalCase
	parts := strings.Split(tableName, "_")
	var result strings.Builder
	for _, part := range parts {
		if len(part) > 0 {
			result.WriteString(strings.ToUpper(part[0:1]))
			if len(part) > 1 {
				result.WriteString(part[1:])
			}
		}
	}
	return result.String()
}

// inferEdgeType infers an edge type from a relation type.
func (sm *DefaultSchemaMapper) inferEdgeType(relationType string) string {
	return strings.ToUpper(relationType)
}

