package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Neo4jPersistence is the persistence layer for Neo4j.
type Neo4jPersistence struct {
	driver neo4j.DriverWithContext
}

// NewNeo4jPersistence creates a new Neo4j persistence layer.
func NewNeo4jPersistence(uri, username, password string) (*Neo4jPersistence, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create neo4j driver: %w", err)
	}

	return &Neo4jPersistence{driver: driver}, nil
}

// flattenValue recursively flattens nested structures to JSON strings for Neo4j compatibility.
func flattenValue(v any) any {
	switch val := v.(type) {
	case map[string]any:
		// Serialize nested maps as JSON strings
		if jsonBytes, err := json.Marshal(val); err == nil {
			return string(jsonBytes)
		}
		return "{}"
	case map[any]any:
		// Handle map[any]any (less common but possible)
		if jsonBytes, err := json.Marshal(val); err == nil {
			return string(jsonBytes)
		}
		return "{}"
	case []any:
		// Check if array contains nested structures
		for _, item := range val {
			if _, ok := item.(map[string]any); ok {
				// Contains nested maps, serialize entire array
				if jsonBytes, err := json.Marshal(val); err == nil {
					return string(jsonBytes)
				}
				return "[]"
			}
			if _, ok := item.(map[any]any); ok {
				// Contains nested maps, serialize entire array
				if jsonBytes, err := json.Marshal(val); err == nil {
					return string(jsonBytes)
				}
				return "[]"
			}
		}
		// Primitive array, keep as-is
		return val
	default:
		// Primitive type, keep as-is
		return val
	}
}

// flattenProperties converts nested maps to JSON strings for Neo4j compatibility.
// Neo4j only supports primitive types and arrays, so nested objects must be serialized.
func flattenProperties(props map[string]any) map[string]any {
	if props == nil {
		return nil
	}
	flattened := make(map[string]any)
	for k, v := range props {
		flattened[k] = flattenValue(v)
	}
	return flattened
}

// SaveGraph saves a graph to Neo4j.
func (p *Neo4jPersistence) SaveGraph(nodes []Node, edges []Edge) error {
	ctx := context.Background()
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, node := range nodes {
			// Serialize all properties as a single JSON string to avoid nested map issues
			propsJSON := "{}"
			if node.Props != nil && len(node.Props) > 0 {
				if jsonBytes, err := json.Marshal(node.Props); err == nil {
					propsJSON = string(jsonBytes)
				}
			}
			
			_, err := tx.Run(ctx,
				"MERGE (n:Node {id: $id}) SET n.type = $type, n.label = $label, n.properties_json = $props",
				map[string]any{
					"id":    node.ID,
					"type":  node.Type,
					"label": node.Label,
					"props": propsJSON,
				})
			if err != nil {
				return nil, fmt.Errorf("failed to save node %s: %w", node.ID, err)
			}
		}

		for _, edge := range edges {
			// Serialize all edge properties as a single JSON string
			propsJSON := "{}"
			if edge.Props != nil && len(edge.Props) > 0 {
				if jsonBytes, err := json.Marshal(edge.Props); err == nil {
					propsJSON = string(jsonBytes)
				}
			}
			
			_, err := tx.Run(ctx,
				"MATCH (source:Node {id: $source_id}) MATCH (target:Node {id: $target_id}) MERGE (source)-[r:RELATIONSHIP]->(target) SET r.label = $label, r.properties_json = $props",
				map[string]any{
					"source_id": edge.SourceID,
					"target_id": edge.TargetID,
					"label":     edge.Label,
					"props":     propsJSON,
				})
			if err != nil {
				return nil, fmt.Errorf("failed to save edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
			}
		}

		return nil, nil
	})

	return err
}
