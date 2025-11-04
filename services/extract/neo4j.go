package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

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
		now := time.Now().UTC().Format(time.RFC3339Nano)
		
		for _, node := range nodes {
			// Serialize all properties as a single JSON string to avoid nested map issues
			propsJSON := "{}"
			if node.Props != nil && len(node.Props) > 0 {
				if jsonBytes, err := json.Marshal(node.Props); err == nil {
					propsJSON = string(jsonBytes)
				}
			}
			
			// Add updated_at timestamp to node for temporal analysis
			_, err := tx.Run(ctx,
				"MERGE (n:Node {id: $id}) SET n.type = $type, n.label = $label, n.properties_json = $props, n.updated_at = $updated_at",
				map[string]any{
					"id":        node.ID,
					"type":      node.Type,
					"label":     node.Label,
					"props":     propsJSON,
					"updated_at": now,
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
			
			// Add updated_at timestamp to edge for temporal analysis
			_, err := tx.Run(ctx,
				"MATCH (source:Node {id: $source_id}) MATCH (target:Node {id: $target_id}) MERGE (source)-[r:RELATIONSHIP]->(target) SET r.label = $label, r.properties_json = $props, r.updated_at = $updated_at",
				map[string]any{
					"source_id":  edge.SourceID,
					"target_id":  edge.TargetID,
					"label":      edge.Label,
					"props":      propsJSON,
					"updated_at": now,
				})
			if err != nil {
				return nil, fmt.Errorf("failed to save edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
			}
		}

		return nil, nil
	})

	return err
}

// QueryResult represents a single row from a Neo4j query result.
type QueryResult struct {
	Columns []string               `json:"columns"`
	Data    []map[string]any       `json:"data"`
}

// ExecuteQuery executes a Cypher query against Neo4j and returns the results.
func (p *Neo4jPersistence) ExecuteQuery(ctx context.Context, cypherQuery string, params map[string]any) (*QueryResult, error) {
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx, cypherQuery, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	records, err := result.Collect(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to collect query results: %w", err)
	}

	if len(records) == 0 {
		return &QueryResult{Columns: []string{}, Data: []map[string]any{}}, nil
	}

	// Get column names from first record
	keys := records[0].Keys
	columns := make([]string, len(keys))
	for i, key := range keys {
		columns[i] = key
	}

	// Collect data rows
	data := make([]map[string]any, 0, len(records))
	for _, record := range records {
		row := make(map[string]any)
		for _, key := range keys {
			value, ok := record.Get(key)
			if !ok {
				row[key] = nil
				continue
			}
			
			// Handle Neo4j types
			row[key] = convertNeo4jValue(value)
		}
		data = append(data, row)
	}

	return &QueryResult{
		Columns: columns,
		Data:    data,
	}, nil
}

// convertNeo4jValue converts Neo4j-specific types to Go-native types.
func convertNeo4jValue(value any) any {
	switch v := value.(type) {
	case neo4j.Node:
		// Convert Neo4j node to map
		props := make(map[string]any)
		for k, val := range v.Props {
			props[k] = convertNeo4jValue(val)
		}
		return map[string]any{
			"id":         v.ElementId,
			"labels":     v.Labels,
			"properties": props,
		}
	case neo4j.Relationship:
		// Convert Neo4j relationship to map
		props := make(map[string]any)
		for k, val := range v.Props {
			props[k] = convertNeo4jValue(val)
		}
		return map[string]any{
			"id":         v.ElementId,
			"type":       v.Type,
			"start":      v.StartElementId,
			"end":        v.EndElementId,
			"properties": props,
		}
	case neo4j.Path:
		// Convert Neo4j path to map
		nodes := make([]map[string]any, 0, len(v.Nodes))
		for _, node := range v.Nodes {
			props := make(map[string]any)
			for k, val := range node.Props {
				props[k] = convertNeo4jValue(val)
			}
			nodes = append(nodes, map[string]any{
				"id":         node.ElementId,
				"labels":     node.Labels,
				"properties": props,
			})
		}
		relationships := make([]map[string]any, 0, len(v.Relationships))
		for _, rel := range v.Relationships {
			props := make(map[string]any)
			for k, val := range rel.Props {
				props[k] = convertNeo4jValue(val)
			}
			relationships = append(relationships, map[string]any{
				"id":         rel.ElementId,
				"type":       rel.Type,
				"start":      rel.StartElementId,
				"end":        rel.EndElementId,
				"properties": props,
			})
		}
		return map[string]any{
			"nodes":         nodes,
			"relationships": relationships,
		}
	case []any:
		// Recursively convert arrays
		result := make([]any, len(v))
		for i, item := range v {
			result[i] = convertNeo4jValue(item)
		}
		return result
	case map[string]any:
		// Recursively convert maps
		result := make(map[string]any)
		for k, val := range v {
			result[k] = convertNeo4jValue(val)
		}
		return result
	default:
		// Primitive types pass through
		return value
	}
}
