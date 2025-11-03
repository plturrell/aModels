package main

import (
	"context"
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

// SaveGraph saves a graph to Neo4j.
func (p *Neo4jPersistence) SaveGraph(nodes []Node, edges []Edge) error {
	ctx := context.Background()
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, node := range nodes {
			_, err := tx.Run(ctx,
				"MERGE (n:Node {id: $id}) SET n.type = $type, n.label = $label, n.properties = $props",
				map[string]any{
					"id":    node.ID,
					"type":  node.Type,
					"label": node.Label,
					"props": node.Props,
				})
			if err != nil {
				return nil, err
			}
		}

		for _, edge := range edges {
			_, err := tx.Run(ctx,
				"MATCH (source:Node {id: $source_id}) MATCH (target:Node {id: $target_id}) MERGE (source)-[r:RELATIONSHIP {label: $label}]->(target) SET r.properties = $props",
				map[string]any{
					"source_id": edge.SourceID,
					"target_id": edge.TargetID,
					"label":     edge.Label,
					"props":     edge.Props,
				})
			if err != nil {
				return nil, err
			}
		}

		return nil, nil
	})

	return err
}
