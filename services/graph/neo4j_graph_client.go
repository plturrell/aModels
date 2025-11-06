package graph

import (
	"context"
	"fmt"
	"log"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

// Neo4jGraphClient implements GraphClient interface using Neo4j driver.
type Neo4jGraphClient struct {
	driver neo4j.DriverWithContext
	logger *log.Logger
}

// NewNeo4jGraphClient creates a new Neo4j graph client.
func NewNeo4jGraphClient(driver neo4j.DriverWithContext, logger *log.Logger) *Neo4jGraphClient {
	return &Neo4jGraphClient{
		driver: driver,
		logger: logger,
	}
}

// UpsertNodes upserts nodes to the Neo4j knowledge graph.
func (c *Neo4jGraphClient) UpsertNodes(ctx context.Context, nodes []DomainNode) error {
	if len(nodes) == 0 {
		return nil
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, node := range nodes {
			// Build Cypher MERGE statement
			cypher := fmt.Sprintf(
				"MERGE (n:%s {id: $id}) SET n += $props, n.label = $label, n.type = $type, n.updated_at = datetime()",
				node.Type,
			)

			params := map[string]any{
				"id":     node.ID,
				"label":  node.Label,
				"type":   node.Type,
				"props":  node.Properties,
			}

			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, fmt.Errorf("failed to upsert node %s: %w", node.ID, err)
			}
		}
		return nil, nil
	})

	if err != nil {
		return err
	}

	if c.logger != nil {
		c.logger.Printf("Upserted %d nodes to Neo4j", len(nodes))
	}

	return nil
}

// UpsertEdges upserts edges to the Neo4j knowledge graph.
func (c *Neo4jGraphClient) UpsertEdges(ctx context.Context, edges []DomainEdge) error {
	if len(edges) == 0 {
		return nil
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		for _, edge := range edges {
			// Build Cypher MERGE statement for relationship
			cypher := `
				MATCH (source {id: $source_id})
				MATCH (target {id: $target_id})
				MERGE (source)-[r:` + edge.Type + `]->(target)
				SET r += $props, r.label = $label, r.updated_at = datetime()
			`

			params := map[string]any{
				"source_id": edge.SourceID,
				"target_id": edge.TargetID,
				"label":     edge.Label,
				"props":     edge.Properties,
			}

			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, fmt.Errorf("failed to upsert edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
			}
		}
		return nil, nil
	})

	if err != nil {
		return err
	}

	if c.logger != nil {
		c.logger.Printf("Upserted %d edges to Neo4j", len(edges))
	}

	return nil
}

// Query executes a Cypher query against Neo4j.
func (c *Neo4jGraphClient) Query(ctx context.Context, cypher string, params map[string]interface{}) ([]map[string]interface{}, error) {
	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	result, err := session.Run(ctx, cypher, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query: %w", err)
	}

	var records []map[string]interface{}
	for result.Next(ctx) {
		record := result.Record()
		recordMap := make(map[string]interface{})
		for _, key := range record.Keys {
			val, _ := record.Get(key)
			recordMap[key] = val
		}
		records = append(records, recordMap)
	}

	if err := result.Err(); err != nil {
		return nil, fmt.Errorf("error iterating results: %w", err)
	}

	return records, nil
}

