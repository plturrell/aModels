package neo4j

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/graph/pkg/models"
)

// Neo4jGraphClient implements GraphClient interface using Neo4j driver.
type Neo4jGraphClient struct {
	driver neo4j.DriverWithContext
	logger *log.Logger
}

// Driver returns the Neo4j driver for direct access if needed.
func (c *Neo4jGraphClient) Driver() neo4j.DriverWithContext {
	return c.driver
}

// Neo4jConfig holds configuration for optimized Neo4j connection.
type Neo4jConfig struct {
	URI                  string
	Username             string
	Password             string
	MaxConnectionPoolSize int
	ConnectionTimeout    time.Duration
	MaxTransactionRetryTime time.Duration
	ConnectionAcquisitionTimeout time.Duration
	FetchSize            int
}

// DefaultNeo4jConfig returns optimized default configuration.
func DefaultNeo4jConfig() Neo4jConfig {
	return Neo4jConfig{
		MaxConnectionPoolSize:        100,  // Increased from default 100
		ConnectionTimeout:            30 * time.Second,
		MaxTransactionRetryTime:      30 * time.Second,
		ConnectionAcquisitionTimeout: 60 * time.Second,
		FetchSize:                    1000, // Batch fetch size for large queries
	}
}

// NewOptimizedNeo4jDriver creates a Neo4j driver with optimized connection pooling.
func NewOptimizedNeo4jDriver(config Neo4jConfig) (neo4j.DriverWithContext, error) {
	configFunc := func(c *neo4j.Config) {
		// Connection pool settings
		c.MaxConnectionPoolSize = config.MaxConnectionPoolSize
		c.ConnectionAcquisitionTimeout = config.ConnectionAcquisitionTimeout
		
		// Timeout settings
		c.SocketConnectTimeout = config.ConnectionTimeout
		c.MaxTransactionRetryTime = config.MaxTransactionRetryTime
		
		// Fetch size for large result sets
		c.FetchSize = config.FetchSize
		
		// Enable connection liveness check
		c.MaxConnectionLifetime = 1 * time.Hour
		c.ConnectionLivenessCheckTimeout = 5 * time.Second
	}
	
	driver, err := neo4j.NewDriverWithContext(
		config.URI,
		neo4j.BasicAuth(config.Username, config.Password, ""),
		configFunc,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create Neo4j driver: %w", err)
	}
	
	// Verify connectivity
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if err := driver.VerifyConnectivity(ctx); err != nil {
		driver.Close(ctx)
		return nil, fmt.Errorf("failed to verify Neo4j connectivity: %w", err)
	}
	
	return driver, nil
}

// NewNeo4jGraphClient creates a new Neo4j graph client.
func NewNeo4jGraphClient(driver neo4j.DriverWithContext, logger *log.Logger) *Neo4jGraphClient {
	return &Neo4jGraphClient{
		driver: driver,
		logger: logger,
	}
}

// UpsertNodes upserts nodes to the Neo4j knowledge graph.
// Uses batch operations with UNWIND for improved performance.
func (c *Neo4jGraphClient) UpsertNodes(ctx context.Context, nodes []models.DomainNode) error {
	if len(nodes) == 0 {
		return nil
	}

	// Use batch processing for large node sets
	const batchSize = 500
	if len(nodes) > batchSize {
		return c.upsertNodesBatch(ctx, nodes, batchSize)
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// Group nodes by type for efficient batch processing
	nodesByType := make(map[string][]map[string]any)
	for _, node := range nodes {
		nodeData := map[string]any{
			"id":    node.ID,
			"label": node.Label,
			"type":  node.Type,
			"props": node.Properties,
		}
		nodesByType[node.Type] = append(nodesByType[node.Type], nodeData)
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		// Process each node type in batch using UNWIND
		for nodeType, nodeList := range nodesByType {
			cypher := fmt.Sprintf(`
				UNWIND $nodes AS nodeData
				MERGE (n:%s {id: nodeData.id})
				SET n += nodeData.props,
				    n.label = nodeData.label,
				    n.type = nodeData.type,
				    n.updated_at = datetime()
			`, nodeType)

			params := map[string]any{
				"nodes": nodeList,
			}

			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, fmt.Errorf("failed to batch upsert %d nodes of type %s: %w", len(nodeList), nodeType, err)
			}
		}
		return nil, nil
	})

	if err != nil {
		return err
	}

	if c.logger != nil {
		c.logger.Printf("Batch upserted %d nodes to Neo4j", len(nodes))
	}

	return nil
}

// upsertNodesBatch processes nodes in batches to avoid memory issues.
func (c *Neo4jGraphClient) upsertNodesBatch(ctx context.Context, nodes []models.DomainNode, batchSize int) error {
	for i := 0; i < len(nodes); i += batchSize {
		end := i + batchSize
		if end > len(nodes) {
			end = len(nodes)
		}
		batch := nodes[i:end]
		if err := c.UpsertNodes(ctx, batch); err != nil {
			return fmt.Errorf("failed to upsert batch %d-%d: %w", i, end, err)
		}
	}
	return nil
}

// UpsertEdges upserts edges to the Neo4j knowledge graph.
// Uses batch operations with UNWIND for improved performance.
func (c *Neo4jGraphClient) UpsertEdges(ctx context.Context, edges []models.DomainEdge) error {
	if len(edges) == 0 {
		return nil
	}

	// Use batch processing for large edge sets
	const batchSize = 500
	if len(edges) > batchSize {
		return c.upsertEdgesBatch(ctx, edges, batchSize)
	}

	session := c.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	// Group edges by type for efficient batch processing
	edgesByType := make(map[string][]map[string]any)
	for _, edge := range edges {
		edgeData := map[string]any{
			"source_id": edge.SourceID,
			"target_id": edge.TargetID,
			"label":     edge.Label,
			"props":     edge.Properties,
		}
		edgesByType[edge.Type] = append(edgesByType[edge.Type], edgeData)
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		// Process each edge type in batch using UNWIND
		for edgeType, edgeList := range edgesByType {
			cypher := fmt.Sprintf(`
				UNWIND $edges AS edgeData
				MATCH (source {id: edgeData.source_id})
				MATCH (target {id: edgeData.target_id})
				MERGE (source)-[r:%s]->(target)
				SET r += edgeData.props,
				    r.label = edgeData.label,
				    r.updated_at = datetime()
			`, edgeType)

			params := map[string]any{
				"edges": edgeList,
			}

			if _, err := tx.Run(ctx, cypher, params); err != nil {
				return nil, fmt.Errorf("failed to batch upsert %d edges of type %s: %w", len(edgeList), edgeType, err)
			}
		}
		return nil, nil
	})

	if err != nil {
		return err
	}

	if c.logger != nil {
		c.logger.Printf("Batch upserted %d edges to Neo4j", len(edges))
	}

	return nil
}

// upsertEdgesBatch processes edges in batches to avoid memory issues.
func (c *Neo4jGraphClient) upsertEdgesBatch(ctx context.Context, edges []models.DomainEdge, batchSize int) error {
	for i := 0; i < len(edges); i += batchSize {
		end := i + batchSize
		if end > len(edges) {
			end = len(edges)
		}
		batch := edges[i:end]
		if err := c.UpsertEdges(ctx, batch); err != nil {
			return fmt.Errorf("failed to upsert batch %d-%d: %w", i, end, err)
		}
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

