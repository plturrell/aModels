package storage

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/plturrell/aModels/services/extract/pkg/graph"
)

// Neo4jPersistence is the persistence layer for Neo4j.
type Neo4jPersistence struct {
	driver                  neo4j.DriverWithContext
	enableCatalogSchema    bool
	catalogResourceBaseURI string
}

// NewNeo4jPersistence creates a new Neo4j persistence layer.
func NewNeo4jPersistence(uri, username, password string, enableCatalogSchema bool, catalogResourceBaseURI string) (*Neo4jPersistence, error) {
	driver, err := neo4j.NewDriverWithContext(uri, neo4j.BasicAuth(username, password, ""))
	if err != nil {
		return nil, fmt.Errorf("failed to create neo4j driver: %w", err)
	}

	return &Neo4jPersistence{
		driver:                  driver,
		enableCatalogSchema:    enableCatalogSchema,
		catalogResourceBaseURI: catalogResourceBaseURI,
	}, nil
}

// generateResourceURI generates a Resource URI for a node, matching catalog service format.
func (p *Neo4jPersistence) generateResourceURI(nodeID string) string {
	baseURI := p.catalogResourceBaseURI
	if baseURI == "" {
		baseURI = "http://amodels.org/catalog"
	}
	// Ensure base URI doesn't end with /
	if len(baseURI) > 0 && baseURI[len(baseURI)-1] == '/' {
		baseURI = baseURI[:len(baseURI)-1]
	}
	return fmt.Sprintf("%s/data-element/%s", baseURI, nodeID)
}

// shouldCreateResourceNode determines if a Resource node should be created for this node type.
func shouldCreateResourceNode(nodeType string) bool {
	// Skip Resource creation for system nodes
	systemNodeTypes := map[string]bool{
		"root":              true,
		"project":           true,
		"system":            true,
		"information-system": true,
	}
	return !systemNodeTypes[nodeType]
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
// Improvement 2: Retry logic is applied at the caller level
// Improvement 5: Optimized with batch transaction processing
func (p *Neo4jPersistence) SaveGraph(nodes []graph.Node, edges []graph.Edge) error {
	ctx := context.Background()
	
	// Improvement 5: Use batch processing for large datasets
	batchSize := 1000
	if len(nodes) > 10000 {
		batchSize = 500 // Smaller batches for very large datasets
	}
	
	// Process nodes in batches
	if err := p.saveNodesInBatches(ctx, nodes, batchSize); err != nil {
		return fmt.Errorf("failed to save nodes: %w", err)
	}
	
	// Process edges in batches
	if err := p.saveEdgesInBatches(ctx, edges, batchSize); err != nil {
		return fmt.Errorf("failed to save edges: %w", err)
	}
	
	return nil
}

// saveNodesInBatches saves nodes in batches for better performance
func (p *Neo4jPersistence) saveNodesInBatches(ctx context.Context, nodes []graph.Node, batchSize int) error {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	
	for i := 0; i < len(nodes); i += batchSize {
		end := i + batchSize
		if end > len(nodes) {
			end = len(nodes)
		}
		
		batch := nodes[i:end]
		
		session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
		_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			for _, node := range batch {
				// Serialize all properties as a single JSON string to avoid nested map issues
				propsJSON := "{}"
				if node.Props != nil && len(node.Props) > 0 {
					if jsonBytes, err := json.Marshal(node.Props); err == nil {
						propsJSON = string(jsonBytes)
					}
				}
				
				// Extract agent_id and domain from properties for separate storage (if available)
				var agentID string
				var domainID string
				if node.Props != nil {
					if aid, ok := node.Props["agent_id"].(string); ok {
						agentID = aid
					}
					if did, ok := node.Props["domain"].(string); ok {
						domainID = did
					}
				}
				
				// Add updated_at timestamp to node for temporal analysis
				// Store agent_id and domain as separate properties for easier querying
				query := "MERGE (n:Node {id: $id}) SET n.type = $type, n.label = $label, n.properties_json = $props, n.updated_at = $updated_at"
				params := map[string]any{
					"id":        node.ID,
					"type":      node.Type,
					"label":     node.Label,
					"props":     propsJSON,
					"updated_at": now,
				}
				
				// Add agent_id and domain as separate properties if available
				if agentID != "" {
					query += ", n.agent_id = $agent_id"
					params["agent_id"] = agentID
				}
				if domainID != "" {
					query += ", n.domain = $domain"
					params["domain"] = domainID
				}
				
				_, err := tx.Run(ctx, query, params)
				if err != nil {
					return nil, fmt.Errorf("failed to save node %s: %w", node.ID, err)
				}
				
				// Create Resource node and MAPS_TO relationship if catalog schema integration is enabled
				if p.enableCatalogSchema && shouldCreateResourceNode(node.Type) {
					resourceURI := p.generateResourceURI(node.ID)
					resourceQuery := `
						MERGE (r:Resource {uri: $resource_uri})
						SET r.updated_at = $updated_at
						WITH r
						MATCH (n:Node {id: $node_id})
						MERGE (n)-[:MAPS_TO]->(r)
					`
					resourceParams := map[string]any{
						"resource_uri": resourceURI,
						"node_id":      node.ID,
						"updated_at":   now,
					}
					
					_, err := tx.Run(ctx, resourceQuery, resourceParams)
					if err != nil {
						return nil, fmt.Errorf("failed to create Resource node for %s: %w", node.ID, err)
					}
				}
			}
			return nil, nil
		})
		session.Close(ctx)
		
		if err != nil {
			return fmt.Errorf("node batch %d-%d failed: %w", i, end, err)
		}
	}
	
	return nil
}

// saveEdgesInBatches saves edges in batches for better performance
func (p *Neo4jPersistence) saveEdgesInBatches(ctx context.Context, edges []graph.Edge, batchSize int) error {
	now := time.Now().UTC().Format(time.RFC3339Nano)
	
	for i := 0; i < len(edges); i += batchSize {
		end := i + batchSize
		if end > len(edges) {
			end = len(edges)
		}
		
		batch := edges[i:end]
		
		session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
		_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			for _, edge := range batch {
				// Serialize all edge properties as a single JSON string
				propsJSON := "{}"
				if edge.Props != nil && len(edge.Props) > 0 {
					if jsonBytes, err := json.Marshal(edge.Props); err == nil {
						propsJSON = string(jsonBytes)
					}
				}
				
				// Extract agent_id and domain from edge properties (if available)
				var agentID string
				var domainID string
				if edge.Props != nil {
					if aid, ok := edge.Props["agent_id"].(string); ok {
						agentID = aid
					}
					if did, ok := edge.Props["domain"].(string); ok {
						domainID = did
					}
				}
				
				// Add updated_at timestamp to edge for temporal analysis
				// Store agent_id and domain as separate properties for easier querying
				query := "MATCH (source:Node {id: $source_id}) MATCH (target:Node {id: $target_id}) MERGE (source)-[r:RELATIONSHIP]->(target) SET r.label = $label, r.properties_json = $props, r.updated_at = $updated_at"
				params := map[string]any{
					"source_id":  edge.SourceID,
					"target_id":  edge.TargetID,
					"label":      edge.Label,
					"props":      propsJSON,
					"updated_at": now,
				}
				
				// Add agent_id and domain as separate properties if available
				if agentID != "" {
					query += ", r.agent_id = $agent_id"
					params["agent_id"] = agentID
				}
				if domainID != "" {
					query += ", r.domain = $domain"
					params["domain"] = domainID
				}
				
				_, err := tx.Run(ctx, query, params)
				if err != nil {
					return nil, fmt.Errorf("failed to save edge %s->%s: %w", edge.SourceID, edge.TargetID, err)
				}
			}
			return nil, nil
		})
		session.Close(ctx)
		
		if err != nil {
			return fmt.Errorf("edge batch %d-%d failed: %w", i, end, err)
		}
	}
	
	return nil
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

// CreateExecution creates an Execution node in Neo4j for tracking job/query executions
func (p *Neo4jPersistence) CreateExecution(ctx context.Context, executionID, executionType, entityID, status string, startedAt time.Time, props map[string]any) error {
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	now := time.Now().UTC().Format(time.RFC3339Nano)
	startedAtStr := startedAt.UTC().Format(time.RFC3339Nano)

	query := `
		MERGE (e:Execution {id: $execution_id})
		SET e.execution_type = $execution_type,
		    e.entity_id = $entity_id,
		    e.status = $status,
		    e.started_at = $started_at,
		    e.updated_at = $updated_at
	`
	params := map[string]any{
		"execution_id":  executionID,
		"execution_type": executionType,
		"entity_id":     entityID,
		"status":        status,
		"started_at":    startedAtStr,
		"updated_at":    now,
	}

	// Add additional properties if provided
	if props != nil && len(props) > 0 {
		if propsJSON, err := json.Marshal(props); err == nil {
			params["props_json"] = string(propsJSON)
			query += ", e.properties_json = $props_json"
		}
	}

	// Link to entity node if it exists
	if entityID != "" {
		query += `
		WITH e
		MATCH (n:Node {id: $entity_id})
		MERGE (e)-[:EXECUTES]->(n)
		`
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})

	return err
}

// CreateExecutionMetrics creates ExecutionMetrics node linked to an Execution
func (p *Neo4jPersistence) CreateExecutionMetrics(ctx context.Context, executionID string, metrics map[string]any) error {
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	now := time.Now().UTC().Format(time.RFC3339Nano)
	metricsID := fmt.Sprintf("%s:metrics", executionID)

	metricsJSON := "{}"
	if metrics != nil && len(metrics) > 0 {
		if jsonBytes, err := json.Marshal(metrics); err == nil {
			metricsJSON = string(jsonBytes)
		}
	}

	query := `
		MATCH (e:Execution {id: $execution_id})
		MERGE (m:ExecutionMetrics {id: $metrics_id})
		SET m.properties_json = $metrics_json,
		    m.updated_at = $updated_at
		MERGE (e)-[:HAS_METRICS]->(m)
	`
	params := map[string]any{
		"execution_id": executionID,
		"metrics_id":   metricsID,
		"metrics_json": metricsJSON,
		"updated_at":   now,
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})

	return err
}

// CreateQualityIssue creates a QualityIssue node linked to an entity
func (p *Neo4jPersistence) CreateQualityIssue(ctx context.Context, issueID, entityID, issueType, severity, description string, props map[string]any) error {
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	now := time.Now().UTC().Format(time.RFC3339Nano)

	query := `
		MERGE (q:QualityIssue {id: $issue_id})
		SET q.issue_type = $issue_type,
		    q.severity = $severity,
		    q.description = $description,
		    q.entity_id = $entity_id,
		    q.created_at = $created_at,
		    q.updated_at = $updated_at
	`
	params := map[string]any{
		"issue_id":    issueID,
		"issue_type":  issueType,
		"severity":    severity,
		"description": description,
		"entity_id":   entityID,
		"created_at":  now,
		"updated_at":  now,
	}

	// Add additional properties if provided
	if props != nil && len(props) > 0 {
		if propsJSON, err := json.Marshal(props); err == nil {
			params["props_json"] = string(propsJSON)
			query += ", q.properties_json = $props_json"
		}
	}

	// Link to entity node if it exists
	if entityID != "" {
		query += `
		WITH q
		MATCH (n:Node {id: $entity_id})
		MERGE (q)-[:AFFECTS]->(n)
		`
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})

	return err
}

// CreatePerformanceMetric creates a PerformanceMetric node linked to an entity
func (p *Neo4jPersistence) CreatePerformanceMetric(ctx context.Context, metricID, entityID, metricType string, value float64, props map[string]any) error {
	session := p.driver.NewSession(ctx, neo4j.SessionConfig{})
	defer session.Close(ctx)

	now := time.Now().UTC().Format(time.RFC3339Nano)

	query := `
		MERGE (p:PerformanceMetric {id: $metric_id})
		SET p.metric_type = $metric_type,
		    p.value = $value,
		    p.entity_id = $entity_id,
		    p.timestamp = $timestamp,
		    p.updated_at = $updated_at
	`
	params := map[string]any{
		"metric_id":   metricID,
		"metric_type": metricType,
		"value":       value,
		"entity_id":   entityID,
		"timestamp":   now,
		"updated_at":  now,
	}

	// Add additional properties if provided
	if props != nil && len(props) > 0 {
		if propsJSON, err := json.Marshal(props); err == nil {
			params["props_json"] = string(propsJSON)
			query += ", p.properties_json = $props_json"
		}
	}

	// Link to entity node if it exists
	if entityID != "" {
		query += `
		WITH p
		MATCH (n:Node {id: $entity_id})
		MERGE (p)-[:MEASURES]->(n)
		`
	}

	_, err := session.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
		return tx.Run(ctx, query, params)
	})

	return err
}
