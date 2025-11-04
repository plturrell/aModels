package hana

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/hanapool"
	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_HANA/pkg/storage"
)

// SQLQueryTool executes SQL queries against HANA
type SQLQueryTool struct {
	pool            *hanapool.Pool
	relationalStore *storage.RelationalStore
}

// NewSQLQueryTool creates a new SQL query tool
func NewSQLQueryTool(pool *hanapool.Pool) *SQLQueryTool {
	return &SQLQueryTool{
		pool:            pool,
		relationalStore: storage.NewRelationalStore(pool),
	}
}

// Name returns the tool name
func (t *SQLQueryTool) Name() string {
	return "sql_query"
}

// Description returns the tool description
func (t *SQLQueryTool) Description() string {
	return "Execute SQL queries against the HANA database. Use this to query relational data, run analytics, and perform data operations."
}

// Call executes the SQL query
// Expected input format: JSON string with "query" and optional "limit" fields
func (t *SQLQueryTool) Call(ctx context.Context, input string) (string, error) {
	// Parse input JSON
	var params map[string]interface{}
	if err := json.Unmarshal([]byte(input), &params); err != nil {
		return "", fmt.Errorf("failed to parse input: %w", err)
	}
	
	query, ok := params["query"].(string)
	if !ok {
		return "", fmt.Errorf("query parameter is required and must be a string")
	}

	limit := 100
	if limitVal, ok := params["limit"].(float64); ok {
		limit = int(limitVal)
	}

	// Validate query for safety
	if err := t.validateQuery(query); err != nil {
		return "", fmt.Errorf("query validation failed: %w", err)
	}

	// Add LIMIT if not present and it's a SELECT query
	if strings.ToUpper(strings.TrimSpace(query))[:6] == "SELECT" && !strings.Contains(strings.ToUpper(query), "LIMIT") {
		query += fmt.Sprintf(" LIMIT %d", limit)
	}

	// Execute query
	rows, err := t.pool.Query(ctx, query)
	if err != nil {
		return "", fmt.Errorf("failed to execute query: %w", err)
	}
	defer rows.Close()

	// Get column names
	columns, err := rows.Columns()
	if err != nil {
		return "", fmt.Errorf("failed to get columns: %w", err)
	}

	// Scan results
	var results []map[string]interface{}
	for rows.Next() {
		values := make([]interface{}, len(columns))
		valuePtrs := make([]interface{}, len(columns))
		for i := range values {
			valuePtrs[i] = &values[i]
		}

		if err := rows.Scan(valuePtrs...); err != nil {
			return "", fmt.Errorf("failed to scan row: %w", err)
		}

		row := make(map[string]interface{})
		for i, col := range columns {
			row[col] = values[i]
		}
		results = append(results, row)
	}

	result := map[string]interface{}{
		"success": true,
		"rows":    results,
		"count":   len(results),
	}
	jsonResult, err := json.Marshal(result)
	if err != nil {
		return "", fmt.Errorf("failed to marshal result: %w", err)
	}
	return string(jsonResult), nil
}

// validateQuery validates the SQL query for safety
func (t *SQLQueryTool) validateQuery(query string) error {
	query = strings.ToUpper(strings.TrimSpace(query))

	// Check for dangerous operations
	dangerousOps := []string{"DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE"}
	for _, op := range dangerousOps {
		if strings.Contains(query, op) {
			return fmt.Errorf("operation %s is not allowed for safety", op)
		}
	}

	// Only allow SELECT queries
	if !strings.HasPrefix(query, "SELECT") {
		return fmt.Errorf("only SELECT queries are allowed")
	}

	return nil
}

// VectorSearchTool performs vector similarity search
type VectorSearchTool struct {
	pool        *hanapool.Pool
	vectorStore *storage.VectorStore
}

// NewVectorSearchTool creates a new vector search tool
func NewVectorSearchTool(pool *hanapool.Pool) *VectorSearchTool {
	return &VectorSearchTool{
		pool:        pool,
		vectorStore: storage.NewVectorStore(pool),
	}
}

// Name returns the tool name
func (t *VectorSearchTool) Name() string {
	return "vector_search"
}

// Description returns the tool description
func (t *VectorSearchTool) Description() string {
	return "Perform vector similarity search to find semantically similar content. Requires an embedding vector as input."
}

// Call performs vector similarity search
func (t *VectorSearchTool) Call(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	vectorInterface, ok := input["vector"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("vector parameter is required and must be an array of numbers")
	}

	// Convert interface{} slice to []float64
	vector := make([]float64, len(vectorInterface))
	for i, v := range vectorInterface {
		if num, ok := v.(float64); ok {
			vector[i] = num
		} else {
			return nil, fmt.Errorf("vector element %d is not a number", i)
		}
	}

	limit := 10
	if limitVal, ok := input["limit"].(float64); ok {
		limit = int(limitVal)
	}

	threshold := 0.0
	if thresholdVal, ok := input["threshold"].(float64); ok {
		threshold = thresholdVal
	}

	// Perform similarity search
	results, err := t.vectorStore.SimilaritySearch(ctx, vector, limit, threshold)
	if err != nil {
		return nil, fmt.Errorf("failed to perform similarity search: %w", err)
	}

	// Convert results to interface{}
	searchResults := make([]map[string]interface{}, len(results))
	for i, result := range results {
		searchResults[i] = map[string]interface{}{
			"id":       result.ID,
			"content":  result.Content,
			"score":    result.Score,
			"metadata": result.Metadata,
		}
	}

	return map[string]interface{}{
		"success": true,
		"results": searchResults,
		"count":   len(searchResults),
	}, nil
}

// GraphTraversalTool performs graph traversal operations
type GraphTraversalTool struct {
	pool       *hanapool.Pool
	graphStore *storage.GraphStore
}

// NewGraphTraversalTool creates a new graph traversal tool
func NewGraphTraversalTool(pool *hanapool.Pool) *GraphTraversalTool {
	return &GraphTraversalTool{
		pool:       pool,
		graphStore: storage.NewGraphStore(pool),
	}
}

// Name returns the tool name
func (t *GraphTraversalTool) Name() string {
	return "graph_traversal"
}

// Description returns the tool description
func (t *GraphTraversalTool) Description() string {
	return "Perform graph traversal operations to find relationships, paths, and connected nodes."
}

// Call performs graph traversal operations
func (t *GraphTraversalTool) Call(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	operation, ok := input["operation"].(string)
	if !ok {
		return nil, fmt.Errorf("operation parameter is required and must be a string")
	}

	limit := 50
	if limitVal, ok := input["limit"].(float64); ok {
		limit = int(limitVal)
	}

	switch operation {
	case "neighbors":
		return t.getNeighbors(ctx, input, limit)
	case "path":
		return t.findPath(ctx, input)
	case "nodes_by_type":
		return t.getNodesByType(ctx, input, limit)
	case "outgoing_edges":
		return t.getOutgoingEdges(ctx, input, limit)
	case "incoming_edges":
		return t.getIncomingEdges(ctx, input, limit)
	default:
		return nil, fmt.Errorf("unknown operation: %s", operation)
	}
}

func (t *GraphTraversalTool) getNeighbors(ctx context.Context, input map[string]interface{}, limit int) (map[string]interface{}, error) {
	nodeID, ok := input["node_id"].(float64)
	if !ok {
		return nil, fmt.Errorf("node_id is required for neighbors operation")
	}

	edgeType, _ := input["edge_type"].(string)

	neighbors, err := t.graphStore.GetNeighbors(ctx, int64(nodeID), edgeType)
	if err != nil {
		return nil, fmt.Errorf("failed to get neighbors: %w", err)
	}

	// Convert to interface{}
	results := make([]map[string]interface{}, len(neighbors))
	for i, neighbor := range neighbors {
		results[i] = map[string]interface{}{
			"id":         neighbor.ID,
			"type":       neighbor.Type,
			"data":       neighbor.Data,
			"created_at": neighbor.CreatedAt,
		}
	}

	return map[string]interface{}{
		"success":   true,
		"neighbors": results,
		"count":     len(results),
	}, nil
}

func (t *GraphTraversalTool) findPath(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	fromNode, ok := input["from_node"].(float64)
	if !ok {
		return nil, fmt.Errorf("from_node is required for path operation")
	}

	toNode, ok := input["to_node"].(float64)
	if !ok {
		return nil, fmt.Errorf("to_node is required for path operation")
	}

	path, err := t.graphStore.FindPath(ctx, int64(fromNode), int64(toNode), 10)
	if err != nil {
		return nil, fmt.Errorf("failed to find path: %w", err)
	}

	// Convert to interface{}
	nodes := make([]map[string]interface{}, len(path.Nodes))
	for i, node := range path.Nodes {
		nodes[i] = map[string]interface{}{
			"id":         node.ID,
			"type":       node.Type,
			"data":       node.Data,
			"created_at": node.CreatedAt,
		}
	}

	edges := make([]map[string]interface{}, len(path.Edges))
	for i, edge := range path.Edges {
		edges[i] = map[string]interface{}{
			"id":         edge.ID,
			"from_id":    edge.FromID,
			"to_id":      edge.ToID,
			"type":       edge.Type,
			"weight":     edge.Weight,
			"data":       edge.Data,
			"created_at": edge.CreatedAt,
		}
	}

	return map[string]interface{}{
		"success": true,
		"path": map[string]interface{}{
			"nodes": nodes,
			"edges": edges,
			"cost":  path.Cost,
		},
	}, nil
}

func (t *GraphTraversalTool) getNodesByType(ctx context.Context, input map[string]interface{}, limit int) (map[string]interface{}, error) {
	nodeType, ok := input["node_type"].(string)
	if !ok {
		return nil, fmt.Errorf("node_type is required for nodes_by_type operation")
	}

	nodes, err := t.graphStore.GetNodesByType(ctx, nodeType, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to get nodes by type: %w", err)
	}

	// Convert to interface{}
	results := make([]map[string]interface{}, len(nodes))
	for i, node := range nodes {
		results[i] = map[string]interface{}{
			"id":         node.ID,
			"type":       node.Type,
			"data":       node.Data,
			"created_at": node.CreatedAt,
		}
	}

	return map[string]interface{}{
		"success": true,
		"nodes":   results,
		"count":   len(results),
	}, nil
}

func (t *GraphTraversalTool) getOutgoingEdges(ctx context.Context, input map[string]interface{}, limit int) (map[string]interface{}, error) {
	nodeID, ok := input["node_id"].(float64)
	if !ok {
		return nil, fmt.Errorf("node_id is required for outgoing_edges operation")
	}

	edgeType, _ := input["edge_type"].(string)

	edges, err := t.graphStore.GetOutgoingEdges(ctx, int64(nodeID), edgeType)
	if err != nil {
		return nil, fmt.Errorf("failed to get outgoing edges: %w", err)
	}

	// Convert to interface{}
	results := make([]map[string]interface{}, len(edges))
	for i, edge := range edges {
		results[i] = map[string]interface{}{
			"id":         edge.ID,
			"from_id":    edge.FromID,
			"to_id":      edge.ToID,
			"type":       edge.Type,
			"weight":     edge.Weight,
			"data":       edge.Data,
			"created_at": edge.CreatedAt,
		}
	}

	return map[string]interface{}{
		"success": true,
		"edges":   results,
		"count":   len(results),
	}, nil
}

func (t *GraphTraversalTool) getIncomingEdges(ctx context.Context, input map[string]interface{}, limit int) (map[string]interface{}, error) {
	nodeID, ok := input["node_id"].(float64)
	if !ok {
		return nil, fmt.Errorf("node_id is required for incoming_edges operation")
	}

	edgeType, _ := input["edge_type"].(string)

	edges, err := t.graphStore.GetIncomingEdges(ctx, int64(nodeID), edgeType)
	if err != nil {
		return nil, fmt.Errorf("failed to get incoming edges: %w", err)
	}

	// Convert to interface{}
	results := make([]map[string]interface{}, len(edges))
	for i, edge := range edges {
		results[i] = map[string]interface{}{
			"id":         edge.ID,
			"from_id":    edge.FromID,
			"to_id":      edge.ToID,
			"type":       edge.Type,
			"weight":     edge.Weight,
			"data":       edge.Data,
			"created_at": edge.CreatedAt,
		}
	}

	return map[string]interface{}{
		"success": true,
		"edges":   results,
		"count":   len(results),
	}, nil
}

// DataInsertTool inserts data into HANA tables
type DataInsertTool struct {
	pool            *hanapool.Pool
	relationalStore *storage.RelationalStore
}

// NewDataInsertTool creates a new data insert tool
func NewDataInsertTool(pool *hanapool.Pool) *DataInsertTool {
	return &DataInsertTool{
		pool:            pool,
		relationalStore: storage.NewRelationalStore(pool),
	}
}

// Name returns the tool name
func (t *DataInsertTool) Name() string {
	return "data_insert"
}

// Description returns the tool description
func (t *DataInsertTool) Description() string {
	return "Insert data into HANA tables. Use this to store new records, update existing ones, or perform data operations."
}

// Call performs data insertion/update/deletion
func (t *DataInsertTool) Call(ctx context.Context, input map[string]interface{}) (map[string]interface{}, error) {
	table, ok := input["table"].(string)
	if !ok {
		return nil, fmt.Errorf("table parameter is required and must be a string")
	}

	dataInterface, ok := input["data"].(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("data parameter is required and must be an object")
	}

	// Convert interface{} map to map[string]interface{}
	data := make(map[string]interface{})
	for k, v := range dataInterface {
		data[k] = v
	}

	operation := "insert"
	if op, ok := input["operation"].(string); ok {
		operation = op
	}

	var err error

	switch operation {
	case "insert":
		_, err = t.relationalStore.Insert(ctx, table, data)
	case "update":
		whereInterface, _ := input["where"].(map[string]interface{})
		where := make(map[string]interface{})
		for k, v := range whereInterface {
			where[k] = v
		}
		_, err = t.relationalStore.Update(ctx, table, data, where)
	case "delete":
		whereInterface, _ := input["where"].(map[string]interface{})
		where := make(map[string]interface{})
		for k, v := range whereInterface {
			where[k] = v
		}
		_, err = t.relationalStore.Delete(ctx, table, where)
	default:
		return nil, fmt.Errorf("unknown operation: %s", operation)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to perform %s operation: %w", operation, err)
	}

	return map[string]interface{}{
		"success":   true,
		"operation": operation,
		"table":     table,
	}, nil
}
