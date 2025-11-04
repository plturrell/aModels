package workflows

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

// PipelineToAgentFlowOptions configures the pipeline-to-AgentFlow conversion.
type PipelineToAgentFlowOptions struct {
	ExtractServiceURL string // URL to extract service for Neo4j queries
	AgentFlowServiceURL string // URL to AgentFlow service for flow creation
}

// ControlMToAgentFlowConverter converts Control-M → SQL → Tables pipelines into LangFlow flows.
type ControlMToAgentFlowConverter struct {
	extractServiceURL string
	httpClient        *http.Client
}

// NewControlMToAgentFlowConverter creates a new converter.
func NewControlMToAgentFlowConverter(extractServiceURL string) *ControlMToAgentFlowConverter {
	return &ControlMToAgentFlowConverter{
		extractServiceURL: extractServiceURL,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

// PipelineSegment represents a segment of the pipeline (Control-M → SQL → Table).
type PipelineSegment struct {
	ControlMJob  *ControlMJobInfo  `json:"controlm_job"`
	SQLQueries   []SQLQueryInfo    `json:"sql_queries"`
	TargetTables []TableInfo       `json:"target_tables"`
	SourceTables []TableInfo       `json:"source_tables"`
	DataFlowPath []DataFlowStep    `json:"data_flow_path"`
}

// ControlMJobInfo represents a Control-M job from the knowledge graph.
type ControlMJobInfo struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Command     string         `json:"command"`
	Application string         `json:"application"`
	Properties  map[string]any `json:"properties"`
}

// SQLQueryInfo represents a SQL query from the knowledge graph.
type SQLQueryInfo struct {
	ID          string         `json:"id"`
	Query       string         `json:"query"`
	Type        string         `json:"type"` // SELECT, INSERT, UPDATE, CREATE, etc.
	Properties  map[string]any `json:"properties"`
}

// TableInfo represents a table from the knowledge graph.
type TableInfo struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	Type        string         `json:"type"` // table, view
	Columns     []ColumnInfo   `json:"columns"`
	Properties  map[string]any `json:"properties"`
}

// ColumnInfo represents a column.
type ColumnInfo struct {
	ID          string         `json:"id"`
	Name        string         `json:"name"`
	DataType    string         `json:"data_type"`
	Properties  map[string]any `json:"properties"`
}

// DataFlowStep represents a step in the data flow.
type DataFlowStep struct {
	Source      string `json:"source"`
	Target      string `json:"target"`
	Relationship string `json:"relationship"`
}

// QueryResult represents the result of a Neo4j query
type QueryResult struct {
	Columns []string         `json:"columns"`
	Data    []map[string]any `json:"data"`
}

// executeQuery executes a Cypher query via the extract service
func (c *ControlMToAgentFlowConverter) executeQuery(ctx context.Context, query string, params map[string]any) (*QueryResult, error) {
	endpoint := strings.TrimRight(c.extractServiceURL, "/") + "/knowledge-graph/query"
	requestBody := map[string]any{
		"query":  query,
		"params": params,
	}

	body, err := json.Marshal(requestBody)
	if err != nil {
		return nil, fmt.Errorf("marshal query request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build query request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("execute query: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("query failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var queryResult struct {
		Columns []string         `json:"columns"`
		Data    []map[string]any `json:"data"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&queryResult); err != nil {
		return nil, fmt.Errorf("decode query response: %w", err)
	}

	return &QueryResult{
		Columns: queryResult.Columns,
		Data:    queryResult.Data,
	}, nil
}

// QueryPipelineFromGraph queries Neo4j to extract Control-M → SQL → Tables pipeline.
func (c *ControlMToAgentFlowConverter) QueryPipelineFromGraph(ctx context.Context, projectID, systemID string) ([]PipelineSegment, error) {
	// Query to find Control-M jobs
	query := `
		MATCH (job:Node)
		WHERE job.type = 'control-m-job'
		RETURN job
		LIMIT 50
	`

	params := map[string]any{}
	if projectID != "" {
		params["project_id"] = projectID
	}
	if systemID != "" {
		params["system_id"] = systemID
	}

	queryResult, err := c.executeQuery(ctx, query, params)
	if err != nil {
		return nil, fmt.Errorf("query Control-M jobs: %w", err)
	}

	// Group results by Control-M job
	segments := make(map[string]*PipelineSegment)

	// First pass: collect all Control-M jobs
	for _, row := range queryResult.Data {
		if jobNode, ok := extractNode(row, "job"); ok {
			jobID := getString(jobNode, "id")
			if jobID == "" {
				continue
			}

			if _, exists := segments[jobID]; !exists {
				// Parse properties_json if available
				props := getMap(jobNode, "properties")
				if propsJSON, ok := props["properties_json"].(string); ok && propsJSON != "" {
					var parsedProps map[string]any
					if err := json.Unmarshal([]byte(propsJSON), &parsedProps); err == nil {
						for k, v := range parsedProps {
							props[k] = v
						}
					}
				}

				segments[jobID] = &PipelineSegment{
					ControlMJob: &ControlMJobInfo{
						ID:          jobID,
						Name:        getString(jobNode, "label"),
						Description: getString(props, "description"),
						Command:     getString(props, "command"),
						Application: getString(props, "application"),
						Properties:  props,
					},
					SQLQueries:   []SQLQueryInfo{},
					TargetTables: []TableInfo{},
					SourceTables: []TableInfo{},
					DataFlowPath: []DataFlowStep{},
				}
			}
		}
	}

	// Second pass: for each job, query related SQL and tables
	for jobID, segment := range segments {
		// Query SQL queries related to this job
		sqlQuery := `
			MATCH (job:Node {id: $job_id})
			OPTIONAL MATCH (job)-[r1:RELATIONSHIP]->(sql:Node)
			WHERE sql.type = 'sql-query' OR sql.type = 'sql'
			OPTIONAL MATCH (sql)-[r2:RELATIONSHIP]->(table:Node {type: 'table'})
			OPTIONAL MATCH (table)-[r3:RELATIONSHIP {label: 'HAS_COLUMN'}]->(col:Node {type: 'column'})
			OPTIONAL MATCH (col)-[r4:RELATIONSHIP {label: 'DATA_FLOW'}]->(targetCol:Node {type: 'column'})
			OPTIONAL MATCH (targetCol)<-[r5:RELATIONSHIP {label: 'HAS_COLUMN'}]-(targetTable:Node {type: 'table'})
			RETURN sql, table, col, targetCol, targetTable
			LIMIT 50
		`

		sqlParams := map[string]any{"job_id": jobID}
		sqlResult, err := c.executeQuery(ctx, sqlQuery, sqlParams)
		if err != nil {
			log.Printf("Failed to query SQL for job %s: %v", jobID, err)
			continue
		}

		// Process SQL results
		for _, sqlRow := range sqlResult.Data {
			// Extract SQL query
			if sqlNode, ok := extractNode(sqlRow, "sql"); ok && sqlNode != nil {
				sqlID := getString(sqlNode, "id")
				if sqlID != "" && !containsSQL(segment.SQLQueries, sqlID) {
					props := getMap(sqlNode, "properties")
					if propsJSON, ok := props["properties_json"].(string); ok && propsJSON != "" {
						var parsedProps map[string]any
						if err := json.Unmarshal([]byte(propsJSON), &parsedProps); err == nil {
							for k, v := range parsedProps {
								props[k] = v
							}
						}
					}

					segment.SQLQueries = append(segment.SQLQueries, SQLQueryInfo{
						ID:         sqlID,
						Query:      getString(props, "query") + getString(sqlNode, "label"),
						Type:       getString(props, "type"),
						Properties: props,
					})
				}
			}

			// Extract source table
			if tableNode, ok := extractNode(sqlRow, "table"); ok && tableNode != nil {
				tableID := getString(tableNode, "id")
				if tableID != "" && !containsTable(segment.SourceTables, tableID) {
					props := getMap(tableNode, "properties")
					if propsJSON, ok := props["properties_json"].(string); ok && propsJSON != "" {
						var parsedProps map[string]any
						if err := json.Unmarshal([]byte(propsJSON), &parsedProps); err == nil {
							for k, v := range parsedProps {
								props[k] = v
							}
						}
					}

					table := TableInfo{
						ID:         tableID,
						Name:       getString(tableNode, "label"),
						Type:       getString(tableNode, "type"),
						Columns:    []ColumnInfo{},
						Properties: props,
					}

					// Extract columns
					if colNode, ok := extractNode(sqlRow, "col"); ok && colNode != nil {
						colID := getString(colNode, "id")
						if colID != "" {
							colProps := getMap(colNode, "properties")
							if colPropsJSON, ok := colProps["properties_json"].(string); ok && colPropsJSON != "" {
								var parsedColProps map[string]any
								if err := json.Unmarshal([]byte(colPropsJSON), &parsedColProps); err == nil {
									for k, v := range parsedColProps {
										colProps[k] = v
									}
								}
							}

							table.Columns = append(table.Columns, ColumnInfo{
								ID:         colID,
								Name:       getString(colNode, "label"),
								DataType:   getString(colProps, "data_type"),
								Properties: colProps,
							})
						}
					}

					segment.SourceTables = append(segment.SourceTables, table)
				}
			}

			// Extract target table (from DATA_FLOW)
			if targetTableNode, ok := extractNode(sqlRow, "targetTable"); ok && targetTableNode != nil {
				tableID := getString(targetTableNode, "id")
				if tableID != "" && !containsTable(segment.TargetTables, tableID) {
					props := getMap(targetTableNode, "properties")
					if propsJSON, ok := props["properties_json"].(string); ok && propsJSON != "" {
						var parsedProps map[string]any
						if err := json.Unmarshal([]byte(propsJSON), &parsedProps); err == nil {
							for k, v := range parsedProps {
								props[k] = v
							}
						}
					}

					segment.TargetTables = append(segment.TargetTables, TableInfo{
						ID:         tableID,
						Name:       getString(targetTableNode, "label"),
						Type:       getString(targetTableNode, "type"),
						Properties: props,
					})
				}
			}

			// Extract data flow step
			if sourceCol, ok := extractNode(sqlRow, "col"); ok && sourceCol != nil {
				if targetCol, ok := extractNode(sqlRow, "targetCol"); ok && targetCol != nil {
					step := DataFlowStep{
						Source:       getString(sourceCol, "label"),
						Target:       getString(targetCol, "label"),
						Relationship: "DATA_FLOW",
					}
					if !containsDataFlowStep(segment.DataFlowPath, step) {
						segment.DataFlowPath = append(segment.DataFlowPath, step)
					}
				}
			}
		}
	}

	// Convert map to slice
	result := make([]PipelineSegment, 0, len(segments))
	for _, segment := range segments {
		result = append(result, *segment)
	}

	return result, nil
}

// GenerateLangFlowFlow generates a LangFlow flow JSON from pipeline segments.
// LangFlow expects nodes with a specific structure including template, inputs, etc.
func (c *ControlMToAgentFlowConverter) GenerateLangFlowFlow(segments []PipelineSegment, flowName string) (map[string]any, error) {
	nodes := []map[string]any{}
	edges := []map[string]any{}

	// Create agent nodes for each segment
	for i, segment := range segments {
		// 1. Control-M Agent Node
		controlMNodeID := fmt.Sprintf("controlm_agent_%d", i)
		controlMNode := map[string]any{
			"id":       controlMNodeID,
			"type":     "ControlMAgent",
			"template": "ControlM Agent",
			"data": map[string]any{
				"type":        "ControlMAgent",
				"node":        map[string]any{"template": "ControlM Agent"},
				"name":        segment.ControlMJob.Name,
				"display_name": segment.ControlMJob.Name,
				"description": segment.ControlMJob.Description,
				"command":     segment.ControlMJob.Command,
				"application": segment.ControlMJob.Application,
				"properties":  segment.ControlMJob.Properties,
			},
			"position": map[string]any{
				"x": float64(i * 300),
				"y": 0.0,
			},
		}
		nodes = append(nodes, controlMNode)

		// 2. SQL Processing Agent Node(s)
		for j, sqlQuery := range segment.SQLQueries {
			sqlNodeID := fmt.Sprintf("sql_agent_%d_%d", i, j)
			sqlNode := map[string]any{
				"id":       sqlNodeID,
				"type":     "SQLAgent",
				"template": "SQL Agent",
				"data": map[string]any{
					"type":        "SQLAgent",
					"node":        map[string]any{"template": "SQL Agent"},
					"name":        fmt.Sprintf("SQL Agent %d", j+1),
					"display_name": fmt.Sprintf("SQL Agent %d", j+1),
					"query":       sqlQuery.Query,
					"query_type": sqlQuery.Type,
					"properties": sqlQuery.Properties,
				},
				"position": map[string]any{
					"x": float64(i*300 + 150),
					"y": float64((j + 1) * 100),
				},
			}
			nodes = append(nodes, sqlNode)

			// Edge from Control-M to SQL
			edges = append(edges, map[string]any{
				"id":     fmt.Sprintf("edge_%s_%s", controlMNodeID, sqlNodeID),
				"source": controlMNodeID,
				"target": sqlNodeID,
				"sourceHandle": "output",
				"targetHandle": "input",
				"type":   "default",
			})
		}

		// 3. Table Processing Agent Node
		tableNodeID := fmt.Sprintf("table_agent_%d", i)
		tableNode := map[string]any{
			"id":       tableNodeID,
			"type":     "TableAgent",
			"template": "Table Agent",
			"data": map[string]any{
				"type":          "TableAgent",
				"node":          map[string]any{"template": "Table Agent"},
				"name":          fmt.Sprintf("Table Agent %d", i+1),
				"display_name": fmt.Sprintf("Table Agent %d", i+1),
				"source_tables": segment.SourceTables,
				"target_tables": segment.TargetTables,
				"data_flow":     segment.DataFlowPath,
			},
			"position": map[string]any{
				"x": float64(i*300 + 300),
				"y": 0.0,
			},
		}
		nodes = append(nodes, tableNode)

		// Edge from SQL to Table
		if len(segment.SQLQueries) > 0 {
			sqlNodeID := fmt.Sprintf("sql_agent_%d_0", i)
			edges = append(edges, map[string]any{
				"id":          fmt.Sprintf("edge_%s_%s", sqlNodeID, tableNodeID),
				"source":      sqlNodeID,
				"target":      tableNodeID,
				"sourceHandle": "output",
				"targetHandle": "input",
				"type":        "default",
			})
		}

		// 4. Data Quality Agent Node
		qualityNodeID := fmt.Sprintf("quality_agent_%d", i)
		qualityNode := map[string]any{
			"id":       qualityNodeID,
			"type":     "QualityAgent",
			"template": "Quality Agent",
			"data": map[string]any{
				"type":          "QualityAgent",
				"node":          map[string]any{"template": "Quality Agent"},
				"name":          fmt.Sprintf("Quality Agent %d", i+1),
				"display_name": fmt.Sprintf("Quality Agent %d", i+1),
				"monitors":     segment.DataFlowPath,
				"source_tables": segment.SourceTables,
				"target_tables": segment.TargetTables,
			},
			"position": map[string]any{
				"x": float64(i*300 + 450),
				"y": 0.0,
			},
		}
		nodes = append(nodes, qualityNode)

		// Edge from Table to Quality
		edges = append(edges, map[string]any{
			"id":          fmt.Sprintf("edge_%s_%s", tableNodeID, qualityNodeID),
			"source":      tableNodeID,
			"target":      qualityNodeID,
			"sourceHandle": "output",
			"targetHandle": "input",
			"type":        "default",
		})
	}

	// Create flow structure compatible with LangFlow
	flow := map[string]any{
		"name":        flowName,
		"description": fmt.Sprintf("Generated from Control-M → SQL → Tables pipeline with %d segments", len(segments)),
		"data": map[string]any{
			"nodes": nodes,
			"edges": edges,
			"viewport": map[string]any{
				"x":    0.0,
				"y":    0.0,
				"zoom": 1.0,
			},
		},
		"is_component": false,
	}

	return flow, nil
}

// CreateFlowInAgentFlow creates the generated flow in AgentFlow/LangFlow service.
func (c *ControlMToAgentFlowConverter) CreateFlowInAgentFlow(ctx context.Context, agentFlowServiceURL string, flowJSON map[string]any, flowID string, projectID string, force bool) (map[string]any, error) {
	if agentFlowServiceURL == "" {
		return nil, fmt.Errorf("agentflow service URL not configured")
	}

	// Prepare flow import request
	flowBytes, err := json.Marshal(flowJSON)
	if err != nil {
		return nil, fmt.Errorf("marshal flow JSON: %w", err)
	}

	importRequest := map[string]any{
		"flow":       json.RawMessage(flowBytes),
		"force":      force,
		"project_id": projectID,
	}

	// Import flow via AgentFlow service (which proxies to LangFlow)
	endpoint := fmt.Sprintf("%s/flows/%s/sync", strings.TrimRight(agentFlowServiceURL, "/"), flowID)
	
	body, err := json.Marshal(importRequest)
	if err != nil {
		return nil, fmt.Errorf("marshal import request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(string(body)))
	if err != nil {
		return nil, fmt.Errorf("build import request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("import flow: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("import flow failed with status %d: %s", resp.StatusCode, string(bodyBytes))
	}

	var result map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode import response: %w", err)
	}

	return result, nil
}

// Helper functions

func extractNode(row map[string]any, key string) (map[string]any, bool) {
	val, ok := row[key]
	if !ok {
		return nil, false
	}

	nodeMap, ok := val.(map[string]any)
	if !ok {
		return nil, false
	}

	props, ok := nodeMap["properties"].(map[string]any)
	if !ok {
		props = make(map[string]any)
	}

	// Extract from properties or node directly
	result := make(map[string]any)
	for k, v := range props {
		result[k] = v
	}

	// Also check direct properties
	if id, ok := nodeMap["id"].(string); ok {
		result["id"] = id
	}
	if labels, ok := nodeMap["labels"].([]any); ok && len(labels) > 0 {
		if label, ok := labels[0].(string); ok {
			result["type"] = label
		}
	}

	return result, true
}

func getString(node map[string]any, key string) string {
	if val, ok := node[key]; ok {
		if str, ok := val.(string); ok {
			return str
		}
	}
	return ""
}

func getMap(node map[string]any, key string) map[string]any {
	if val, ok := node[key]; ok {
		if m, ok := val.(map[string]any); ok {
			return m
		}
	}
	return make(map[string]any)
}

func containsSQL(queries []SQLQueryInfo, id string) bool {
	for _, q := range queries {
		if q.ID == id {
			return true
		}
	}
	return false
}

func containsTable(tables []TableInfo, id string) bool {
	for _, t := range tables {
		if t.ID == id {
			return true
		}
	}
	return false
}

func containsDataFlowStep(steps []DataFlowStep, step DataFlowStep) bool {
	for _, s := range steps {
		if s.Source == step.Source && s.Target == step.Target {
			return true
		}
	}
	return false
}

