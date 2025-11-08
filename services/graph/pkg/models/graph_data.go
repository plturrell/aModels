package models

import (
	"encoding/json"
	"fmt"
)

// GraphData represents a unified graph data format used across all services.
// This standardizes the format between Neo4j (KG) and GNN services.
type GraphData struct {
	Nodes    []Node   `json:"nodes"`
	Edges    []Edge   `json:"edges"`
	Metadata Metadata `json:"metadata,omitempty"`
	Quality  *Quality `json:"quality,omitempty"`
}

// Node represents a node in the unified graph format.
type Node struct {
	ID         string         `json:"id"`
	Type       string         `json:"type"`
	Label      string         `json:"label,omitempty"`
	Properties map[string]any `json:"properties,omitempty"`
}

// Edge represents an edge in the unified graph format.
type Edge struct {
	SourceID   string         `json:"source"`
	TargetID   string         `json:"target"`
	Label      string         `json:"label,omitempty"`
	Properties map[string]any `json:"properties,omitempty"`
}

// Metadata contains graph-level metadata.
type Metadata struct {
	ProjectID           string            `json:"project_id,omitempty"`
	SystemID            string            `json:"system_id,omitempty"`
	InformationSystemID string            `json:"information_system_id,omitempty"`
	RootNodeID          string            `json:"root_node_id,omitempty"`
	MetadataEntropy     float64           `json:"metadata_entropy,omitempty"`
	KLDivergence        float64           `json:"kl_divergence,omitempty"`
	Warnings            []string          `json:"warnings,omitempty"`
	Additional          map[string]any    `json:"additional,omitempty"`
}

// Quality represents data quality metrics.
type Quality struct {
	Score             float64  `json:"score"`
	Level             string   `json:"level"`
	Issues            []string `json:"issues,omitempty"`
	Recommendations   []string `json:"recommendations,omitempty"`
	ProcessingStrategy string `json:"processing_strategy,omitempty"`
}

// FromNeo4j converts Neo4j Cypher query result to unified GraphData format.
func FromNeo4j(neo4jResult map[string]any) (*GraphData, error) {
	gd := &GraphData{
		Nodes: []Node{},
		Edges: []Edge{},
		Metadata: Metadata{},
	}

	// Handle Neo4j Cypher result format
	// Format: {"columns": [...], "data": [[...], ...]}
	if columns, ok := neo4jResult["columns"].([]any); ok {
		if data, ok := neo4jResult["data"].([]any); ok {
			// Extract nodes and edges from Cypher result
			for _, row := range data {
				if rowArray, ok := row.([]any); ok {
					for i, col := range columns {
						if i >= len(rowArray) {
							continue
						}
						value := rowArray[i]
						
						// Check if value is a node or relationship
						if valueMap, ok := value.(map[string]any); ok {
							// Check for Neo4j node format
							if labels, hasLabels := valueMap["labels"].([]any); hasLabels {
								// This is a Neo4j node
								nodeID := ""
								if id, ok := valueMap["id"].(float64); ok {
									nodeID = fmt.Sprintf("%.0f", id)
								} else if elementId, ok := valueMap["element_id"].(string); ok {
									nodeID = elementId
								}
								
								properties := make(map[string]any)
								if props, ok := valueMap["properties"].(map[string]any); ok {
									properties = props
								}
								
								nodeType := "unknown"
								if len(labels) > 0 {
									if labelStr, ok := labels[0].(string); ok {
										nodeType = labelStr
									}
								}
								
								gd.Nodes = append(gd.Nodes, Node{
									ID:         nodeID,
									Type:       nodeType,
									Label:      nodeID, // Default label to ID
									Properties: properties,
								})
							} else if startNode, hasStart := valueMap["startNodeElementId"].(string); hasStart {
								// This is a Neo4j relationship
								endNode := ""
								if end, ok := valueMap["endNodeElementId"].(string); ok {
									endNode = end
								}
								
								relType := ""
								if rType, ok := valueMap["type"].(string); ok {
									relType = rType
								}
								
								properties := make(map[string]any)
								if props, ok := valueMap["properties"].(map[string]any); ok {
									properties = props
								}
								
								gd.Edges = append(gd.Edges, Edge{
									SourceID:   startNode,
									TargetID:   endNode,
									Label:      relType,
									Properties: properties,
								})
							}
						}
					}
				}
			}
		}
	}

	// Handle direct nodes/edges format (from knowledge graph processing)
	if nodesData, ok := neo4jResult["nodes"].([]any); ok {
		gd.Nodes = convertNodesFromAny(nodesData)
	}
	if edgesData, ok := neo4jResult["edges"].([]any); ok {
		gd.Edges = convertEdgesFromAny(edgesData)
	}

	// Extract metadata
	if metadata, ok := neo4jResult["metadata"].(map[string]any); ok {
		if projectID, ok := metadata["project_id"].(string); ok {
			gd.Metadata.ProjectID = projectID
		}
		if systemID, ok := metadata["system_id"].(string); ok {
			gd.Metadata.SystemID = systemID
		}
		if rootNodeID, ok := metadata["root_node_id"].(string); ok {
			gd.Metadata.RootNodeID = rootNodeID
		}
	}

	// Extract quality
	if qualityData, ok := neo4jResult["quality"].(map[string]any); ok {
		quality := &Quality{}
		if score, ok := qualityData["score"].(float64); ok {
			quality.Score = score
		}
		if level, ok := qualityData["level"].(string); ok {
			quality.Level = level
		}
		if issues, ok := qualityData["issues"].([]any); ok {
			quality.Issues = convertStringSlice(issues)
		}
		if recommendations, ok := qualityData["recommendations"].([]any); ok {
			quality.Recommendations = convertStringSlice(recommendations)
		}
		gd.Quality = quality
	}

	return gd, nil
}

// ToGNN converts unified GraphData to GNN service format (nodes/edges arrays).
func (gd *GraphData) ToGNN() map[string]any {
	return map[string]any{
		"nodes": gd.Nodes,
		"edges": gd.Edges,
	}
}

// FromGNN converts GNN service response to unified GraphData format.
func FromGNN(gnnResponse map[string]any) (*GraphData, error) {
	gd := &GraphData{
		Nodes:    []Node{},
		Edges:    []Edge{},
		Metadata: Metadata{},
	}

	// Extract nodes
	if nodesData, ok := gnnResponse["nodes"].([]any); ok {
		gd.Nodes = convertNodesFromAny(nodesData)
	} else if nodesData, ok := gnnResponse["node_embeddings"].(map[string]any); ok {
		// Handle node embeddings format
		for nodeID := range nodesData {
			gd.Nodes = append(gd.Nodes, Node{
				ID:   nodeID,
				Type: "unknown",
			})
		}
	}

	// Extract edges
	if edgesData, ok := gnnResponse["edges"].([]any); ok {
		gd.Edges = convertEdgesFromAny(edgesData)
	}

	// Extract metadata if present
	if metadata, ok := gnnResponse["metadata"].(map[string]any); ok {
		if projectID, ok := metadata["project_id"].(string); ok {
			gd.Metadata.ProjectID = projectID
		}
		if systemID, ok := metadata["system_id"].(string); ok {
			gd.Metadata.SystemID = systemID
		}
	}

	return gd, nil
}

// ToNeo4j converts unified GraphData to Neo4j-compatible format.
func (gd *GraphData) ToNeo4j() map[string]any {
	result := map[string]any{
		"nodes": gd.Nodes,
		"edges": gd.Edges,
	}

	if gd.Metadata.ProjectID != "" || gd.Metadata.SystemID != "" {
		result["metadata"] = map[string]any{
			"project_id": gd.Metadata.ProjectID,
			"system_id":  gd.Metadata.SystemID,
		}
	}

	if gd.Quality != nil {
		result["quality"] = gd.Quality
	}

	return result
}

// Merge combines another GraphData into this one, deduplicating nodes and edges.
func (gd *GraphData) Merge(other *GraphData) {
	// Create node map for deduplication
	nodeMap := make(map[string]Node)
	for _, node := range gd.Nodes {
		nodeMap[node.ID] = node
	}
	for _, node := range other.Nodes {
		if existing, exists := nodeMap[node.ID]; exists {
			// Merge properties
			if existing.Properties == nil {
				existing.Properties = make(map[string]any)
			}
			for k, v := range node.Properties {
				existing.Properties[k] = v
			}
			nodeMap[node.ID] = existing
		} else {
			nodeMap[node.ID] = node
		}
	}

	// Rebuild nodes slice
	gd.Nodes = make([]Node, 0, len(nodeMap))
	for _, node := range nodeMap {
		gd.Nodes = append(gd.Nodes, node)
	}

	// Create edge map for deduplication (key: source-target-label)
	edgeMap := make(map[string]Edge)
	for _, edge := range gd.Edges {
		key := fmt.Sprintf("%s-%s-%s", edge.SourceID, edge.TargetID, edge.Label)
		edgeMap[key] = edge
	}
	for _, edge := range other.Edges {
		key := fmt.Sprintf("%s-%s-%s", edge.SourceID, edge.TargetID, edge.Label)
		if existing, exists := edgeMap[key]; exists {
			// Merge properties
			if existing.Properties == nil {
				existing.Properties = make(map[string]any)
			}
			for k, v := range edge.Properties {
				existing.Properties[k] = v
			}
			edgeMap[key] = existing
		} else {
			edgeMap[key] = edge
		}
	}

	// Rebuild edges slice
	gd.Edges = make([]Edge, 0, len(edgeMap))
	for _, edge := range edgeMap {
		gd.Edges = append(gd.Edges, edge)
	}

	// Merge metadata (prefer other's values if present)
	if other.Metadata.ProjectID != "" {
		gd.Metadata.ProjectID = other.Metadata.ProjectID
	}
	if other.Metadata.SystemID != "" {
		gd.Metadata.SystemID = other.Metadata.SystemID
	}
	if other.Metadata.RootNodeID != "" {
		gd.Metadata.RootNodeID = other.Metadata.RootNodeID
	}

	// Merge quality (use better quality score)
	if other.Quality != nil {
		if gd.Quality == nil || other.Quality.Score > gd.Quality.Score {
			gd.Quality = other.Quality
		}
	}
}

// Helper functions

func convertNodesFromAny(nodesData []any) []Node {
	nodes := make([]Node, 0, len(nodesData))
	for _, nodeData := range nodesData {
		if nodeMap, ok := nodeData.(map[string]any); ok {
			node := Node{
				Properties: make(map[string]any),
			}
			
			if id, ok := nodeMap["id"].(string); ok {
				node.ID = id
			} else if id, ok := nodeMap["node_id"].(string); ok {
				node.ID = id
			}
			
			if nodeType, ok := nodeMap["type"].(string); ok {
				node.Type = nodeType
			} else if nodeType, ok := nodeMap["node_type"].(string); ok {
				node.Type = nodeType
			}
			
			if label, ok := nodeMap["label"].(string); ok {
				node.Label = label
			}
			
			if props, ok := nodeMap["properties"].(map[string]any); ok {
				node.Properties = props
			} else {
				// Copy all non-standard fields to properties
				for k, v := range nodeMap {
					if k != "id" && k != "node_id" && k != "type" && k != "node_type" && k != "label" {
						node.Properties[k] = v
					}
				}
			}
			
			if node.ID != "" {
				nodes = append(nodes, node)
			}
		}
	}
	return nodes
}

func convertEdgesFromAny(edgesData []any) []Edge {
	edges := make([]Edge, 0, len(edgesData))
	for _, edgeData := range edgesData {
		if edgeMap, ok := edgeData.(map[string]any); ok {
			edge := Edge{
				Properties: make(map[string]any),
			}
			
			if source, ok := edgeMap["source"].(string); ok {
				edge.SourceID = source
			} else if source, ok := edgeMap["source_id"].(string); ok {
				edge.SourceID = source
			}
			
			if target, ok := edgeMap["target"].(string); ok {
				edge.TargetID = target
			} else if target, ok := edgeMap["target_id"].(string); ok {
				edge.TargetID = target
			}
			
			if label, ok := edgeMap["label"].(string); ok {
				edge.Label = label
			} else if relType, ok := edgeMap["relation_type"].(string); ok {
				edge.Label = relType
			}
			
			if props, ok := edgeMap["properties"].(map[string]any); ok {
				edge.Properties = props
			} else {
				// Copy all non-standard fields to properties
				for k, v := range edgeMap {
					if k != "source" && k != "source_id" && k != "target" && k != "target_id" && k != "label" && k != "relation_type" {
						edge.Properties[k] = v
					}
				}
			}
			
			if edge.SourceID != "" && edge.TargetID != "" {
				edges = append(edges, edge)
			}
		}
	}
	return edges
}

func convertStringSlice(slice []any) []string {
	result := make([]string, 0, len(slice))
	for _, item := range slice {
		if str, ok := item.(string); ok {
			result = append(result, str)
		}
	}
	return result
}

// Validate checks if GraphData is valid.
func (gd *GraphData) Validate() error {
	// Check for duplicate node IDs
	nodeIDs := make(map[string]bool)
	for _, node := range gd.Nodes {
		if node.ID == "" {
			return fmt.Errorf("node with empty ID found")
		}
		if nodeIDs[node.ID] {
			return fmt.Errorf("duplicate node ID: %s", node.ID)
		}
		nodeIDs[node.ID] = true
	}

	// Check edges reference valid nodes
	for _, edge := range gd.Edges {
		if edge.SourceID == "" || edge.TargetID == "" {
			return fmt.Errorf("edge with empty source or target ID")
		}
		if !nodeIDs[edge.SourceID] {
			return fmt.Errorf("edge references non-existent source node: %s", edge.SourceID)
		}
		if !nodeIDs[edge.TargetID] {
			return fmt.Errorf("edge references non-existent target node: %s", edge.TargetID)
		}
	}

	return nil
}

// ToJSON converts GraphData to JSON bytes.
func (gd *GraphData) ToJSON() ([]byte, error) {
	return json.Marshal(gd)
}

// FromJSON creates GraphData from JSON bytes.
func FromJSON(data []byte) (*GraphData, error) {
	var gd GraphData
	if err := json.Unmarshal(data, &gd); err != nil {
		return nil, fmt.Errorf("unmarshal graph data: %w", err)
	}
	return &gd, nil
}

