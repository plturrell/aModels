package main

import (
	"fmt"
	"log"
	"strings"
)

// ValidationResult represents the result of data validation
type ValidationResult struct {
	Valid   bool
	Errors  []string
	Warnings []string
	Metrics ValidationMetrics
}

// ValidationMetrics tracks validation statistics
type ValidationMetrics struct {
	NodesValidated   int
	EdgesValidated   int
	NodesRejected    int
	EdgesRejected    int
	ValidationErrors int
}

// ValidateNodes validates a slice of nodes before storage
func ValidateNodes(nodes []Node, logger *log.Logger) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
		Metrics: ValidationMetrics{},
	}

	for i, node := range nodes {
		result.Metrics.NodesValidated++

		// Validate required fields
		if strings.TrimSpace(node.ID) == "" {
			result.Valid = false
			result.Metrics.NodesRejected++
			result.Metrics.ValidationErrors++
			result.Errors = append(result.Errors, fmt.Sprintf("node[%d]: missing required field 'id'", i))
			continue
		}

		if strings.TrimSpace(node.Type) == "" {
			result.Valid = false
			result.Metrics.NodesRejected++
			result.Metrics.ValidationErrors++
			result.Errors = append(result.Errors, fmt.Sprintf("node[%d] (id=%s): missing required field 'type'", i, node.ID))
			continue
		}

		// Validate ID format (should not contain special characters that could break queries)
		if strings.ContainsAny(node.ID, " \t\n\r") {
			result.Warnings = append(result.Warnings, fmt.Sprintf("node[%d] (id=%s): ID contains whitespace", i, node.ID))
		}

		// Validate properties if present
		if node.Props != nil {
			// Check for common issues in properties
			for key, value := range node.Props {
				if strings.TrimSpace(key) == "" {
					result.Warnings = append(result.Warnings, fmt.Sprintf("node[%d] (id=%s): empty property key", i, node.ID))
				}
				if value == nil {
					result.Warnings = append(result.Warnings, fmt.Sprintf("node[%d] (id=%s): property '%s' has nil value", i, node.ID, key))
				}
			}
		}
	}

	if logger != nil && len(result.Errors) > 0 {
		logger.Printf("Node validation: %d errors, %d warnings, %d nodes rejected", len(result.Errors), len(result.Warnings), result.Metrics.NodesRejected)
	}

	return result
}

// ValidateEdges validates a slice of edges before storage
func ValidateEdges(edges []Edge, nodes []Node, logger *log.Logger) ValidationResult {
	result := ValidationResult{
		Valid:    true,
		Errors:   []string{},
		Warnings: []string{},
		Metrics: ValidationMetrics{},
	}

	// Create a map of valid node IDs for quick lookup
	nodeIDMap := make(map[string]bool)
	for _, node := range nodes {
		nodeIDMap[node.ID] = true
	}

	for i, edge := range edges {
		result.Metrics.EdgesValidated++

		// Validate required fields
		if strings.TrimSpace(edge.SourceID) == "" {
			result.Valid = false
			result.Metrics.EdgesRejected++
			result.Metrics.ValidationErrors++
			result.Errors = append(result.Errors, fmt.Sprintf("edge[%d]: missing required field 'source_id'", i))
			continue
		}

		if strings.TrimSpace(edge.TargetID) == "" {
			result.Valid = false
			result.Metrics.EdgesRejected++
			result.Metrics.ValidationErrors++
			result.Errors = append(result.Errors, fmt.Sprintf("edge[%d]: missing required field 'target_id'", i))
			continue
		}

		// Validate that source and target nodes exist
		if !nodeIDMap[edge.SourceID] {
			result.Warnings = append(result.Warnings, fmt.Sprintf("edge[%d]: source node '%s' not found in node list", i, edge.SourceID))
		}

		if !nodeIDMap[edge.TargetID] {
			result.Warnings = append(result.Warnings, fmt.Sprintf("edge[%d]: target node '%s' not found in node list", i, edge.TargetID))
		}

		// Validate self-loops (warn but don't reject)
		if edge.SourceID == edge.TargetID {
			result.Warnings = append(result.Warnings, fmt.Sprintf("edge[%d]: self-loop detected (source=%s, target=%s)", i, edge.SourceID, edge.TargetID))
		}

		// Validate properties if present
		if edge.Props != nil {
			for key, value := range edge.Props {
				if strings.TrimSpace(key) == "" {
					result.Warnings = append(result.Warnings, fmt.Sprintf("edge[%d]: empty property key", i))
				}
				if value == nil {
					result.Warnings = append(result.Warnings, fmt.Sprintf("edge[%d]: property '%s' has nil value", i, key))
				}
			}
		}
	}

	if logger != nil && len(result.Errors) > 0 {
		logger.Printf("Edge validation: %d errors, %d warnings, %d edges rejected", len(result.Errors), len(result.Warnings), result.Metrics.EdgesRejected)
	}

	return result
}

// ValidateGraph validates both nodes and edges together
func ValidateGraph(nodes []Node, edges []Edge, logger *log.Logger) ValidationResult {
	nodeResult := ValidateNodes(nodes, logger)
	edgeResult := ValidateEdges(edges, nodes, logger)

	combined := ValidationResult{
		Valid:    nodeResult.Valid && edgeResult.Valid,
		Errors:   append(nodeResult.Errors, edgeResult.Errors...),
		Warnings: append(nodeResult.Warnings, edgeResult.Warnings...),
		Metrics: ValidationMetrics{
			NodesValidated:   nodeResult.Metrics.NodesValidated,
			EdgesValidated:   edgeResult.Metrics.EdgesValidated,
			NodesRejected:    nodeResult.Metrics.NodesRejected,
			EdgesRejected:    edgeResult.Metrics.EdgesRejected,
			ValidationErrors: nodeResult.Metrics.ValidationErrors + edgeResult.Metrics.ValidationErrors,
		},
	}

	if logger != nil {
		logger.Printf("Graph validation complete: valid=%v, nodes=%d/%d valid, edges=%d/%d valid, errors=%d, warnings=%d",
			combined.Valid,
			nodeResult.Metrics.NodesValidated-nodeResult.Metrics.NodesRejected, nodeResult.Metrics.NodesValidated,
			edgeResult.Metrics.EdgesValidated-edgeResult.Metrics.EdgesRejected, edgeResult.Metrics.EdgesValidated,
			len(combined.Errors), len(combined.Warnings))
	}

	return combined
}

// FilterValidNodes filters out invalid nodes based on validation result
func FilterValidNodes(nodes []Node, validationResult ValidationResult) []Node {
	if validationResult.Valid {
		return nodes
	}

	// Create a map of rejected node indices
	rejectedIndices := make(map[int]bool)
	for _, err := range validationResult.Errors {
		// Parse error message to extract node index
		// Format: "node[%d]: ..."
		if strings.HasPrefix(err, "node[") {
			var idx int
			if _, err := fmt.Sscanf(err, "node[%d]:", &idx); err == nil {
				rejectedIndices[idx] = true
			}
		}
	}

	// Filter out rejected nodes
	validNodes := make([]Node, 0, len(nodes))
	for i, node := range nodes {
		if !rejectedIndices[i] {
			validNodes = append(validNodes, node)
		}
	}

	return validNodes
}

// FilterValidEdges filters out invalid edges based on validation result
func FilterValidEdges(edges []Edge, validationResult ValidationResult) []Edge {
	if validationResult.Valid {
		return edges
	}

	// Create a map of rejected edge indices
	rejectedIndices := make(map[int]bool)
	for _, err := range validationResult.Errors {
		// Parse error message to extract edge index
		// Format: "edge[%d]: ..."
		if strings.HasPrefix(err, "edge[") {
			var idx int
			if _, err := fmt.Sscanf(err, "edge[%d]:", &idx); err == nil {
				rejectedIndices[idx] = true
			}
		}
	}

	// Filter out rejected edges
	validEdges := make([]Edge, 0, len(edges))
	for i, edge := range edges {
		if !rejectedIndices[i] {
			validEdges = append(validEdges, edge)
		}
	}

	return validEdges
}

