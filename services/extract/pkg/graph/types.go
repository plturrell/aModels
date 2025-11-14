package graph

import "fmt"

// Node represents a node in the extract service graph.
type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label"`
	Props map[string]any `json:"properties,omitempty"`
}

// NewNode creates a new Node with the given parameters.
// It does not validate the node type - use NewValidatedNode for validation.
func NewNode(id, nodeType, label string, props map[string]any) Node {
	return Node{
		ID:    id,
		Type:  nodeType,
		Label: label,
		Props: props,
	}
}

// NewValidatedNode creates a new Node and validates that the node type is recognized.
// Returns the node and an error if the node type is invalid.
func NewValidatedNode(id, nodeType, label string, props map[string]any) (Node, error) {
	if !IsValidNodeType(nodeType) {
		return Node{}, &InvalidNodeTypeError{Type: nodeType}
	}
	return NewNode(id, nodeType, label, props), nil
}

// InvalidNodeTypeError represents an error when an invalid node type is used.
type InvalidNodeTypeError struct {
	Type string
}

func (e *InvalidNodeTypeError) Error() string {
	return fmt.Sprintf("invalid node type: %q (use graph.NodeType* constants)", e.Type)
}

// Edge represents an edge in the extract service graph.
type Edge struct {
	SourceID string         `json:"source"`
	TargetID string         `json:"target"`
	Label    string         `json:"label"`
	Props    map[string]any `json:"properties,omitempty"`
}

