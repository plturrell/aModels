package graph

// Node represents a node in the extract service graph.
type Node struct {
	ID    string         `json:"id"`
	Type  string         `json:"type"`
	Label string         `json:"label"`
	Props map[string]any `json:"properties,omitempty"`
}

// Edge represents an edge in the extract service graph.
type Edge struct {
	SourceID string         `json:"source"`
	TargetID string         `json:"target"`
	Label    string         `json:"label"`
	Props    map[string]any `json:"properties,omitempty"`
}

