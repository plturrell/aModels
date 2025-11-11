package cli

import "strings"

const DefaultDevCheckpoint = "sqlite:langgraph.dev.db"

// DefaultGraphConfig returns the built-in linear pipeline used by the demo.
func DefaultGraphConfig() GraphConfig {
	return GraphConfig{
		Nodes: []GraphNode{
			{ID: "ingest", Op: "add", Args: []any{1}},
			{ID: "transform", Op: "multiply", Args: []any{2}},
			{ID: "evaluate", Op: "add", Args: []any{-3}},
		},
		Edges: []GraphEdge{
			{From: "ingest", To: "transform"},
			{From: "transform", To: "evaluate"},
		},
		Entry: "ingest",
		Exit:  "evaluate",
	}
}

// EnsureProjectDefaults populates derived defaults on the provided config.
func EnsureProjectDefaults(cfg *ProjectConfig) {
	if cfg == nil {
		return
	}
    if cfg.Checkpoint == "" {
        cfg.Checkpoint = DefaultDevCheckpoint
	}
	if len(cfg.Graph.Nodes) == 0 {
		cfg.Graph = DefaultGraphConfig()
	}
	if cfg.Graph.Options.Parallelism < 0 {
		cfg.Graph.Options.Parallelism = 0
	}
	if mode := strings.TrimSpace(cfg.Graph.Options.ExecutionMode); mode != "" {
		cfg.Graph.Options.ExecutionMode = strings.ToLower(mode)
	}
	if cfg.Graph.Entry == "" && len(cfg.Graph.Nodes) > 0 {
		cfg.Graph.Entry = cfg.Graph.Nodes[0].ID
	}
	if cfg.Graph.Exit == "" && len(cfg.Graph.Nodes) > 0 {
		cfg.Graph.Exit = cfg.Graph.Nodes[len(cfg.Graph.Nodes)-1].ID
	}
	for i := range cfg.Graph.Nodes {
		if cfg.Graph.Nodes[i].Options == nil {
			cfg.Graph.Nodes[i].Options = map[string]any{}
		}
	}
}
