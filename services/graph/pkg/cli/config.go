package cli

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

// ProjectConfig captures the minimal data needed to describe a demo graph.
type ProjectConfig struct {
	Name         string            `json:"name" yaml:"name"`
	Description  string            `json:"description" yaml:"description"`
	Checkpoint   string            `json:"checkpoint" yaml:"checkpoint"`
	InitialInput float64           `json:"initial_input" yaml:"initial_input"`
	Graph        GraphConfig       `json:"graph" yaml:"graph"`
	Metadata     map[string]string `json:"metadata" yaml:"metadata"`
}

// GraphConfig declaratively describes a simple linear pipeline for now.
type GraphConfig struct {
	Nodes        []GraphNode            `json:"nodes" yaml:"nodes"`
	Edges        []GraphEdge            `json:"edges" yaml:"edges"`
	Conditionals []GraphConditionalEdge `json:"conditional_edges" yaml:"conditional_edges"`
	Entry        string                 `json:"entry" yaml:"entry"`
	Exit         string                 `json:"exit" yaml:"exit"`
	Options      GraphOptions           `json:"options" yaml:"options"`
}

// GraphConditionalEdge maps route labels emitted by a node to downstream destinations.
type GraphConditionalEdge struct {
	Source  string            `json:"source" yaml:"source"`
	PathMap map[string]string `json:"path_map" yaml:"path_map"`
}

// GraphNode defines a named node with an operation applied to the incoming value.
type GraphNode struct {
	ID      string         `json:"id" yaml:"id"`
	Op      string         `json:"op" yaml:"op"`
	Args    []any          `json:"args" yaml:"args"`
	Options map[string]any `json:"options" yaml:"options"`
}

// GraphOptions controls execution behavior.
type GraphOptions struct {
	Parallelism   int    `json:"parallelism" yaml:"parallelism"`
	ExecutionMode string `json:"execution_mode" yaml:"execution_mode"`
}

// GraphEdge connects two nodes.
type GraphEdge struct {
	From  string `json:"from" yaml:"from"`
	To    string `json:"to" yaml:"to"`
	Label string `json:"label" yaml:"label"`
}

// LoadProjectConfig loads config from a JSON or YAML file based on extension.
func LoadProjectConfig(path string) (*ProjectConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}

	cfg := &ProjectConfig{}
	if hasYAMLExt(path) {
		if err := yaml.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse yaml: %w", err)
		}
	} else {
		if err := json.Unmarshal(data, cfg); err != nil {
			return nil, fmt.Errorf("parse json: %w", err)
		}
	}

	return cfg, nil
}

func hasYAMLExt(path string) bool {
	lower := strings.ToLower(path)
	return strings.HasSuffix(lower, ".yaml") || strings.HasSuffix(lower, ".yml")
}
