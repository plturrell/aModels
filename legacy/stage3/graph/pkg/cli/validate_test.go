package cli_test

import (
	"strings"
	"testing"

	"github.com/langchain-ai/langgraph-go/pkg/cli"
)

func TestValidateProjectConfigValid(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Entry: "entry",
			Exit:  "exit",
			Nodes: []cli.GraphNode{
				{ID: "entry", Op: "set", Args: []any{1}},
				{ID: "branch", Op: "branch"},
				{ID: "join", Op: "join"},
				{ID: "exit", Op: "set", Args: []any{0}},
			},
			Edges: []cli.GraphEdge{
				{From: "entry", To: "branch"},
				{From: "branch", To: "join", Label: "positive"},
				{From: "branch", To: "join", Label: "negative"},
				{From: "join", To: "exit"},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err != nil {
		t.Fatalf("validation should succeed, got %v", err)
	}
}

func TestValidateProjectConfigCycle(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{{ID: "a"}, {ID: "b"}},
			Edges: []cli.GraphEdge{
				{From: "a", To: "b"},
				{From: "b", To: "a"},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err == nil {
		t.Fatalf("expected cycle detection error")
	}
}

func TestValidateProjectConfigBranchLabels(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{{ID: "branch", Op: "branch"}, {ID: "out"}},
			Edges: []cli.GraphEdge{
				{From: "branch", To: "out"},
			},
		},
	}

	err := cli.ValidateProjectConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "branch node") {
		t.Fatalf("expected branch validation error, got %v", err)
	}
}

func TestValidateProjectConfigConditionalEdges(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Entry: "branch",
			Nodes: []cli.GraphNode{
				{ID: "branch", Op: "branch"},
				{ID: "pos", Op: "set"},
				{ID: "neg", Op: "set"},
			},
			Conditionals: []cli.GraphConditionalEdge{
				{Source: "branch", PathMap: map[string]string{"positive": "pos", "negative": "neg"}},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err != nil {
		t.Fatalf("validation should succeed with conditional edges, got %v", err)
	}

	cfg.Graph.Conditionals[0].PathMap["unknown"] = "missing"
	if err := cli.ValidateProjectConfig(cfg); err == nil || !strings.Contains(err.Error(), "unknown node") {
		t.Fatalf("expected invalid conditional mapping error, got %v", err)
	}
}

func TestValidateProjectConfigUnreachable(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Entry: "start",
			Nodes: []cli.GraphNode{{ID: "start"}, {ID: "exit"}, {ID: "isolated"}},
			Edges: []cli.GraphEdge{{From: "start", To: "exit"}},
		},
	}

	err := cli.ValidateProjectConfig(cfg)
	if err == nil || !strings.Contains(err.Error(), "unreachable") {
		t.Fatalf("expected unreachable error, got %v", err)
	}
}

func TestValidateProjectConfigUnknownNodeOption(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{
				{ID: "node", Op: "set", Options: map[string]any{"bad": true}},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err == nil || !strings.Contains(err.Error(), "unknown option \"bad\"") {
		t.Fatalf("expected unknown option error, got %v", err)
	}
}

func TestValidateProjectConfigJoinOptionValidation(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{
				{ID: "start", Op: "set"},
				{ID: "left", Op: "set"},
				{ID: "right", Op: "set"},
				{ID: "join", Op: "join", Options: map[string]any{"aggregate": "bogus"}},
			},
			Edges: []cli.GraphEdge{
				{From: "start", To: "left"},
				{From: "start", To: "right"},
				{From: "left", To: "join"},
				{From: "right", To: "join"},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err == nil || !strings.Contains(err.Error(), "unknown join aggregate") {
		t.Fatalf("expected join aggregate error, got %v", err)
	}
}

func TestValidateProjectConfigInvalidCheckpoint(t *testing.T) {
	cfg := &cli.ProjectConfig{Checkpoint: "memory"}
	if err := cli.ValidateProjectConfig(cfg); err == nil || !strings.Contains(err.Error(), "unsupported checkpoint") {
		t.Fatalf("expected checkpoint validation error, got %v", err)
	}
}

func TestValidateProjectConfigUnknownExecutionMode(t *testing.T) {
	cfg := &cli.ProjectConfig{
		Checkpoint: cli.DefaultDevCheckpoint,
		Graph: cli.GraphConfig{
			Options: cli.GraphOptions{
				ExecutionMode: "unsupported",
			},
			Nodes: []cli.GraphNode{
				{ID: "start", Op: "set"},
			},
		},
	}

	if err := cli.ValidateProjectConfig(cfg); err == nil || !strings.Contains(err.Error(), "unknown execution mode") {
		t.Fatalf("expected execution mode error, got %v", err)
	}
}
