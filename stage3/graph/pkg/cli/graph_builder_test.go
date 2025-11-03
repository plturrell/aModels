package cli_test

import (
	"context"
	"strings"
	"testing"

	"github.com/langchain-ai/langgraph-go/pkg/cli"
)

func TestBuildGraphJoinOptions(t *testing.T) {
	cfg := cli.GraphConfig{
		Entry: "src",
		Exit:  "join",
		Nodes: []cli.GraphNode{
			{ID: "src", Op: "set", Args: []any{0}},
			{ID: "left", Op: "set", Args: []any{2}},
			{ID: "right", Op: "set", Args: []any{3}},
			{ID: "join", Op: "join", Options: map[string]any{"aggregate": "sum"}},
		},
		Edges: []cli.GraphEdge{
			{From: "src", To: "left"},
			{From: "src", To: "right"},
			{From: "left", To: "join"},
			{From: "right", To: "join"},
		},
	}

	g, err := cli.BuildGraphFromConfig(cfg, nil)
	if err != nil {
		t.Fatalf("BuildGraphFromConfig error: %v", err)
	}

	res, err := g.RunResult(context.Background(), 0)
	if err != nil {
		t.Fatalf("RunResult error: %v", err)
	}
	if got := res.Default.(float64); got != 5 {
		t.Fatalf("expected sum 5, got %v", got)
	}
}

func TestBuildGraphNodeExecOptionValidation(t *testing.T) {
	cfg := cli.GraphConfig{
		Nodes: []cli.GraphNode{
			{ID: "slow", Op: "noop", Options: map[string]any{"timeout_ms": -10}},
		},
	}

	if _, err := cli.BuildGraphFromConfig(cfg, nil); err == nil || !strings.Contains(err.Error(), "timeout_ms cannot be negative") {
		t.Fatalf("expected timeout validation error, got %v", err)
	}

	cfg.Nodes[0].Options = map[string]any{"retry": 1.5}
	if _, err := cli.BuildGraphFromConfig(cfg, nil); err == nil || !strings.Contains(err.Error(), "retry must be an integer") {
		t.Fatalf("expected retry integer validation error, got %v", err)
	}

	cfg.Nodes[0].Options = map[string]any{"retry_delay_ms": -5}
	if _, err := cli.BuildGraphFromConfig(cfg, nil); err == nil || !strings.Contains(err.Error(), "retry_delay_ms cannot be negative") {
		t.Fatalf("expected retry delay validation error, got %v", err)
	}
}

func TestBuildGraphUnknownNodeOption(t *testing.T) {
	cfg := cli.GraphConfig{
		Nodes: []cli.GraphNode{
			{ID: "noop", Op: "noop", Options: map[string]any{"bogus": true}},
		},
	}

	if _, err := cli.BuildGraphFromConfig(cfg, nil); err == nil || !strings.Contains(err.Error(), "unknown option \"bogus\"") {
		t.Fatalf("expected unknown option error, got %v", err)
	}
}

func TestBuildGraphUnknownJoinOption(t *testing.T) {
	cfg := cli.GraphConfig{
		Nodes: []cli.GraphNode{
			{ID: "join", Op: "join", Options: map[string]any{"aggregate": "sum", "weird": 1}},
		},
	}

	if _, err := cli.BuildGraphFromConfig(cfg, nil); err == nil || !strings.Contains(err.Error(), "unknown option \"weird\"") {
		t.Fatalf("expected unknown join option error, got %v", err)
	}
}

func TestBuildGraphBranchRoutes(t *testing.T) {
	cfg := cli.GraphConfig{
		Entry: "branch",
		Nodes: []cli.GraphNode{
			{ID: "branch", Op: "branch", Args: []any{"gt", 0, "positive", "negative"}},
			{ID: "positive", Op: "set", Args: []any{10}},
			{ID: "negative", Op: "set", Args: []any{-10}},
		},
		Conditionals: []cli.GraphConditionalEdge{
			{Source: "branch", PathMap: map[string]string{"positive": "positive", "negative": "negative"}},
		},
	}

	g, err := cli.BuildGraphFromConfig(cfg, nil)
	if err != nil {
		t.Fatalf("BuildGraphFromConfig error: %v", err)
	}

	res, err := g.RunResult(context.Background(), 5)
	if err != nil {
		t.Fatalf("RunResult positive input: %v", err)
	}
	if _, ok := res.Outputs["positive"]; !ok {
		t.Fatalf("expected positive node to execute, outputs=%v", res.Outputs)
	}
	if _, ok := res.Outputs["negative"]; ok {
		t.Fatalf("negative path should not execute for positive input, outputs=%v", res.Outputs)
	}

	res, err = g.RunResult(context.Background(), -3)
	if err != nil {
		t.Fatalf("RunResult negative input: %v", err)
	}
	if _, ok := res.Outputs["negative"]; !ok {
		t.Fatalf("expected negative node to execute, outputs=%v", res.Outputs)
	}
	if _, ok := res.Outputs["positive"]; ok {
		t.Fatalf("positive path should not execute for negative input, outputs=%v", res.Outputs)
	}
}
