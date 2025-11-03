package cli_test

import (
	"context"
	"io"
	"log"
	"strings"
	"testing"

	"github.com/langchain-ai/langgraph-go/pkg/cli"
)

func testLogger() cli.Logger {
	return cli.Logger{Logger: log.New(io.Discard, "", 0)}
}

func TestExecuteProjectOverrideMode(t *testing.T) {
	project := &cli.ProjectConfig{
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{
				{ID: "start", Op: "set", Args: []any{1}},
			},
		},
	}

	cfg := cli.RunConfig{OverrideMode: "sync"}
	if err := cli.ExecuteProject(context.Background(), project, cfg, testLogger()); err != nil {
		t.Fatalf("ExecuteProject with sync override failed: %v", err)
	}
}

func TestExecuteProjectInvalidOverrideMode(t *testing.T) {
	project := &cli.ProjectConfig{
		Graph: cli.GraphConfig{
			Nodes: []cli.GraphNode{
				{ID: "start", Op: "set", Args: []any{1}},
			},
		},
	}

	cfg := cli.RunConfig{OverrideMode: "invalid-mode"}
	err := cli.ExecuteProject(context.Background(), project, cfg, testLogger())
	if err == nil || !strings.Contains(err.Error(), "unknown execution mode") {
		t.Fatalf("expected execution mode error, got %v", err)
	}
}
