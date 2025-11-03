package cli

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"strings"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
)

// RunConfig describes the parameters supplied to the run command.
type RunConfig struct {
	ProjectPath         string
	Resume              bool
	Input               float64
	OverrideInput       bool
	EventSink           io.Writer
	OverrideParallelism int
	OverrideMode        string
}

// RunProject loads a project config from disk and executes it using the Go
// runtime.
func RunProject(ctx context.Context, cfg RunConfig, logger Logger) error {
	projectFile := cfg.ProjectPath
	if projectFile == "" {
		projectFile = "langgraph.project.json"
	}
	if !filepath.IsAbs(projectFile) {
		projectFile = filepath.Clean(projectFile)
	}

	project, err := LoadProjectConfig(projectFile)
	if err != nil {
		return err
	}

	return ExecuteProject(ctx, project, cfg, logger)
}

// ExecuteProject runs the supplied project configuration with the provided run
// options. This is shared between the CLI run command and the demo subcommand.
func ExecuteProject(ctx context.Context, project *ProjectConfig, cfg RunConfig, logger Logger) error {
	if project == nil {
		return fmt.Errorf("project config is nil")
	}
	EnsureProjectDefaults(project)

	if err := ValidateProjectConfig(project); err != nil {
		return err
	}

	logger = logger.WithEventSink(cfg.EventSink)

	stateManager, cleanup, err := BuildStateManager(project.Checkpoint)
	if err != nil {
		return err
	}
	if cleanup != nil {
		defer cleanup()
	}

	runtimeGraph, err := BuildGraphFromConfig(project.Graph, stateManager)
	if err != nil {
		return err
	}

	input := project.InitialInput
	if cfg.OverrideInput {
		input = cfg.Input
	}

	var opts []graph.RunOption
	parallel := project.Graph.Options.Parallelism
	if cfg.OverrideParallelism > 0 {
		parallel = cfg.OverrideParallelism
	}
	if parallel > 1 {
		opts = append(opts, graph.WithParallelism(parallel))
	}
	if cfg.Resume {
		opts = append(opts, graph.WithResume())
	}
	modeStr := project.Graph.Options.ExecutionMode
	if strings.TrimSpace(cfg.OverrideMode) != "" {
		modeStr = cfg.OverrideMode
	}
	mode, err := parseExecutionMode(modeStr)
	if err != nil {
		return err
	}
	if mode != graph.ExecutionModeAsync {
		opts = append(opts, graph.WithExecutionMode(mode))
	}

	logger.Event("graph_start", map[string]any{
		"project":    project.Name,
		"checkpoint": project.Checkpoint,
		"resume":     cfg.Resume,
		"input":      input,
		"mode":       mode.String(),
	})

	result, err := runtimeGraph.RunResult(ctx, input, opts...)
	if err != nil {
		return err
	}

	logger.Println("Project:", project.Name)
	logger.Println("Checkpoint backend:", project.Checkpoint)
	logger.Println("Execution mode:", mode.String())
	logger.Printf("Input: %.3f\n", input)
	logger.Printf("Output: %v\n", result.Default)
	logger.Event("graph_complete", map[string]any{
		"project": project.Name,
		"output":  result.Default,
		"resume":  cfg.Resume,
	})

	switch {
	case strings.HasPrefix(project.Checkpoint, "sqlite:"):
		logger.Printf("Checkpoint detail: SQLite (%s)\n", strings.TrimPrefix(project.Checkpoint, "sqlite:"))
	case strings.HasPrefix(project.Checkpoint, "redis"):
		logger.Println("Checkpoint detail: Redis (ensure REDIS_* env vars are set for production)")
	case project.Checkpoint == "hana":
		logger.Println("Checkpoint detail: HANA (environment configured)")
	default:
		logger.Println("Checkpoint detail: custom backend")
	}

	order := make([]string, 0, len(project.Graph.Nodes))
	nodeSeen := map[string]struct{}{}
	for _, node := range project.Graph.Nodes {
		if _, ok := nodeSeen[node.ID]; !ok {
			order = append(order, node.ID)
			nodeSeen[node.ID] = struct{}{}
		}
	}
	for id := range result.Outputs {
		key := string(id)
		if _, ok := nodeSeen[key]; !ok {
			order = append(order, key)
			nodeSeen[key] = struct{}{}
		}
	}

	logger.Println("Trace:")
	for _, id := range order {
		val, ok := result.Outputs[graph.NodeID(id)]
		if !ok {
			continue
		}
		logger.Printf("  %s %v\n", id, val)
		logger.Event("node_output", map[string]any{
			"node":  id,
			"value": fmt.Sprintf("%v", val),
		})
	}

	if cfg.Resume {
		if stateManager != nil {
			logger.Println("Resume enabled: future runs will reuse persisted state.")
			logger.Event("resume_ready", map[string]any{"project": project.Name, "checkpoint": project.Checkpoint})
		} else {
			logger.Println("Resume flag set but checkpoint backend does not support persistence.")
			logger.Event("resume_skipped", map[string]any{"project": project.Name})
		}
	}

	return nil
}
