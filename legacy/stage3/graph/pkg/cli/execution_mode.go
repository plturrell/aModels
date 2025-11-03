package cli

import (
	"fmt"
	"strings"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
)

func parseExecutionMode(raw string) (graph.ExecutionMode, error) {
	val := strings.ToLower(strings.TrimSpace(raw))
	switch val {
	case "", "async", "asynchronous", "default":
		return graph.ExecutionModeAsync, nil
	case "sync", "synchronous", "barrier", "pregel":
		return graph.ExecutionModeSynchronous, nil
	default:
		if raw == "" {
			return graph.ExecutionModeAsync, nil
		}
		return graph.ExecutionModeAsync, fmt.Errorf("unknown execution mode %q", raw)
	}
}
