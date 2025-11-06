//go:build hana && blockchain

package cli

import (
	"fmt"

	"github.com/langchain-ai/langgraph-go/pkg/graph"
)

// buildBlockchainCheckpointManager is stubbed - blockchain functionality not available
func buildBlockchainCheckpointManager(mode string) (*graph.StateManager, func() error, error) {
	return nil, nil, fmt.Errorf("blockchain checkpoint manager not available (stubbed)")
}
