package flightcatalog

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightclient"
)

// Catalog summarizes the Arrow Flight listings the graph service cares about.
type Catalog struct {
	Suites []flightclient.ServiceSuiteInfo
	Tools  []flightclient.ToolInfo
}

// Fetch retrieves the catalog from the Agent SDK Arrow Flight server.
func Fetch(ctx context.Context, addr string) (Catalog, error) {
	client, err := flightclient.Dial(ctx, addr)
	if err != nil {
		return Catalog{}, fmt.Errorf("dial flight server: %w", err)
	}
	defer client.Close()

	suites, err := client.ListServiceSuites(ctx)
	if err != nil {
		return Catalog{}, fmt.Errorf("list service suites: %w", err)
	}

	tools, err := client.ListTools(ctx)
	if err != nil {
		return Catalog{}, fmt.Errorf("list tools: %w", err)
	}

	return Catalog{
		Suites: suites,
		Tools:  tools,
	}, nil
}
