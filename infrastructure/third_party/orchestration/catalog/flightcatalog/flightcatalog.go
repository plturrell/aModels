package flightcatalog

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightclient"
)

// Catalog bundles suite and tool metadata fetched from the Agent SDK Flight server.
type Catalog struct {
	Suites []flightclient.ServiceSuiteInfo
	Tools  []flightclient.ToolInfo
}

// Fetch contacts the Agent SDK Arrow Flight endpoint and returns the current catalog.
func Fetch(ctx context.Context, flightAddr string) (Catalog, error) {
	client, err := flightclient.Dial(ctx, flightAddr)
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
