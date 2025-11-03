package flightcatalog

import (
	"context"
	"fmt"

	"github.com/plturrell/agenticAiETH/agenticAiETH_layer4_AgentSDK/pkg/flightclient"
)

// Catalog mirrors the Arrow Flight datasets exposed by the Agent SDK.
type Catalog struct {
	Suites []flightclient.ServiceSuiteInfo
	Tools  []flightclient.ToolInfo
}

// Fetch retrieves the current catalog from the Agent SDK Flight server.
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
