package flightcatalog

import (
	"context"
	"fmt"
)

// Catalog aggregates suite and tool metadata required by training pipelines.
type Catalog struct {
	Suites []ServiceSuiteInfo
	Tools  []ToolInfo
}

// Fetch retrieves the current Agent SDK catalog via Arrow Flight.
// Note: This is a stub implementation that returns an empty catalog.
// The actual AgentSDK dependency has been replaced with local stubs.
func Fetch(ctx context.Context, addr string) (Catalog, error) {
	client, err := Dial(ctx, addr)
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

	return Catalog{Suites: suites, Tools: tools}, nil
}
