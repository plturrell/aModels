package flightcatalog

import (
	"context"
	"fmt"
)

// Catalog mirrors the Agent SDK Flight datasets the search service cares about.
// Stub implementation since AgentSDK is not available in aModels repo
type Catalog struct {
	Suites []ServiceSuiteInfo
	Tools  []ToolInfo
}

// ServiceSuiteInfo represents a service suite (stub)
type ServiceSuiteInfo struct {
	Name           string
	ToolNames      []string
	ToolCount      int
	Implementation string
	Version        string
	AttachedAt     string
}

// ToolInfo represents a tool (stub)
type ToolInfo struct {
	Name        string
	Description string
}

// Fetch is disabled - AgentSDK dependency removed
func Fetch(ctx context.Context, addr string) (Catalog, error) {
	return Catalog{}, fmt.Errorf("AgentSDK dependency removed - catalog fetching not available")
}
