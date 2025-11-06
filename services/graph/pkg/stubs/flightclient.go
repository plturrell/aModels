// Package stubs provides stubs for missing agenticAiETH dependencies
// This replaces the missing agenticAiETH_layer4_AgentSDK/pkg/flightclient package
package stubs

import (
	"context"
)

// ServiceSuiteInfo represents a service suite
type ServiceSuiteInfo struct {
	Name        string `json:"name"`
	Description string `json:"description"`
	Version     string `json:"version"`
}

// ToolInfo represents a tool
type ToolInfo struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
}

// FlightClient is a stub for flight client
type FlightClient struct{}

// Dial creates a new flight client connection
func Dial(ctx context.Context, addr string) (*FlightClient, error) {
	return &FlightClient{}, nil
}

// Close closes the client connection
func (c *FlightClient) Close() error { return nil }

// ListServiceSuites lists available service suites
func (c *FlightClient) ListServiceSuites(ctx context.Context) ([]ServiceSuiteInfo, error) {
	return []ServiceSuiteInfo{}, nil
}

// ListTools lists available tools
func (c *FlightClient) ListTools(ctx context.Context) ([]ToolInfo, error) {
	return []ToolInfo{}, nil
}

