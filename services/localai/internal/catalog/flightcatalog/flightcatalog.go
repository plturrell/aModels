package flightcatalog

import (
	"context"
	"fmt"
)

// Catalog mirrors the Arrow Flight datasets exposed by the Agent SDK.
// This is a stub implementation since AgentSDK is not available in aModels repo.
type Catalog struct {
	// Stub structure - AgentSDK dependency removed
}

// Fetch is disabled - AgentSDK dependency removed
func Fetch(ctx context.Context, addr string) (Catalog, error) {
	return Catalog{}, fmt.Errorf("AgentSDK dependency removed - catalog fetching not available")
}
