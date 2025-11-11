package graph

import (
	"context"
	"log"
	"os"
)

// ExampleMurexIntegrationSetup demonstrates how to set up Murex integration with OpenAPI support.
func ExampleMurexIntegrationSetup() {
	logger := log.New(os.Stdout, "[MurexIntegration] ", log.LstdFlags)

	// Configure Murex integration with OpenAPI support
	config := map[string]interface{}{
		// Murex API configuration
		"base_url": getEnvOrDefault("MUREX_API_BASE_URL", "https://api.murex.com"),
		"api_key":  getEnvOrDefault("MUREX_API_KEY", ""), // Should be set via environment variable

		// OpenAPI specification (choose one):
		// Option 1: Load from GitHub repository
		"openapi_spec_url": getEnvOrDefault("MUREX_OPENAPI_SPEC_URL",
			"https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml"),

		// Option 2: Load from local file (alternative to URL)
		// "openapi_spec_path": "/path/to/murex-openapi.yaml",
	}

	// Create domain model mapper
	mapper := NewDefaultModelMapper()

	// Create graph client (implementation depends on your Neo4j setup)
	// This is a placeholder - you would use your actual graph client implementation
	var graphClient GraphClient // = your Neo4jGraphClient instance

	// Initialize Murex integration
	murexIntegration := NewMurexIntegration(config, mapper, graphClient, logger)

	// Example: Discover schema from OpenAPI spec
	ctx := context.Background()
	schema, err := murexIntegration.DiscoverSchema(ctx)
	if err != nil {
		logger.Printf("Failed to discover schema: %v", err)
	} else {
		logger.Printf("Successfully discovered schema: %d tables", len(schema.Tables))
		logger.Printf("API version: %v", schema.Metadata["version"])
	}

	// Example: Ingest trades with filters
	filters := map[string]interface{}{
		"trade_date_from": "2024-01-01",
		"trade_date_to":   "2024-12-31",
		"status":           "Executed",
	}
	if err := murexIntegration.IngestTrades(ctx, filters); err != nil {
		logger.Printf("Failed to ingest trades: %v", err)
	}

	// Example: Full synchronization
	if err := murexIntegration.SyncFullSync(ctx); err != nil {
		logger.Printf("Failed to sync: %v", err)
	}
}

// Helper function to get environment variable or default value
func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

