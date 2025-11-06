package graph

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/orchestration/agents"
	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// MurexIntegration integrates Murex trading system with the knowledge graph.
type MurexIntegration struct {
	connector     *connectors.MurexConnector
	mapper        ModelMapper
	logger        *log.Logger
	graphClient   GraphClient
}

// GraphClient interface for knowledge graph operations.
type GraphClient interface {
	UpsertNodes(ctx context.Context, nodes []DomainNode) error
	UpsertEdges(ctx context.Context, edges []DomainEdge) error
	Query(ctx context.Context, cypher string, params map[string]interface{}) ([]map[string]interface{}, error)
}

// NewMurexIntegration creates a new Murex integration with OpenAPI support.
// Config should include:
//   - base_url: Murex API base URL (e.g., "https://api.murex.com")
//   - api_key: API key for authentication (required for production)
//   - openapi_spec_url: URL to Murex OpenAPI spec (optional, e.g., from GitHub)
//     Default: "https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml"
//   - openapi_spec_path: Local path to OpenAPI spec file (optional, alternative to URL)
//
// Example usage:
//   config := map[string]interface{}{
//       "base_url": "https://api.murex.com",
//       "api_key": "your-api-key",
//       "openapi_spec_url": "https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml",
//   }
//   integration := NewMurexIntegration(config, mapper, graphClient, logger)
func NewMurexIntegration(config map[string]interface{}, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *MurexIntegration {
	// Initialize config if nil
	if config == nil {
		config = make(map[string]interface{})
	}

	// Set default OpenAPI spec URL if not provided (from Murex GitHub repository)
	if _, ok := config["openapi_spec_url"]; !ok {
		if _, hasPath := config["openapi_spec_path"]; !hasPath {
			// Default to Murex OpenAPI spec from GitHub if neither URL nor path is specified
			config["openapi_spec_url"] = "https://raw.githubusercontent.com/mxenabled/openapi/master/openapi/trades.yaml"
			if logger != nil {
				logger.Printf("Using default Murex OpenAPI spec URL: %s", config["openapi_spec_url"])
			}
		}
	}

	// Set default base URL if not provided
	if _, ok := config["base_url"]; !ok {
		config["base_url"] = "https://api.murex.com"
		if logger != nil {
			logger.Printf("Using default Murex API base URL: %s", config["base_url"])
		}
	}

	connector := connectors.NewMurexConnector(config, logger)
	return &MurexIntegration{
		connector:   connector,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestTrades ingests trades from Murex and creates trade nodes in the knowledge graph.
func (mi *MurexIntegration) IngestTrades(ctx context.Context, filters map[string]interface{}) error {
	if mi.logger != nil {
		mi.logger.Printf("Ingesting trades from Murex (using OpenAPI-enabled connector)")
	}

	// Connect to Murex (OpenAPI connector will handle authentication)
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	// Discover schema from OpenAPI spec if not already done
	if mi.logger != nil {
		schema, err := mi.connector.DiscoverSchema(ctx)
		if err == nil && schema != nil {
			mi.logger.Printf("Discovered Murex schema: %d tables, version: %v", 
				len(schema.Tables), schema.Metadata["version"])
		}
	}

	// Extract trade data
	query := map[string]interface{}{
		"table": "trades",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := mi.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract trades: %w", err)
	}

	// Map to domain model and create nodes
	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		// Add source system identifier
		record["source_system"] = "Murex"

		// Map to Trade domain model
		trade, err := mi.mapper.MapTrade(ctx, record)
		if err != nil {
			if mi.logger != nil {
				mi.logger.Printf("Warning: Failed to map trade: %v", err)
			}
			continue
		}

		// Convert to graph node
		graphNode := trade.ToGraphNode()
		nodes = append(nodes, *graphNode)

		// Create relationships if counterparty exists
		if counterpartyID, ok := record["counterparty_id"].(string); ok && counterpartyID != "" {
			counterpartyNode := &DomainNode{
				ID:    fmt.Sprintf("counterparty-%s", counterpartyID),
				Type:  NodeTypeCounterparty,
				Label: fmt.Sprintf("Counterparty %s", counterpartyID),
				Properties: map[string]interface{}{
					"counterparty_id": counterpartyID,
					"source_system":    "Murex",
				},
			}
			nodes = append(nodes, *counterpartyNode)

			edge := &DomainEdge{
				SourceID:   trade.ID,
				TargetID:  counterpartyNode.ID,
				Type:       EdgeTypeHasCounterparty,
				Label:      "has counterparty",
				Properties: map[string]interface{}{},
			}
			edges = append(edges, *edge)
		}
	}

	// Upsert to knowledge graph
	if len(nodes) > 0 {
		if err := mi.graphClient.UpsertNodes(ctx, nodes); err != nil {
			return fmt.Errorf("failed to upsert trade nodes: %w", err)
		}
		if mi.logger != nil {
			mi.logger.Printf("Upserted %d trade nodes", len(nodes))
		}
	}

	if len(edges) > 0 {
		if err := mi.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert trade edges: %w", err)
		}
		if mi.logger != nil {
			mi.logger.Printf("Upserted %d trade edges", len(edges))
		}
	}

	return nil
}

// IngestCashflows ingests cashflows from Murex and links them to trades.
func (mi *MurexIntegration) IngestCashflows(ctx context.Context, filters map[string]interface{}) error {
	if mi.logger != nil {
		mi.logger.Printf("Ingesting cashflows from Murex")
	}

	// Connect to Murex
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	// Extract cashflow data
	query := map[string]interface{}{
		"table": "cashflows",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := mi.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract cashflows: %w", err)
	}

	var edges []DomainEdge

	for _, record := range data {
		// Link cashflow to trade
		if tradeID, ok := record["trade_id"].(string); ok {
			cashflowID := fmt.Sprintf("cashflow-%v", record["cashflow_id"])
			tradeNodeID := fmt.Sprintf("trade-%s", tradeID)

			edge := &DomainEdge{
				SourceID:   tradeNodeID,
				TargetID:  cashflowID,
				Type:       EdgeTypeTradesTo,
				Label:      "has cashflow",
				Properties: map[string]interface{}{
					"amount":   record["amount"],
					"currency": record["currency"],
					"source_system": "Murex",
				},
			}
			edges = append(edges, *edge)
		}
	}

	if len(edges) > 0 {
		if err := mi.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert cashflow edges: %w", err)
		}
		if mi.logger != nil {
			mi.logger.Printf("Upserted %d cashflow edges", len(edges))
		}
	}

	return nil
}

// IngestRegulatoryCalculations ingests regulatory calculations from Murex FMRP.
func (mi *MurexIntegration) IngestRegulatoryCalculations(ctx context.Context, filters map[string]interface{}) error {
	if mi.logger != nil {
		mi.logger.Printf("Ingesting regulatory calculations from Murex FMRP")
	}

	// Connect to Murex
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	// Extract regulatory calculation data
	query := map[string]interface{}{
		"table": "regulatory_calculations",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := mi.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract regulatory calculations: %w", err)
	}

	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		// Add source system identifier
		record["source_system"] = "Murex_FMRP"

		// Map to RegulatoryCalculation domain model
		calc, err := mi.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil {
			if mi.logger != nil {
				mi.logger.Printf("Warning: Failed to map regulatory calculation: %v", err)
			}
			continue
		}

		// Convert to graph node
		graphNode := calc.ToGraphNode()
		nodes = append(nodes, *graphNode)

		// Link to trade if trade_id exists
		if tradeID, ok := record["trade_id"].(string); ok {
			tradeNodeID := fmt.Sprintf("trade-%s", tradeID)
			edge := CreateTradeToCalculationEdge(tradeNodeID, calc.ID, map[string]interface{}{
				"source_system": "Murex_FMRP",
			})
			edges = append(edges, *edge)
		}
	}

	// Upsert to knowledge graph
	if len(nodes) > 0 {
		if err := mi.graphClient.UpsertNodes(ctx, nodes); err != nil {
			return fmt.Errorf("failed to upsert regulatory calculation nodes: %w", err)
		}
		if mi.logger != nil {
			mi.logger.Printf("Upserted %d regulatory calculation nodes", len(nodes))
		}
	}

	if len(edges) > 0 {
		if err := mi.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert regulatory calculation edges: %w", err)
		}
		if mi.logger != nil {
			mi.logger.Printf("Upserted %d regulatory calculation edges", len(edges))
		}
	}

	return nil
}

// SyncFullSync performs a full synchronization of Murex data to the knowledge graph.
func (mi *MurexIntegration) SyncFullSync(ctx context.Context) error {
	if mi.logger != nil {
		mi.logger.Printf("Starting full Murex synchronization (using OpenAPI endpoints)")
	}

	// Connect once and reuse for all sync operations
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	// Discover and log schema information
	if schema, err := mi.connector.DiscoverSchema(ctx); err == nil && schema != nil {
		if mi.logger != nil {
			mi.logger.Printf("Murex schema discovered: %d tables available", len(schema.Tables))
		}
	}

	// Sync in order: trades, cashflows, regulatory calculations
	if err := mi.IngestTrades(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync trades: %w", err)
	}

	if err := mi.IngestCashflows(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync cashflows: %w", err)
	}

	if err := mi.IngestRegulatoryCalculations(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync regulatory calculations: %w", err)
	}

	if mi.logger != nil {
		mi.logger.Printf("Full Murex synchronization completed")
	}

	return nil
}

// DiscoverSchema discovers the Murex API schema from OpenAPI specification.
func (mi *MurexIntegration) DiscoverSchema(ctx context.Context) (*agents.SourceSchema, error) {
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return nil, fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	return mi.connector.DiscoverSchema(ctx)
}

