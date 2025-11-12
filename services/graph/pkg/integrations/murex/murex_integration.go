package murex

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/plturrell/aModels/services/graph/pkg/clients"
	"github.com/plturrell/aModels/services/graph/pkg/connectors"
	"github.com/plturrell/aModels/services/graph/pkg/models"
)

// MurexIntegration integrates Murex trading system with the knowledge graph.
type MurexIntegration struct {
	connector    connectors.Connector
	mapper       models.ModelMapper
	logger       *log.Logger
	graphClient  clients.GraphClient
}

// NewMurexIntegration creates a new Murex integration using an injected connector.
func NewMurexIntegration(conn connectors.Connector, mapper models.ModelMapper, graphClient clients.GraphClient, logger *log.Logger) *MurexIntegration {
	return &MurexIntegration{
		connector:   conn,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestTrades ingests trades from Murex and creates trade nodes in the knowledge graph.
func (mi *MurexIntegration) IngestTrades(ctx context.Context, filters map[string]interface{}) error {
	if mi.logger != nil {
		mi.logger.Printf("Ingesting trades from Murex (using connector)")
	}

	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	// Discover schema for logging if available
	if mi.logger != nil {
		if schema, err := mi.connector.DiscoverSchema(ctx); err == nil && schema != nil {
			mi.logger.Printf("Discovered Murex schema: %d tables", len(schema.Tables))
		}
	}

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

	var nodes []models.DomainNode
	var edges []models.DomainEdge

	for _, record := range data {
		record["source_system"] = "Murex"

		trade, err := mi.mapper.MapTrade(ctx, record)
		if err != nil {
			if mi.logger != nil {
				mi.logger.Printf("Warning: Failed to map trade: %v", err)
			}
			continue
		}

		graphNode := trade.ToGraphNode()
		nodes = append(nodes, *graphNode)

		if counterpartyID, ok := record["counterparty_id"].(string); ok && counterpartyID != "" {
			counterpartyNode := &models.DomainNode{
				ID:    fmt.Sprintf("counterparty-%s", counterpartyID),
				Type:  models.NodeTypeCounterparty,
				Label: fmt.Sprintf("Counterparty %s", counterpartyID),
				Properties: map[string]interface{}{
					"counterparty_id": counterpartyID,
					"source_system":   "Murex",
				},
			}
			nodes = append(nodes, *counterpartyNode)

			edge := &models.DomainEdge{
				SourceID:   trade.ID,
				TargetID:   counterpartyNode.ID,
				Type:       models.EdgeTypeHasCounterparty,
				Label:      "has counterparty",
				Properties: map[string]interface{}{},
			}
			edges = append(edges, *edge)
		}
	}

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
	if err := mi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Murex: %w", err)
	}
	defer mi.connector.Close()

	query := map[string]interface{}{
		"table": "cashflows",
		"limit": 1000,
	}
	for k, v := range filters { query[k] = v }

	data, err := mi.connector.ExtractData(ctx, query)
	if err != nil { return fmt.Errorf("failed to extract cashflows: %w", err) }

	var edges []models.DomainEdge
	for _, record := range data {
		if tradeID, ok := record["trade_id"].(string); ok {
			cashflowID := fmt.Sprintf("cashflow-%v", record["cashflow_id"])
			tradeNodeID := fmt.Sprintf("trade-%s", tradeID)
			edge := &models.DomainEdge{
				SourceID:   tradeNodeID,
				TargetID:   cashflowID,
				Type:       models.EdgeTypeTradesTo,
				Label:      "has cashflow",
				Properties: map[string]interface{}{
					"amount":       record["amount"],
					"currency":     record["currency"],
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

// IngestRegulatoryCalculations ingests regulatory calculations.
func (mi *MurexIntegration) IngestRegulatoryCalculations(ctx context.Context, filters map[string]interface{}) error {
	if mi.logger != nil { mi.logger.Printf("Ingesting regulatory calculations from Murex FMRP") }
	if err := mi.connector.Connect(ctx, nil); err != nil { return fmt.Errorf("failed to connect to Murex: %w", err) }
	defer mi.connector.Close()

	query := map[string]interface{}{
		"table": "regulatory_calculations",
		"limit": 1000,
	}
	for k, v := range filters { query[k] = v }

	data, err := mi.connector.ExtractData(ctx, query)
	if err != nil { return fmt.Errorf("failed to extract regulatory calculations: %w", err) }

	var nodes []models.DomainNode
	var edges []models.DomainEdge
	for _, record := range data {
		record["source_system"] = "Murex_FMRP"
		calc, err := mi.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil {
			if mi.logger != nil { mi.logger.Printf("Warning: Failed to map regulatory calculation: %v", err) }
			continue
		}
		graphNode := calc.ToGraphNode()
		nodes = append(nodes, *graphNode)
		if tradeID, ok := record["trade_id"].(string); ok {
			tradeNodeID := fmt.Sprintf("trade-%s", tradeID)
			edge := models.CreateTradeToCalculationEdge(tradeNodeID, calc.ID, map[string]interface{}{ "source_system": "Murex_FMRP" })
			edges = append(edges, *edge)
		}
	}

	if len(nodes) > 0 {
		if err := mi.graphClient.UpsertNodes(ctx, nodes); err != nil { return fmt.Errorf("failed to upsert regulatory calculation nodes: %w", err) }
		if mi.logger != nil { mi.logger.Printf("Upserted %d regulatory calculation nodes", len(nodes)) }
	}
	if len(edges) > 0 {
		if err := mi.graphClient.UpsertEdges(ctx, edges); err != nil { return fmt.Errorf("failed to upsert regulatory calculation edges: %w", err) }
		if mi.logger != nil { mi.logger.Printf("Upserted %d regulatory calculation edges", len(edges)) }
	}
	return nil
}

// SyncFullSync performs a full synchronization of Murex data to the knowledge graph.
func (mi *MurexIntegration) SyncFullSync(ctx context.Context) error {
	if mi.logger != nil { mi.logger.Printf("Starting full Murex synchronization") }
	if err := mi.connector.Connect(ctx, nil); err != nil { return fmt.Errorf("failed to connect to Murex: %w", err) }
	defer mi.connector.Close()
	if schema, err := mi.connector.DiscoverSchema(ctx); err == nil && schema != nil {
		if mi.logger != nil { mi.logger.Printf("Murex schema discovered: %d tables available", len(schema.Tables)) }
	}
	if err := mi.IngestTrades(ctx, nil); err != nil { return fmt.Errorf("failed to sync trades: %w", err) }
	if err := mi.IngestCashflows(ctx, nil); err != nil { return fmt.Errorf("failed to sync cashflows: %w", err) }
	if err := mi.IngestRegulatoryCalculations(ctx, nil); err != nil { return fmt.Errorf("failed to sync regulatory calculations: %w", err) }
	if mi.logger != nil { mi.logger.Printf("Full Murex synchronization completed") }
	return nil
}

// DiscoverSchema discovers the Murex API schema from connector.
func (mi *MurexIntegration) DiscoverSchema(ctx context.Context) (*connectors.SourceSchema, error) {
	if err := mi.connector.Connect(ctx, nil); err != nil { return nil, fmt.Errorf("failed to connect to Murex: %w", err) }
	defer mi.connector.Close()
	return mi.connector.DiscoverSchema(ctx)
}

