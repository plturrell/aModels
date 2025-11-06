package graph

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// BCRSIntegration integrates Banking Credit Risk System with the knowledge graph.
type BCRSIntegration struct {
	connector     *connectors.BCRSConnector
	mapper        ModelMapper
	logger        *log.Logger
	graphClient   GraphClient
}

// NewBCRSIntegration creates a new BCRS integration.
func NewBCRSIntegration(config map[string]interface{}, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *BCRSIntegration {
	connector := connectors.NewBCRSConnector(config, logger)
	return &BCRSIntegration{
		connector:   connector,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestCreditExposures ingests credit exposures from BCRS.
func (bi *BCRSIntegration) IngestCreditExposures(ctx context.Context, filters map[string]interface{}) error {
	if bi.logger != nil {
		bi.logger.Printf("Ingesting credit exposures from BCRS")
	}

	// Connect to BCRS
	if err := bi.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to BCRS: %w", err)
	}
	defer bi.connector.Close()

	// Extract credit exposure data
	query := map[string]interface{}{
		"table": "credit_exposures",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := bi.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract credit exposures: %w", err)
	}

	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		// Add source system identifier
		record["source_system"] = "BCRS"
		record["calculation_type"] = "CreditRisk"

		// Map to RegulatoryCalculation domain model
		calc, err := bi.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil {
			if bi.logger != nil {
				bi.logger.Printf("Warning: Failed to map credit exposure: %v", err)
			}
			continue
		}

		// Convert to graph node
		graphNode := calc.ToGraphNode()
		nodes = append(nodes, *graphNode)

		// Link to counterparty if counterparty_id exists
		if counterpartyID, ok := record["counterparty_id"].(string); ok {
			counterpartyNodeID := fmt.Sprintf("counterparty-%s", counterpartyID)
			edge := &DomainEdge{
				SourceID:   counterpartyNodeID,
				TargetID:  calc.ID,
				Type:       EdgeTypeCalculatesRisk,
				Label:      "has credit exposure",
				Properties: map[string]interface{}{
					"source_system": "BCRS",
				},
			}
			edges = append(edges, *edge)
		}
	}

	// Upsert to knowledge graph
	if len(nodes) > 0 {
		if err := bi.graphClient.UpsertNodes(ctx, nodes); err != nil {
			return fmt.Errorf("failed to upsert credit exposure nodes: %w", err)
		}
		if bi.logger != nil {
			bi.logger.Printf("Upserted %d credit exposure nodes", len(nodes))
		}
	}

	if len(edges) > 0 {
		if err := bi.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert credit exposure edges: %w", err)
		}
		if bi.logger != nil {
			bi.logger.Printf("Upserted %d credit exposure edges", len(edges))
		}
	}

	return nil
}

// SyncFullSync performs a full synchronization of BCRS data to the knowledge graph.
func (bi *BCRSIntegration) SyncFullSync(ctx context.Context) error {
	if bi.logger != nil {
		bi.logger.Printf("Starting full BCRS synchronization")
	}

	if err := bi.IngestCreditExposures(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync credit exposures: %w", err)
	}

	if bi.logger != nil {
		bi.logger.Printf("Full BCRS synchronization completed")
	}

	return nil
}

