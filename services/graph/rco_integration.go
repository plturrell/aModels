package graph

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// RCOIntegration integrates Regulatory Capital Operations with the knowledge graph.
type RCOIntegration struct {
	connector     *connectors.RCOConnector
	mapper        ModelMapper
	logger        *log.Logger
	graphClient   GraphClient
}

// NewRCOIntegration creates a new RCO integration.
func NewRCOIntegration(config map[string]interface{}, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *RCOIntegration {
	connector := connectors.NewRCOConnector(config, logger)
	return &RCOIntegration{
		connector:   connector,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestCapitalCalculations ingests capital calculations from RCO.
func (ri *RCOIntegration) IngestCapitalCalculations(ctx context.Context, filters map[string]interface{}) error {
	if ri.logger != nil {
		ri.logger.Printf("Ingesting capital calculations from RCO")
	}

	// Connect to RCO
	if err := ri.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to RCO: %w", err)
	}
	defer ri.connector.Close()

	// Extract capital calculation data
	query := map[string]interface{}{
		"table": "capital_calculations",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := ri.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract capital calculations: %w", err)
	}

	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		// Add source system identifier
		record["source_system"] = "RCO"
		record["calculation_type"] = "Capital"
		record["regulatory_framework"] = "Basel III"

		// Map to RegulatoryCalculation domain model
		calc, err := ri.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil {
			if ri.logger != nil {
				ri.logger.Printf("Warning: Failed to map capital calculation: %v", err)
			}
			continue
		}

		// Convert to graph node
		graphNode := calc.ToGraphNode()
		nodes = append(nodes, *graphNode)
	}

	// Upsert to knowledge graph
	if len(nodes) > 0 {
		if err := ri.graphClient.UpsertNodes(ctx, nodes); err != nil {
			return fmt.Errorf("failed to upsert capital calculation nodes: %w", err)
		}
		if ri.logger != nil {
			ri.logger.Printf("Upserted %d capital calculation nodes", len(nodes))
		}
	}

	if len(edges) > 0 {
		if err := ri.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert capital calculation edges: %w", err)
		}
		if ri.logger != nil {
			ri.logger.Printf("Upserted %d capital calculation edges", len(edges))
		}
	}

	return nil
}

// SyncFullSync performs a full synchronization of RCO data to the knowledge graph.
func (ri *RCOIntegration) SyncFullSync(ctx context.Context) error {
	if ri.logger != nil {
		ri.logger.Printf("Starting full RCO synchronization")
	}

	if err := ri.IngestCapitalCalculations(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync capital calculations: %w", err)
	}

	if ri.logger != nil {
		ri.logger.Printf("Full RCO synchronization completed")
	}

	return nil
}

