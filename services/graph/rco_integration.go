package graph

import (
	"context"
	"fmt"
	"log"
)

// RCOIntegration integrates Regulatory Capital Operations with the knowledge graph.
type RCOIntegration struct {
	connector   Connector
	mapper      ModelMapper
	logger      *log.Logger
	graphClient GraphClient
}

// NewRCOIntegration creates a new RCO integration using an injected connector.
func NewRCOIntegration(conn Connector, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *RCOIntegration {
	return &RCOIntegration{
		connector:   conn,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestCapital ingests capital data from RCO.
func (ri *RCOIntegration) IngestCapital(ctx context.Context, filters map[string]any) error {
	if ri.logger != nil { ri.logger.Printf("Ingesting RCO capital data") }
	if err := ri.connector.Connect(ctx, nil); err != nil { return fmt.Errorf("connect RCO: %w", err) }
	defer ri.connector.Close()

	query := map[string]any{"table": "capital", "limit": 1000}
	for k, v := range filters { query[k] = v }

	data, err := ri.connector.ExtractData(ctx, query)
	if err != nil { return fmt.Errorf("extract RCO capital: %w", err) }

	var nodes []DomainNode
	for _, record := range data {
		record["source_system"] = "RCO"
		calc, err := ri.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil { if ri.logger != nil { ri.logger.Printf("map rco calc: %v", err) }; continue }
		nodes = append(nodes, *calc.ToGraphNode())
	}
	if len(nodes) > 0 {
		if err := ri.graphClient.UpsertNodes(ctx, nodes); err != nil { return fmt.Errorf("upsert rco nodes: %w", err) }
		if ri.logger != nil { ri.logger.Printf("Upserted %d RCO nodes", len(nodes)) }
	}
	return nil
}

