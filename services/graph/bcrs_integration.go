package graph

import (
	"context"
	"fmt"
	"log"
)

// BCRSIntegration integrates Banking Credit Risk System with the knowledge graph.
type BCRSIntegration struct {
	connector   Connector
	mapper      ModelMapper
	logger      *log.Logger
	graphClient GraphClient
}

// NewBCRSIntegration creates a new BCRS integration using an injected connector.
func NewBCRSIntegration(conn Connector, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *BCRSIntegration {
	return &BCRSIntegration{
		connector:   conn,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestCreditRisk ingests credit risk data from BCRS.
func (bi *BCRSIntegration) IngestCreditRisk(ctx context.Context, filters map[string]any) error {
	if bi.logger != nil { bi.logger.Printf("Ingesting BCRS credit risk data") }
	if err := bi.connector.Connect(ctx, nil); err != nil { return fmt.Errorf("connect BCRS: %w", err) }
	defer bi.connector.Close()

	query := map[string]any{"table": "credit_risk", "limit": 1000}
	for k, v := range filters { query[k] = v }

	data, err := bi.connector.ExtractData(ctx, query)
	if err != nil { return fmt.Errorf("extract BCRS credit risk: %w", err) }

	var nodes []DomainNode
	for _, record := range data {
		record["source_system"] = "BCRS"
		// MapCreditRisk not available in ModelMapper interface - create node directly
		node := DomainNode{
			ID:    fmt.Sprintf("bcrs-credit-risk-%v", record["exposure_id"]),
			Type:  "CreditRisk",
			Label: "CreditRisk",
			Properties: record,
		}
		nodes = append(nodes, node)
	}
	if len(nodes) > 0 {
		if err := bi.graphClient.UpsertNodes(ctx, nodes); err != nil { return fmt.Errorf("upsert bcrs nodes: %w", err) }
		if bi.logger != nil { bi.logger.Printf("Upserted %d BCRS nodes", len(nodes)) }
	}
	return nil
}

