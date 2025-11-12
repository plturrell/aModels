package axiom

import (
	"context"
	"fmt"
	"log"
)

// AxiomIntegration integrates Axiom risk management system with the knowledge graph.
type AxiomIntegration struct {
	connector   Connector
	mapper      ModelMapper
	logger      *log.Logger
	graphClient GraphClient
}

// NewAxiomIntegration creates a new Axiom integration using an injected connector.
func NewAxiomIntegration(conn Connector, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *AxiomIntegration {
	return &AxiomIntegration{
		connector:   conn,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestRiskMetrics ingests risk metrics from Axiom.
func (ai *AxiomIntegration) IngestRiskMetrics(ctx context.Context, filters map[string]interface{}) error {
	if ai.logger != nil {
		ai.logger.Printf("Ingesting risk metrics from Axiom")
	}

	if err := ai.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to Axiom: %w", err)
	}
	defer ai.connector.Close()

	query := map[string]interface{}{
		"table": "risk_metrics",
		"limit": 1000,
	}
	for k, v := range filters { query[k] = v }

	data, err := ai.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract risk metrics: %w", err)
	}

	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		record["source_system"] = "Axiom"
		record["calculation_type"] = "Risk"

		calc, err := ai.mapper.MapRegulatoryCalculation(ctx, record)
		if err != nil {
			if ai.logger != nil { ai.logger.Printf("Warning: Failed to map risk metric: %v", err) }
			continue
		}

		graphNode := calc.ToGraphNode()
		nodes = append(nodes, *graphNode)

		if riskType, ok := record["risk_type"].(string); ok {
			riskFactorNode := &DomainNode{
				ID:    fmt.Sprintf("risk-factor-%s", riskType),
				Type:  NodeTypeRiskFactor,
				Label: fmt.Sprintf("Risk Factor %s", riskType),
				Properties: map[string]interface{}{
					"risk_type":     riskType,
					"source_system": "Axiom",
				},
			}
			nodes = append(nodes, *riskFactorNode)

			edge := &DomainEdge{
				SourceID:   calc.ID,
				TargetID:   riskFactorNode.ID,
				Type:       EdgeTypeCalculatesRisk,
				Label:      "calculates risk",
				Properties: map[string]interface{}{ "source_system": "Axiom" },
			}
			edges = append(edges, *edge)
		}
	}

	if len(nodes) > 0 {
		if err := ai.graphClient.UpsertNodes(ctx, nodes); err != nil { return fmt.Errorf("failed to upsert risk metric nodes: %w", err) }
		if ai.logger != nil { ai.logger.Printf("Upserted %d risk metric nodes", len(nodes)) }
	}
	if len(edges) > 0 {
		if err := ai.graphClient.UpsertEdges(ctx, edges); err != nil { return fmt.Errorf("failed to upsert risk metric edges: %w", err) }
		if ai.logger != nil { ai.logger.Printf("Upserted %d risk metric edges", len(edges)) }
	}
	return nil
}

// SyncFullSync performs a full synchronization of Axiom data to the knowledge graph.
func (ai *AxiomIntegration) SyncFullSync(ctx context.Context) error {
	if ai.logger != nil { ai.logger.Printf("Starting full Axiom synchronization") }
	if err := ai.IngestRiskMetrics(ctx, nil); err != nil { return fmt.Errorf("failed to sync risk metrics: %w", err) }
	if ai.logger != nil { ai.logger.Printf("Full Axiom synchronization completed") }
	return nil
}

