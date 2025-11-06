package graph

import (
	"context"
	"fmt"
	"log"

	"github.com/plturrell/aModels/services/orchestration/agents/connectors"
)

// SAPGLIntegration integrates SAP General Ledger with the knowledge graph.
type SAPGLIntegration struct {
	connector     *connectors.SAPGLConnector
	mapper        ModelMapper
	logger        *log.Logger
	graphClient   GraphClient
}

// NewSAPGLIntegration creates a new SAP GL integration.
func NewSAPGLIntegration(config map[string]interface{}, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *SAPGLIntegration {
	connector := connectors.NewSAPGLConnector(config, logger)
	return &SAPGLIntegration{
		connector:   connector,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestJournalEntries ingests journal entries from SAP GL and creates journal entry nodes.
func (si *SAPGLIntegration) IngestJournalEntries(ctx context.Context, filters map[string]interface{}) error {
	if si.logger != nil {
		si.logger.Printf("Ingesting journal entries from SAP GL")
	}

	// Connect to SAP GL
	if err := si.connector.Connect(ctx, nil); err != nil {
		return fmt.Errorf("failed to connect to SAP GL: %w", err)
	}
	defer si.connector.Close()

	// Extract journal entry data
	query := map[string]interface{}{
		"table": "journal_entries",
		"limit": 1000,
	}
	for k, v := range filters {
		query[k] = v
	}

	data, err := si.connector.ExtractData(ctx, query)
	if err != nil {
		return fmt.Errorf("failed to extract journal entries: %w", err)
	}

	var nodes []DomainNode
	var edges []DomainEdge

	for _, record := range data {
		// Add source system identifier
		record["source_system"] = "SAP_GL"

		// Map to JournalEntry domain model
		entry, err := si.mapper.MapJournalEntry(ctx, record)
		if err != nil {
			if si.logger != nil {
				si.logger.Printf("Warning: Failed to map journal entry: %v", err)
			}
			continue
		}

		// Convert to graph node
		graphNode := entry.ToGraphNode()
		nodes = append(nodes, *graphNode)

		// Link to trade if trade_id exists
		if tradeID, ok := record["trade_id"].(string); ok {
			tradeNodeID := fmt.Sprintf("trade-%s", tradeID)
			edge := CreateTradeToJournalEntryEdge(tradeNodeID, entry.ID, map[string]interface{}{
				"source_system": "SAP_GL",
			})
			edges = append(edges, *edge)
		}
	}

	// Upsert to knowledge graph
	if len(nodes) > 0 {
		if err := si.graphClient.UpsertNodes(ctx, nodes); err != nil {
			return fmt.Errorf("failed to upsert journal entry nodes: %w", err)
		}
		if si.logger != nil {
			si.logger.Printf("Upserted %d journal entry nodes", len(nodes))
		}
	}

	if len(edges) > 0 {
		if err := si.graphClient.UpsertEdges(ctx, edges); err != nil {
			return fmt.Errorf("failed to upsert journal entry edges: %w", err)
		}
		if si.logger != nil {
			si.logger.Printf("Upserted %d journal entry edges", len(edges))
		}
	}

	return nil
}

// SyncFullSync performs a full synchronization of SAP GL data to the knowledge graph.
func (si *SAPGLIntegration) SyncFullSync(ctx context.Context) error {
	if si.logger != nil {
		si.logger.Printf("Starting full SAP GL synchronization")
	}

	if err := si.IngestJournalEntries(ctx, nil); err != nil {
		return fmt.Errorf("failed to sync journal entries: %w", err)
	}

	if si.logger != nil {
		si.logger.Printf("Full SAP GL synchronization completed")
	}

	return nil
}

