package sapgl

import (
	"context"
	"fmt"
	"log"
)

// SAPGLIntegration integrates SAP General Ledger with the knowledge graph.
type SAPGLIntegration struct {
	connector   Connector
	mapper      ModelMapper
	logger      *log.Logger
	graphClient GraphClient
}

// NewSAPGLIntegration creates a new SAP GL integration using an injected connector.
func NewSAPGLIntegration(conn Connector, mapper ModelMapper, graphClient GraphClient, logger *log.Logger) *SAPGLIntegration {
	return &SAPGLIntegration{
		connector:   conn,
		mapper:      mapper,
		logger:      logger,
		graphClient: graphClient,
	}
}

// IngestGL ingests general ledger entries.
func (si *SAPGLIntegration) IngestGL(ctx context.Context, filters map[string]any) error {
	if si.logger != nil { si.logger.Printf("Ingesting SAP GL entries") }
	if err := si.connector.Connect(ctx, nil); err != nil { return fmt.Errorf("connect SAP GL: %w", err) }
	defer si.connector.Close()

	query := map[string]any{"table": "gl_entries", "limit": 1000}
	for k, v := range filters { query[k] = v }

	data, err := si.connector.ExtractData(ctx, query)
	if err != nil { return fmt.Errorf("extract SAP GL entries: %w", err) }

	var nodes []DomainNode
	for _, record := range data {
		record["source_system"] = "SAP_GL"
		entry, err := si.mapper.MapLedgerEntry(ctx, record)
		if err != nil { if si.logger != nil { si.logger.Printf("map ledger entry: %v", err) }; continue }
		node := entry.ToGraphNode()
		nodes = append(nodes, *node)
	}
	if len(nodes) > 0 {
		if err := si.graphClient.UpsertNodes(ctx, nodes); err != nil { return fmt.Errorf("upsert gl nodes: %w", err) }
		if si.logger != nil { si.logger.Printf("Upserted %d GL nodes", len(nodes)) }
	}
	return nil
}

