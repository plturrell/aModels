package clients

import (
	"context"

	"github.com/plturrell/aModels/services/graph/pkg/models"
)

// GraphClient interface for knowledge graph operations.
type GraphClient interface {
	UpsertNodes(ctx context.Context, nodes []models.DomainNode) error
	UpsertEdges(ctx context.Context, edges []models.DomainEdge) error
	Query(ctx context.Context, cypher string, params map[string]interface{}) ([]map[string]interface{}, error)
}

