package connectors

import (
	"context"

	graph "github.com/plturrell/aModels/services/graph"
)

type BCRSAdapter struct{}

func NewBCRSAdapter(cfg map[string]any, logger any) *BCRSAdapter { return &BCRSAdapter{} }

func (a *BCRSAdapter) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (a *BCRSAdapter) Close() error { return nil }
func (a *BCRSAdapter) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return []map[string]any{}, nil }
func (a *BCRSAdapter) DiscoverSchema(ctx context.Context) (*graph.SourceSchema, error) { return &graph.SourceSchema{}, nil }
