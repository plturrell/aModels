package connectors

import (
	"context"

	"github.com/plturrell/aModels/services/graph/pkg/connectors"
)

type AxiomAdapter struct{}

func NewAxiomAdapter(cfg map[string]any, logger any) *AxiomAdapter { return &AxiomAdapter{} }

func (a *AxiomAdapter) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (a *AxiomAdapter) Close() error { return nil }
func (a *AxiomAdapter) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return []map[string]any{}, nil }
func (a *AxiomAdapter) DiscoverSchema(ctx context.Context) (*connectors.SourceSchema, error) { return &connectors.SourceSchema{}, nil }
