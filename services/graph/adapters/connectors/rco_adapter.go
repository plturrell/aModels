package connectors

import (
	"context"

	"github.com/plturrell/aModels/services/graph/pkg/connectors"
)

type RCOAdapter struct{}

func NewRCOAdapter(cfg map[string]any, logger any) *RCOAdapter { return &RCOAdapter{} }

func (a *RCOAdapter) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (a *RCOAdapter) Close() error { return nil }
func (a *RCOAdapter) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return []map[string]any{}, nil }
func (a *RCOAdapter) DiscoverSchema(ctx context.Context) (*connectors.SourceSchema, error) { return &connectors.SourceSchema{}, nil }
