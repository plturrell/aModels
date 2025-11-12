package connectors

import (
	"context"

	"github.com/plturrell/aModels/services/graph/pkg/connectors"
)

type MurexAdapter struct{}

func NewMurexAdapter(cfg map[string]any, logger any) *MurexAdapter { return &MurexAdapter{} }

func (a *MurexAdapter) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (a *MurexAdapter) Close() error { return nil }
func (a *MurexAdapter) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return []map[string]any{}, nil }
func (a *MurexAdapter) DiscoverSchema(ctx context.Context) (*connectors.SourceSchema, error) { return &connectors.SourceSchema{}, nil }
