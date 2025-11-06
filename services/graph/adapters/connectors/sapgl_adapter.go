package connectors

import (
	"context"

	graph "github.com/plturrell/aModels/services/graph"
)

type SAPGLAdapter struct{}

func NewSAPGLAdapter(cfg map[string]any, logger any) *SAPGLAdapter { return &SAPGLAdapter{} }

func (a *SAPGLAdapter) Connect(ctx context.Context, opts map[string]any) error { return nil }
func (a *SAPGLAdapter) Close() error { return nil }
func (a *SAPGLAdapter) ExtractData(ctx context.Context, query map[string]any) ([]map[string]any, error) { return []map[string]any{}, nil }
func (a *SAPGLAdapter) DiscoverSchema(ctx context.Context) (*graph.SourceSchema, error) { return &graph.SourceSchema{}, nil }
